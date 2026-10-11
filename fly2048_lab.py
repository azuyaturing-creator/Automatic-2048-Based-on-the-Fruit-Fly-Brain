"""fly2048_lab.py — 网页互动台：实时观察果蝇大脑（Brian2 连接组子图）玩 2048。

脑回路复用 fly2048_core.py：
    16 棋盘格 → 16 个感觉输入（泊松放电）→ 连接组 LIF 子图
    → 4 方向读出层（唯一可学习部分，插上训练好的权重）。

页面 fly2048_lab.html 实时显示：
    棋盘动画 / 16 格感觉放电率热图 / 神经元放电栅格 / 4 方向概率
    / 因果链 / 学习曲线（若有训练日志）。

用法：
    python fly2048_lab.py [--port 8766]
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

STAGE9_PATH = Path(__file__).with_name("fly2048_core.py")
DEFAULT_HTML = Path(__file__).with_name("fly2048_lab.html")
CONSOLE_HTML = Path(__file__).with_name("train_console.html")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_SEED = 20261005
MAX_SPIKE_HISTORY_CYCLES = 12

# --------------------------------------------------------------------------
# 命名权重（profile）：一个权重 = 一个独立 out-dir（core 脚本以 --out-dir 写入，
# 权重 + 各自的练习/教学日志天然隔离）。
#   内置 profile：outputs/fly2048（"默认权重"，位置永不变）
#   用户 profile：outputs/fly2048/weights/<id>/，显示名记录在 registry.json
# --------------------------------------------------------------------------
DEFAULT_PROFILE_DIR = Path("outputs/fly2048")
DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "默认权重"
WEIGHTS_ROOT = DEFAULT_PROFILE_DIR / "weights"
REGISTRY_PATH = WEIGHTS_ROOT / "registry.json"
MAX_CONSOLE_LOG_LINES = 120

# 当前激活的 profile 目录（全局单选择；操作运行期间不允许切换）
ACTIVE_DIR: Path = DEFAULT_PROFILE_DIR
ACTIVE_ID: str = DEFAULT_PROFILE_ID
_REG_LOCK = threading.Lock()


# --- profile 内常用路径（全部跟随 ACTIVE_DIR） ---
def p_state() -> Path:
    return ACTIVE_DIR / "train_state.json"


def p_live() -> Path:
    return ACTIVE_DIR / "live_episode.json"


def p_eval() -> Path:
    return ACTIVE_DIR / "eval_results.json"


def p_train_log() -> Path:
    return ACTIVE_DIR / "training_log.jsonl"


def p_teach_log() -> Path:
    return ACTIVE_DIR / "teach_log.jsonl"


def p_train_console() -> Path:
    return ACTIVE_DIR / "train_console.log"


def p_teach_console() -> Path:
    return ACTIVE_DIR / "teach_console.log"


def p_auto_console() -> Path:
    return ACTIVE_DIR / "auto_console.log"


def p_readout() -> Path:
    return ACTIVE_DIR / "readout.npz"


def _load_registry() -> dict[str, Any]:
    """读取用户权重注册表；损坏/缺失时返回空结构。"""
    if not REGISTRY_PATH.exists():
        return {"active_id": None, "entries": []}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            return {"active_id": None, "entries": []}
        data.setdefault("active_id", None)
        return data
    except (OSError, json.JSONDecodeError):
        return {"active_id": None, "entries": []}


def _save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(REGISTRY_PATH)


def _entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """单个 profile 的展示信息（含权重是否存在、练习局数）。"""
    pid = str(entry["id"])
    directory = (
        DEFAULT_PROFILE_DIR if pid == DEFAULT_PROFILE_ID
        else WEIGHTS_ROOT / pid
    )
    readout = directory / "readout.npz"
    train_log = directory / "training_log.jsonl"
    episodes = 0
    if train_log.exists():
        try:
            episodes = sum(1 for ln in train_log.read_text(encoding="utf-8").splitlines() if ln.strip())
        except OSError:
            episodes = 0
    return {
        "id": pid,
        "name": str(entry.get("name", pid)),
        "built_in": pid == DEFAULT_PROFILE_ID,
        "dir": str(directory),
        "weights_exists": readout.exists(),
        "episodes": episodes,
        "created_at": entry.get("created_at"),
    }


def weight_profiles() -> dict[str, Any]:
    """列出全部权重（内置 + 注册表），并标注当前激活项。"""
    with _REG_LOCK:
        registry = _load_registry()
    entries = [
        _entry_summary({"id": DEFAULT_PROFILE_ID, "name": DEFAULT_PROFILE_NAME})
    ]
    valid: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        info = _entry_summary(entry)
        # 目录/权重已不存在的条目过滤掉（外部误删时不报错）
        if (PROJECT_DIR / info["dir"]).exists():
            entries.append(info)
            valid.append(entry)
    active_id = ACTIVE_ID
    if not any(e["id"] == active_id for e in entries):
        active_id = DEFAULT_PROFILE_ID
    return {"active": active_id, "profiles": entries}


def _find_entry(registry: dict[str, Any], profile_id: str) -> dict[str, Any] | None:
    for entry in registry["entries"]:
        if str(entry.get("id")) == profile_id:
            return entry
    return None


def _profile_dir(profile_id: str) -> Path:
    if profile_id == DEFAULT_PROFILE_ID:
        return DEFAULT_PROFILE_DIR
    return WEIGHTS_ROOT / profile_id


def _validate_name(name: str, registry: dict[str, Any], exclude_id: str | None = None) -> str:
    cleaned = name.strip()
    if not cleaned or len(cleaned) > 24:
        raise ValueError("权重名称长度需在 1-24 个字符之间")
    if re.search(r'[\\/:*?"<>|\x00-\x1f]', cleaned):
        raise ValueError("名称不能包含 \\ / : * ? \" < > | 等特殊字符")
    for entry in registry["entries"]:
        if exclude_id is not None and str(entry.get("id")) == exclude_id:
            continue
        if str(entry.get("name", "")).strip() == cleaned:
            raise ValueError(f"已存在同名权重：{cleaned}")
    if cleaned == DEFAULT_PROFILE_NAME:
        raise ValueError("该名称为内置权重保留名")
    return cleaned


def _set_active(profile_id: str) -> Path:
    """切换激活 profile（调用方负责先确认无操作在运行），返回新目录。"""
    global ACTIVE_DIR, ACTIVE_ID
    if profile_id == DEFAULT_PROFILE_ID:
        directory = DEFAULT_PROFILE_DIR
    else:
        with _REG_LOCK:
            registry = _load_registry()
            if _find_entry(registry, profile_id) is None:
                raise ValueError(f"权重不存在：{profile_id}")
        directory = WEIGHTS_ROOT / profile_id
    if not (PROJECT_DIR / directory).exists():
        raise ValueError(f"权重目录已丢失：{directory}")
    ACTIVE_DIR = directory
    ACTIVE_ID = profile_id
    with _REG_LOCK:
        registry = _load_registry()
        registry["active_id"] = None if profile_id == DEFAULT_PROFILE_ID else profile_id
        _save_registry(registry)
    return directory


def create_weight(name: str, mode: str, n_features: int) -> dict[str, Any]:
    """新建命名权重。

    mode="copy"：把当前激活权重复制一份（"另存为"）；
    mode="random"：全新随机初始化。
    返回新 profile 的展示信息。
    """
    if mode not in ("copy", "random"):
        raise ValueError("mode 必须是 copy 或 random")
    with _REG_LOCK:
        registry = _load_registry()
        cleaned = _validate_name(name, registry)
        profile_id = "w_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + \
            os.urandom(2).hex()
        directory = WEIGHTS_ROOT / profile_id
        directory.mkdir(parents=True, exist_ok=False)
        if mode == "copy":
            source = p_readout()
            if not source.exists():
                shutil.rmtree(directory)
                raise ValueError("当前权重还未生成（无 readout.npz），无法另存")
            shutil.copy2(source, directory / "readout.npz")
        else:
            rng = np.random.default_rng()
            agent = stage9.ReadoutAgent(n_features, rng)
            agent.save(directory / "readout.npz")
        entry = {
            "id": profile_id,
            "name": cleaned,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        registry["entries"].append(entry)
        _save_registry(registry)
    return _entry_summary(entry)


def rename_weight(profile_id: str, new_name: str) -> dict[str, Any]:
    """重命名用户权重（只改注册表，目录 id 不变）。"""
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("内置默认权重不能重命名")
    with _REG_LOCK:
        registry = _load_registry()
        entry = _find_entry(registry, profile_id)
        if entry is None:
            raise ValueError(f"权重不存在：{profile_id}")
        entry["name"] = _validate_name(new_name, registry, exclude_id=profile_id)
        _save_registry(registry)
    return _entry_summary(entry)


def delete_weight(profile_id: str) -> None:
    """删除用户权重及其目录；正在激活的权重需先切换到其他权重。"""
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("内置默认权重不能删除")
    global ACTIVE_DIR, ACTIVE_ID
    with _REG_LOCK:
        registry = _load_registry()
        if _find_entry(registry, profile_id) is None:
            raise ValueError(f"权重不存在：{profile_id}")
        directory = WEIGHTS_ROOT / profile_id
        shutil.rmtree(directory, ignore_errors=False)
        registry["entries"] = [
            e for e in registry["entries"] if str(e.get("id")) != profile_id
        ]
        if registry.get("active_id") == profile_id:
            registry["active_id"] = None
        _save_registry(registry)
    if ACTIVE_ID == profile_id:
        ACTIVE_DIR = DEFAULT_PROFILE_DIR
        ACTIVE_ID = DEFAULT_PROFILE_ID


def restore_active_from_registry() -> None:
    """服务启动时恢复上次激活的权重。"""
    global ACTIVE_DIR, ACTIVE_ID
    registry = _load_registry()
    target = registry.get("active_id")
    if target and _find_entry(registry, str(target)) is not None:
        directory = WEIGHTS_ROOT / str(target)
        if (PROJECT_DIR / directory).exists():
            ACTIVE_DIR = directory
            ACTIVE_ID = str(target)


def active_profile_name() -> str:
    """当前激活 profile 的显示名（供快照/标签使用）。"""
    if ACTIVE_ID == DEFAULT_PROFILE_ID:
        return DEFAULT_PROFILE_NAME
    registry = _load_registry()
    entry = _find_entry(registry, ACTIVE_ID)
    return str(entry["name"]) if entry is not None else ACTIVE_ID


def open_console_log(path: Path):
    """以写模式打开控制台日志，带瞬时冲突重试。

    Windows 上 Python 读句柄的共享模式不含「允许写」：前端轮询日志时
    服务端正在 read_text，此刻 open("w") 会抛 PermissionError（Errno 13）。
    这是毫秒级窗口的偶发冲突，短暂重试即可，不必让练习/教学/循环启动失败。
    """
    last_err: PermissionError | None = None
    for _ in range(6):
        try:
            return path.open("w", encoding="utf-8")
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.15)
    raise PermissionError(
        f"日志文件被持续占用，无法写入：{path}（请关闭打开该文件的程序后重试）"
    ) from last_err


def load_stage9() -> Any:
    spec = importlib.util.spec_from_file_location("fly2048_core_mod", STAGE9_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {STAGE9_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage9 = load_stage9()


class Fly2048Lab:
    """一盘 2048 + 果蝇大脑，每次 step 跑一个 100ms 脑循环并 greedy 决策。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(args.seed)
        self.brain = stage9.FlyBrain(args.nodes, args.edges, args.seed)
        self.agent = stage9.ReadoutAgent(
            stage9.feature_dim(self.brain.n_neurons), self.rng
        )
        self.weights_loaded = False
        self.weights_path: Path | None = None
        self.loaded_mtime: float | None = None   # 记录加载时的 mtime 数值
        # 先恢复上次激活的命名权重，再加载其 readout（--weights 显式指定时优先）
        restore_active_from_registry()
        candidate = args.weights
        if candidate is None:
            candidate = p_readout()
        if candidate is not None and candidate.exists():
            self.agent.load(candidate)
            self.weights_loaded = True
            self.weights_path = candidate
            self.loaded_mtime = candidate.stat().st_mtime
        self.env = stage9.Game2048(self.rng)
        # 手动教学（人当老师）状态：人类每走一步，读出层就以该动作为标签做一次交叉熵
        self.teach_mode = False
        self.teach_aug_k = 1
        self.teach_lr = float(stage9.TEACH_LR)
        self.teach_transforms: list[tuple[Any, ...]] = []
        self.teach_aug_buffers: list[Any] = []
        self.teach_steps_total = 0
        self.teach_agreed_total = 0
        self.teach_games = 0
        self.teach_game_steps = 0
        self.teach_game_agreed = 0
        self.teach_human_action = -1
        self.teach_student_action = -1
        self.teach_last_agreed: bool | None = None
        self.reset_lab()

    def use_active_profile(self) -> None:
        """切换到当前激活 profile：加载其权重（无权重则随机初始化），并重置棋盘。"""
        candidate = p_readout()
        if candidate.exists():
            self.agent.load(candidate)
            self.weights_loaded = True
            self.weights_path = candidate
            self.loaded_mtime = candidate.stat().st_mtime
        else:
            # 目标 profile 尚未训练：内存中恢复成全新随机读出，避免沿用旧权重产生错觉
            rng = np.random.default_rng()
            self.agent = stage9.ReadoutAgent(
                stage9.feature_dim(self.brain.n_neurons), rng
            )
            self.weights_loaded = False
            self.weights_path = None
            self.loaded_mtime = None
        self.reset_lab()

    def reset_lab(self) -> dict[str, Any]:
        self.env.reset()
        self.rates = stage9.board_to_rates(self.env.board_flat)
        self.counts = np.zeros(self.brain.n_neurons, dtype=int)
        self.probs = np.full(stage9.N_ACTIONS, 0.25, dtype=float)
        self.logits = np.zeros(stage9.N_ACTIONS, dtype=float)
        self.action = -1
        self.moved = False
        self.gain = 0
        self.reward = 0.0
        self.net_reward = 0.0
        self.new_tiles: list[int] = []
        self.step_ms = 0.0
        self.history: list[list[int]] = []
        self.buffer = stage9.FeatureBuffer(self.brain.n_neurons)
        # 手动教学：每局新棋盘都要重建 K 路对称脑流的独立时间窗缓冲
        if self.teach_mode:
            self.teach_transforms = stage9.d4_subset(self.teach_aug_k)
            self.teach_aug_buffers = [
                stage9.FeatureBuffer(self.brain.n_neurons)
                for _ in self.teach_transforms
            ]
            self.teach_game_steps = 0
            self.teach_game_agreed = 0
            self.teach_human_action = -1
            self.teach_student_action = -1
            self.teach_last_agreed = None
        return self.snapshot()

    def step_lab(self) -> dict[str, Any]:
        if self.teach_mode:
            raise RuntimeError("手动教学进行中：AI 自动步已禁用，请用方向键教学或结束教学")
        if self.env.game_over():
            return self.snapshot()
        start = time.perf_counter()
        before = self.env.board_flat.copy()
        # 动作屏蔽：只在合法方向中 greedy，非法方向概率为零，互动台不再撞墙
        legal = self.env.legal_actions()
        self.rates = stage9.board_to_rates(self.env.board_flat)
        self.counts = self.brain.run_step(self.rates)
        features = self.buffer.push(self.counts, self.env.board_flat)
        self.logits = self.agent.logits(features)
        self.action, self.probs = self.agent.greedy(features, legal)
        self.moved, self.gain = self.env.move(self.action)
        # 奖惩展示（屏蔽下每步都合法）：
        #   合并得分 → +log2(合并值)；纯滑动无合并 → 0；非法（理论不出现）→ -1.0
        if not self.moved:
            self.reward = -1.0
        elif self.gain > 0:
            self.reward = round(float(np.log2(max(1.0, float(self.gain)))), 2)
        else:
            self.reward = 0.0
        self.net_reward = round(self.net_reward + self.reward, 2)
        after = self.env.board_flat
        self.new_tiles = [int(i) for i in range(16) if before[i] == 0 and after[i] != 0]
        self.step_ms = (time.perf_counter() - start) * 1000.0
        self.history.append([int(c) for c in self.counts])
        if len(self.history) > MAX_SPIKE_HISTORY_CYCLES:
            del self.history[: len(self.history) - MAX_SPIKE_HISTORY_CYCLES]
        return self.snapshot()

    # --- 手动教学（人当老师） ---
    def _teach_save(self) -> None:
        """把模仿人类后的权重写回当前激活 profile（并记住 mtime，防止误触发热加载）。"""
        path = p_readout()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
        self.weights_loaded = True
        self.weights_path = path
        try:
            self.loaded_mtime = path.stat().st_mtime
        except OSError:
            pass

    def _teach_finish_game(self) -> None:
        """一局人类教学结束：汇总写入 human_teach_log.jsonl（累计局数 +1）。"""
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": "human",
            "game": self.teach_games + 1,
            "score": int(self.env.score),
            "max_tile": int(self.env.max_tile),
            "moves": int(self.env.moves),
            "taught_steps": self.teach_game_steps,
            "agreed_steps": self.teach_game_agreed,
            "agreement": round(
                self.teach_game_agreed / max(1, self.teach_game_steps), 4
            ),
            "aug_k": self.teach_aug_k,
            "aug_updates": self.teach_game_steps * self.teach_aug_k,
        }
        log_path = p_readout().parent / "human_teach_log.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.teach_games += 1

    def teach_manual_start(
        self, aug_k: int = 1, lr: float | None = None
    ) -> dict[str, Any]:
        if self.teach_mode:
            return self.snapshot()
        if aug_k not in (1, 2, 4, 8):
            raise ValueError("aug_k 必须是 1/2/4/8")
        if lr is not None and not (0.0 < float(lr) <= 1.0):
            raise ValueError("教学学习率必须在 (0, 1] 之间")
        self.teach_mode = True
        self.teach_aug_k = int(aug_k)
        if lr is not None:
            self.teach_lr = float(lr)
        # 新一轮手动教学会话：累计统计清零，开一盘新棋
        self.teach_steps_total = 0
        self.teach_agreed_total = 0
        self.teach_games = 0
        self.reset_lab()
        return self.snapshot()

    def teach_manual_stop(self) -> dict[str, Any]:
        if not self.teach_mode:
            return self.snapshot()
        # 已结束的局在最后一步 teach_manual_step 里结算过，不能重复写日志
        if self.teach_game_steps > 0 and not self.env.game_over():
            self._teach_finish_game()
        self._teach_save()
        self.teach_mode = False
        self.reset_lab()
        return self.snapshot()

    def teach_manual_step(self, human_action: int) -> dict[str, Any]:
        """人类走一步：K 路对称呈现各跑一次脑，全部以人类动作（几何置换后）为标签。"""
        if not self.teach_mode:
            raise RuntimeError("手动教学未开启")
        human_action = int(human_action)
        if human_action not in (0, 1, 2, 3):
            raise ValueError("动作必须是 0(上)/1(右)/2(下)/3(左)")
        if self.env.game_over():
            return self.snapshot()
        start = time.perf_counter()
        before = self.env.board_flat.copy()
        legal = self.env.legal_actions()
        if not bool(legal[human_action]):
            raise ValueError("该方向当前不能滑动，请选择合法方向")

        # K 路对称脑流：第 0 路恒等（真实棋盘），一致率只看它
        ways: list[tuple[dict[str, Any], np.ndarray, int]] = []
        for idx, ((use_flip, rot_k, perm), buf) in enumerate(
            zip(self.teach_transforms, self.teach_aug_buffers)
        ):
            if idx == 0:
                tflat = self.env.board_flat
            else:
                tboard = stage9.transform_board_4x4(
                    self.env.board, use_flip, rot_k
                )
                tflat = tboard.reshape(-1)
            rates = stage9.board_to_rates(tflat)
            counts = self.brain.run_step(rates)
            features = buf.push(counts, tflat)
            tlegal = legal if idx == 0 else stage9.transform_legal(legal, perm)
            logits, _, cache = self.agent.forward(features)
            probs = self.agent.masked_softmax(logits, 1.0, tlegal)
            if idx == 0:
                self.rates = rates
                self.counts = counts
                self.logits = logits
                self.probs = probs
            ways.append((cache, probs, int(perm[human_action])))

        # 更新前学生自己的选择（只在合法方向内比较）
        student_action = int(np.argmax(ways[0][1]))
        agreed = student_action == human_action
        way_lr = self.teach_lr / float(len(self.teach_transforms))
        for cache, probs, taction in ways:
            self.agent.update(cache, taction, probs, 1.0, way_lr)

        # 棋盘按人类动作推进（合法，必然移动）
        self.moved, self.gain = self.env.move(human_action)
        self.reward = (
            round(float(np.log2(max(1.0, float(self.gain)))), 2)
            if self.gain > 0 else 0.0
        )
        self.net_reward = round(self.net_reward + self.reward, 2)
        after = self.env.board_flat
        self.new_tiles = [
            int(i) for i in range(16) if before[i] == 0 and after[i] != 0
        ]
        self.step_ms = (time.perf_counter() - start) * 1000.0
        self.history.append([int(c) for c in self.counts])
        if len(self.history) > MAX_SPIKE_HISTORY_CYCLES:
            del self.history[: len(self.history) - MAX_SPIKE_HISTORY_CYCLES]

        self.action = student_action
        self.teach_student_action = student_action
        self.teach_human_action = human_action
        self.teach_last_agreed = agreed
        self.teach_game_steps += 1
        self.teach_steps_total += 1
        if agreed:
            self.teach_game_agreed += 1
            self.teach_agreed_total += 1

        if self.teach_game_steps % 5 == 0 or self.env.game_over():
            self._teach_save()
        if self.env.game_over():
            self._teach_finish_game()
        return self.snapshot()

    def _reload_weights_if_newer(self) -> None:
        """当前激活 profile 的权重被训练进程更新后自动换上（按 mtime 数值判断）。

        必须比较"磁盘当前 mtime"与"上次加载时的 mtime 数值"；
        若只存路径再 stat，同一路径两边永远相等，热加载会失效。
        """
        candidate = p_readout()
        if not candidate.exists():
            return
        try:
            disk_mtime = candidate.stat().st_mtime
        except OSError:
            return
        is_newer = (
            self.loaded_mtime is None
            or disk_mtime > self.loaded_mtime + 1e-6
        )
        if is_newer:
            self.agent.load(candidate)
            self.weights_loaded = True
            self.weights_path = candidate
            self.loaded_mtime = disk_mtime

    def snapshot(self) -> dict[str, Any]:
        return {
            "ready": True,
            "board": [int(v) for v in self.env.board_flat],
            "score": int(self.env.score),
            "max_tile": int(self.env.max_tile),
            "moves": self.env.moves,
            "game_over": bool(self.env.game_over()),
            "new_tiles": self.new_tiles,
            "rates": [round(float(r), 1) for r in self.rates],
            "counts": [int(c) for c in self.counts],
            "n_neurons": int(self.brain.n_neurons),
            "center_index": int(self.brain.center_index),
            "input_indices": [int(i) for i in self.brain.input_indices],
            "probs": [round(float(p), 4) for p in self.probs],
            "logits": [round(float(v), 3) for v in self.logits],
            "action": int(self.action),
            "action_name": stage9.ACTION_NAMES[self.action] if self.action >= 0 else "",
            "moved": bool(self.moved),
            "gain": int(self.gain),
            "reward": float(self.reward),
            "net_reward": float(self.net_reward),
            "total_spikes": int(self.counts.sum()),
            "step_ms": round(self.step_ms, 1),
            "weights_loaded": bool(self.weights_loaded),
            "weights_name": active_profile_name(),
            "weights_path": str(self.weights_path) if self.weights_path is not None else None,
            "spike_history": self.history,
            "teach": self._teach_snapshot(),
        }

    def _teach_snapshot(self) -> dict[str, Any]:
        if not self.teach_mode:
            return {"active": False}
        return {
            "active": True,
            "aug_k": int(self.teach_aug_k),
            "lr": round(float(self.teach_lr), 4),
            "games": int(self.teach_games),
            "game_steps": int(self.teach_game_steps),
            "steps_total": int(self.teach_steps_total),
            "agreed_total": int(self.teach_agreed_total),
            "agreement": round(
                self.teach_agreed_total / max(1, self.teach_steps_total), 3
            ),
            "aug_updates_total": int(
                self.teach_steps_total * self.teach_aug_k
            ),
            "student_action": int(self.teach_student_action),
            "human_action": int(self.teach_human_action),
            "last_agreed": self.teach_last_agreed,
        }


def learning_series(log_path: Path | None) -> dict[str, Any]:
    """读取训练日志，供前端画学习曲线。"""
    if log_path is None:
        log_path = p_train_log() if p_train_log().exists() else None
    if log_path is None or not log_path.exists():
        return {"exists": False}
    episodes: list[int] = []
    scores: list[int] = []
    tiles: list[int] = []
    invalids: list[int] = []
    moves: list[int] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            episodes.append(int(record["episode"]))
            scores.append(int(record["score"]))
            tiles.append(int(record["max_tile"]))
            invalids.append(int(record.get("invalid", 0)))
            moves.append(int(record.get("moves", 0)))
    return {
        "exists": True,
        "path": str(log_path),
        "episodes": episodes,
        "scores": scores,
        "tiles": tiles,
        "invalids": invalids,
        "moves": moves,
    }


# --------------------------------------------------------------------------
# 训练控制台：状态读取 / 进程管理
# --------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """判断进程是否存活（Windows 用 OpenProcess/GetExitCodeProcess）。"""
    try:
        process_query_limited = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong(0)
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == still_active
    except (AttributeError, OSError):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def read_train_state() -> dict[str, Any] | None:
    if not p_state().exists():
        return None
    try:
        # utf-8-sig 兼容 PowerShell 写入的 BOM 头
        return json.loads(p_state().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def count_episodes() -> int:
    if not p_train_log().exists():
        return 0
    try:
        return sum(1 for _ in p_train_log().open(encoding="utf-8"))
    except OSError:
        return 0


def latest_record() -> dict[str, Any] | None:
    if not p_train_log().exists():
        return None
    try:
        lines = p_train_log().read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except (OSError, json.JSONDecodeError):
        return None


def read_live() -> dict[str, Any] | None:
    """读取训练实时状态（本局进行中的步数/得分），旧版训练进程没有此文件。"""
    if not p_live().exists():
        return None
    try:
        return json.loads(p_live().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def training_records() -> list[dict[str, Any]]:
    """训练日志全量记录（顺序 = 局号升序）。"""
    if not p_train_log().exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in p_train_log().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return records


def training_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从全量记录算汇总：近 25/50 局均分、历史最高、256/512 次数、非法率。"""
    if not records:
        return None
    scores = [float(r.get("score", 0)) for r in records]
    tiles = [int(r.get("max_tile", 0)) for r in records]
    inv = sum(int(r.get("invalid", 0)) for r in records)
    mov = sum(int(r.get("moves", 0)) for r in records)

    def mean_last(xs: list[float], k: int) -> float:
        tail = xs[-k:]
        return sum(tail) / len(tail)

    return {
        "n": len(records),
        "mean25": round(mean_last(scores, 25)),
        "mean50": round(mean_last(scores, 50)),
        "best": int(max(scores)),
        "n256": sum(1 for t in tiles if t >= 256),
        "n512": sum(1 for t in tiles if t >= 512),
        "invalid_rate": round(100.0 * inv / max(1, inv + mov), 1),
    }


def read_eval_results() -> dict[str, Any] | None:
    """读取训练后的 greedy 评估结果（09 写入）。"""
    if not p_eval().exists():
        return None
    try:
        data = json.loads(p_eval().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def train_status(train_proc: subprocess.Popen | None = None) -> dict[str, Any]:
    """汇总训练状态：运行中/启动阶段/空闲 + 进度 + 预估剩余时间。"""
    state = read_train_state()
    running = False
    pid: int | None = None
    target = 0
    started_at: str | None = None
    phase = "idle"
    if state is not None:
        pid = int(state.get("pid", 0))
        target = int(state.get("episodes", 0))
        started_at = str(state.get("started_at", ""))
        # live 文件足够新鲜（30 秒内更新）说明工作进程仍在跑，
        # 即便记录的 launcher PID 已退出也不判死、不清状态
        live_probe = read_live()
        live_fresh = False
        if live_probe is not None:
            ts = live_probe.get("updated_at")
            if ts:
                try:
                    age = time.time() - datetime.fromisoformat(str(ts)).timestamp()
                    live_fresh = 0 <= age < 30
                except ValueError:
                    pass
        if _pid_alive(pid) or live_fresh:
            running = True
            phase = "running"
        else:
            # 进程已结束但状态文件还在（旧版 09 或异常退出）：清理陈旧文件
            try:
                p_state().unlink(missing_ok=True)
            except OSError:
                pass
            phase = "finished"
    elif train_proc is not None and train_proc.poll() is None:
        running = True
        pid = int(train_proc.pid)
        phase = "starting"

    done = count_episodes()
    records_all = training_records()
    latest = latest_record() or {}
    elapsed_s = 0.0
    eta_s: float | None = None
    if started_at:
        try:
            stamp = datetime.fromisoformat(started_at)
            elapsed_s = max(0.0, time.time() - stamp.timestamp())
        except ValueError:
            pass
    if running and done > 0 and elapsed_s > 0 and target > done:
        eta_s = elapsed_s / done * (target - done)

    # 旧版训练进程没有 live 文件：用日志 mtime 合成"当前第 N 局进行中"
    live = read_live()
    current: dict[str, Any] | None = None
    if live is None and running and done > 0 and p_train_log().exists():
        try:
            current = {
                "episode": done + 1,
                "synthesized": True,
                "cur_elapsed_s": round(
                    max(0.0, time.time() - p_train_log().stat().st_mtime), 1
                ),
            }
        except OSError:
            pass

    return {
        "running": running,
        "phase": phase,
        "pid": pid,
        "target": target,
        "done": done,
        "started_at": started_at,
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": None if eta_s is None else round(eta_s, 0),
        "latest": latest,
        "weights_ready": p_readout().exists(),
        "live": read_live(),
        "current": current,
        "summary": training_summary(records_all),
        "eval": read_eval_results(),
    }


# --------------------------------------------------------------------------
# 教学（teach）进程管理
# --------------------------------------------------------------------------
def _find_teach_pids() -> list[int]:
    """兜底查找教学进程：命令行含 fly2048_core.py ... teach 的 python 进程。

    服务重启后 teach_proc 引用丢失时用它定位（正常路径直接用 Popen 句柄）。
    """
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -match 'fly2048_core\\.py' "
                "-and $_.CommandLine -match ' teach' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            timeout=15, text=True,
        )
        for token in out.split():
            pids.append(int(token))
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return pids


def _kill_process_tree(pid: int) -> bool:
    """taskkill /T /F：连 launcher 带工作子进程整棵树杀掉。"""
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True, timeout=10,
        )
        return not _pid_alive(pid)
    except (subprocess.SubprocessError, OSError):
        return False


def teach_records() -> list[dict[str, Any]]:
    if not p_teach_log().exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in p_teach_log().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return records


def teach_status(teach_proc: subprocess.Popen | None = None) -> dict[str, Any]:
    """汇总教学状态：运行中/空闲 + 进度（教学局数）+ 当前教学局实时数据。"""
    running = False
    pid: int | None = None
    phase = "idle"

    if teach_proc is not None and teach_proc.poll() is None:
        running = True
        pid = int(teach_proc.pid)
        phase = "running"
    else:
        # 服务重启兜底：live 文件显示正在教学且 60 秒内有更新
        live = read_live()
        fresh = False
        if live is not None and live.get("phase") == "teach":
            ts = live.get("updated_at")
            if ts:
                try:
                    age = time.time() - datetime.fromisoformat(str(ts)).timestamp()
                    fresh = 0 <= age < 60
                except ValueError:
                    pass
        wmi_pids = _find_teach_pids()
        if fresh and wmi_pids:
            running = True
            pid = wmi_pids[0]
            phase = "running"

    records = teach_records()
    return {
        "running": running,
        "phase": phase,
        "pid": pid,
        "taught_total": len(records),
        "latest": records[-1] if records else {},
        "live": read_live() if running else None,
    }


def _read_console_text(path: Path) -> str:
    """子进程 stdout 日志：正常为 UTF-8；Windows 重定向时可能写成系统代码页(GBK)。"""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def _child_env() -> dict[str, str]:
    """子进程环境：强制标准流 UTF-8，避免重定向日志变成 GBK。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def teach_console_log_lines() -> list[str]:
    """教学 stdout 日志；没有则从 teach_log.jsonl 合成。"""
    if p_teach_console().exists():
        try:
            lines = [
                ln for ln in _read_console_text(p_teach_console()).splitlines()
                if ln.strip()
            ]
            if lines:
                return lines[-MAX_CONSOLE_LOG_LINES:]
        except OSError:
            pass
    out: list[str] = []
    for rec in teach_records()[-40:]:
        out.append(
            f"教学局{rec.get('episode')}  score={rec.get('score')}  "
            f"max={rec.get('max_tile')}  施教步={rec.get('confident_steps')}  "
            f"一致率={rec.get('agreement'):.2f}"
        )
    return out


def _find_auto_pids() -> list[int]:
    """兜底查找自动循环进程：命令行含 fly2048_core.py ... auto。"""
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -match 'fly2048_core\\.py' "
                "-and $_.CommandLine -match ' auto' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            timeout=15, text=True,
        )
        for token in out.split():
            pids.append(int(token))
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return pids


def auto_status(auto_proc: subprocess.Popen | None = None) -> dict[str, Any]:
    """自动循环状态：运行中 + 轮次/阶段（从 live 或 state 读取）。"""
    running = False
    pid: int | None = None
    cycle = None
    cycles = None
    phase = "idle"

    if auto_proc is not None and auto_proc.poll() is None:
        running = True
        pid = int(auto_proc.pid)
    else:
        wmi_pids = _find_auto_pids()
        # live 新鲜度兜底：90 秒内有更新才算在跑（auto 阶段切换有短暂空窗）
        live = read_live()
        fresh = False
        if live is not None and live.get("updated_at"):
            try:
                age = time.time() - datetime.fromisoformat(
                    str(live["updated_at"])
                ).timestamp()
                fresh = 0 <= age < 90
            except ValueError:
                pass
        state = read_train_state()
        is_auto_state = bool(state) and state.get("mode") == "auto"
        if wmi_pids and (fresh or is_auto_state):
            running = True
            pid = wmi_pids[0]

    # 轮次/阶段优先 live，其次 state
    source: dict[str, Any] = {}
    live = read_live()
    if live is not None and (live.get("auto_phase") or live.get("cycle")):
        source = live
    else:
        st = read_train_state()
        if st and st.get("mode") == "auto":
            source = st
    if source:
        cycle = source.get("cycle")
        cycles = source.get("cycles")
        phase = str(source.get("auto_phase", "auto"))

    return {
        "running": running,
        "pid": pid,
        "cycle": cycle,
        "cycles": cycles,
        "phase": phase if running else "idle",
        "live": live if running else None,
    }


def auto_console_log_lines() -> list[str]:
    if p_auto_console().exists():
        try:
            lines = [
                ln for ln in _read_console_text(p_auto_console()).splitlines()
                if ln.strip()
            ]
            if lines:
                return lines[-MAX_CONSOLE_LOG_LINES:]
        except OSError:
            pass
    return []


def console_log_lines() -> list[str]:
    """控制台日志：优先训练 stdout 文件；没有则从训练日志 JSONL 合成。"""
    if p_train_console().exists():
        try:
            lines = _read_console_text(p_train_console()).splitlines()
            lines = [ln for ln in lines if ln.strip()]
            if lines:
                return lines[-MAX_CONSOLE_LOG_LINES:]
        except OSError:
            pass
    if p_train_log().exists():
        try:
            tail = p_train_log().read_text(encoding="utf-8").splitlines()[-40:]
            out: list[str] = []
            for ln in tail:
                rec = json.loads(ln)
                out.append(
                    f"第{rec.get('episode')}局  score={rec.get('score')}  "
                    f"max={rec.get('max_tile')}  moves={rec.get('moves')}  "
                    f"invalid={rec.get('invalid')}  T={rec.get('temperature')}"
                )
            return out
        except (OSError, json.JSONDecodeError):
            pass
    return []


class Fly2048LabHandler(BaseHTTPRequestHandler):
    server_version = "Fly2048Lab/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path = DEFAULT_HTML) -> None:
        if not path.exists():
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"missing page: {path}"},
            )
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        if content_length > 64 * 1024:
            raise ValueError("request body too large")
        raw = self.rfile.read(content_length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/":
                self._send_html()
            elif self.path == "/console":
                self._send_html(CONSOLE_HTML)
            elif self.path == "/state":
                with self.server.lab.lock:  # type: ignore[attr-defined]
                    self.server.lab._reload_weights_if_newer()  # type: ignore[attr-defined]
                    payload = self.server.lab.snapshot()  # type: ignore[attr-defined]
                self._send_json(HTTPStatus.OK, payload)
            elif self.path == "/weights":
                self._send_json(HTTPStatus.OK, weight_profiles())
            elif self.path == "/learning":
                self._send_json(
                    HTTPStatus.OK,
                    learning_series(self.server.args.log),  # type: ignore[attr-defined]
                )
            elif self.path == "/train/status":
                with self.server.lab.lock:  # type: ignore[attr-defined]
                    self.server.lab._reload_weights_if_newer()  # type: ignore[attr-defined]
                status = train_status(getattr(self.server, "train_proc", None))
                status["teach"] = teach_status(
                    getattr(self.server, "teach_proc", None)
                )
                status["auto"] = auto_status(
                    getattr(self.server, "auto_proc", None)
                )
                self._send_json(HTTPStatus.OK, status)
            elif self.path == "/train/log":
                self._send_json(
                    HTTPStatus.OK, {"lines": console_log_lines()}
                )
            elif self.path == "/teach/log":
                self._send_json(
                    HTTPStatus.OK, {"lines": teach_console_log_lines()}
                )
            elif self.path == "/auto/log":
                self._send_json(
                    HTTPStatus.OK, {"lines": auto_console_log_lines()}
                )
            elif self.path == "/train/records":
                self._send_json(
                    HTTPStatus.OK, {"records": training_records()}
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"internal error: {exc}"},
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            if self.path == "/step":
                if self.server.lab.teach_mode:  # type: ignore[attr-defined]
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "手动教学进行中：请用方向键教学，或先结束教学"},
                    )
                    return
                with self.server.lab.lock:  # type: ignore[attr-defined]
                    self.server.lab._reload_weights_if_newer()  # type: ignore[attr-defined]
                    result = self.server.lab.step_lab()  # type: ignore[attr-defined]
                self._send_json(HTTPStatus.OK, result)
            elif self.path == "/reset":
                with self.server.lab.lock:  # type: ignore[attr-defined]
                    self.server.lab._reload_weights_if_newer()  # type: ignore[attr-defined]
                    result = self.server.lab.reset_lab()  # type: ignore[attr-defined]
                self._send_json(HTTPStatus.OK, result)
            elif self.path == "/weights/select":
                self._weights_select(body)
            elif self.path == "/weights/create":
                self._weights_create(body)
            elif self.path == "/weights/rename":
                self._weights_rename(body)
            elif self.path == "/weights/delete":
                self._weights_delete(body)
            elif self.path == "/train/start":
                self._train_start(body)
            elif self.path == "/train/stop":
                self._train_stop()
            elif self.path == "/teach/start":
                self._teach_start(body)
            elif self.path == "/teach/stop":
                self._teach_stop()
            elif self.path == "/auto/start":
                self._auto_start(body)
            elif self.path == "/auto/stop":
                self._auto_stop()
            elif self.path == "/teach_manual/start":
                self._teach_manual_start(body)
            elif self.path == "/teach_manual/step":
                self._teach_manual_step(body)
            elif self.path == "/teach_manual/stop":
                self._teach_manual_stop()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"internal error: {exc}"},
            )

    # --- 命名权重操作 ---
    def _any_operation_running(self) -> bool:
        if train_status(getattr(self.server, "train_proc", None)).get("running"):
            return True
        if teach_status(getattr(self.server, "teach_proc", None)).get("running"):
            return True
        if auto_status(getattr(self.server, "auto_proc", None)).get("running"):
            return True
        if getattr(self.server.lab, "teach_mode", False):
            return True
        return False

    def _manual_teach_conflict(self) -> bool:
        """练习/AI教学/自动循环启动前的互斥：手动教学会话进行中则拒绝。"""
        if not getattr(self.server.lab, "teach_mode", False):
            return False
        self._send_json(
            HTTPStatus.CONFLICT,
            {"error": "手动教学进行中：请先在互动台「结束并保存」，再启动练习/AI教学/自动循环"},
        )
        return True

    # --- 手动教学（人当老师） ---
    def _teach_manual_start(self, body: dict[str, Any]) -> None:
        if (train_status(getattr(self.server, "train_proc", None)).get("running")
                or teach_status(getattr(self.server, "teach_proc", None)).get("running")
                or auto_status(getattr(self.server, "auto_proc", None)).get("running")):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "练习/AI教学/自动循环进行中，请先停止再开始手动教学"},
            )
            return
        aug_k = int(body.get("aug_k", 1))
        lr_raw = body.get("lr")
        lr = float(lr_raw) if lr_raw is not None else None
        with self.server.lab.lock:  # type: ignore[attr-defined]
            snapshot = self.server.lab.teach_manual_start(aug_k, lr)  # type: ignore[attr-defined]
        self._send_json(HTTPStatus.OK, snapshot)

    def _teach_manual_step(self, body: dict[str, Any]) -> None:
        action = int(body.get("action", -1))
        with self.server.lab.lock:  # type: ignore[attr-defined]
            snapshot = self.server.lab.teach_manual_step(action)  # type: ignore[attr-defined]
        self._send_json(HTTPStatus.OK, snapshot)

    def _teach_manual_stop(self) -> None:
        with self.server.lab.lock:  # type: ignore[attr-defined]
            snapshot = self.server.lab.teach_manual_stop()  # type: ignore[attr-defined]
        self._send_json(HTTPStatus.OK, snapshot)

    def _weights_select(self, body: dict[str, Any]) -> None:
        if self._any_operation_running():
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "练习/教学/自动循环进行中，请停止后再切换权重"},
            )
            return
        profile_id = str(body.get("id", "")).strip()
        with self.server.lab.lock:  # type: ignore[attr-defined]
            _set_active(profile_id)
            self.server.lab.use_active_profile()  # type: ignore[attr-defined]
            snapshot = self.server.lab.snapshot()  # type: ignore[attr-defined]
        self._send_json(
            HTTPStatus.OK,
            {**weight_profiles(), "state": snapshot},
        )

    def _weights_create(self, body: dict[str, Any]) -> None:
        if self._any_operation_running():
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "练习/教学/自动循环进行中，请停止后再新建权重"},
            )
            return
        name = str(body.get("name", ""))
        mode = str(body.get("mode", "copy"))
        n_features = stage9.feature_dim(self.server.lab.brain.n_neurons)  # type: ignore[attr-defined]
        info = create_weight(name, mode, n_features)
        # 新建后自动切换过去（复制的是当前权重，直接可玩）
        with self.server.lab.lock:  # type: ignore[attr-defined]
            _set_active(info["id"])
            self.server.lab.use_active_profile()  # type: ignore[attr-defined]
            snapshot = self.server.lab.snapshot()  # type: ignore[attr-defined]
        self._send_json(
            HTTPStatus.OK,
            {**weight_profiles(), "profile": info, "state": snapshot},
        )

    def _weights_rename(self, body: dict[str, Any]) -> None:
        profile_id = str(body.get("id", ACTIVE_ID)).strip()
        new_name = str(body.get("name", ""))
        info = rename_weight(profile_id, new_name)
        self._send_json(
            HTTPStatus.OK,
            {**weight_profiles(), "profile": info},
        )

    def _weights_delete(self, body: dict[str, Any]) -> None:
        if self._any_operation_running():
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "练习/教学/自动循环进行中，请停止后再删除权重"},
            )
            return
        profile_id = str(body.get("id", "")).strip()
        if profile_id == ACTIVE_ID:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "不能删除当前使用的权重，请先切换到其他权重"},
            )
            return
        delete_weight(profile_id)
        self._send_json(HTTPStatus.OK, weight_profiles())

    def _train_start(self, body: dict[str, Any]) -> None:
        if self._manual_teach_conflict():
            return
        current = train_status(getattr(self.server, "train_proc", None))
        if current.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"已有训练在运行（pid={current.get('pid')}），先停止再启动", **current},
            )
            return
        teach = teach_status(getattr(self.server, "teach_proc", None))
        if teach.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"教学进行中（pid={teach.get('pid')}），先停止教学再练习"},
            )
            return
        astat = auto_status(getattr(self.server, "auto_proc", None))
        if astat.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"自动循环进行中（pid={astat.get('pid')}），先停止再练习"},
            )
            return
        episodes = int(body.get("episodes", 300))
        if episodes < 1 or episodes > 100000:
            raise ValueError("episodes must be between 1 and 100000")
        baseline = bool(body.get("random_baseline", False))
        resume = bool(body.get("resume", True))   # 默认续训，保护已有成果
        # 边界：要求续训但当前 profile 下没有权重时（例如全新 profile 首次训练），
        # 自动降级为全新训练，否则 core 脚本会因找不到权重直接退出
        resume_effective = resume and p_readout().exists()

        cmd = [
            sys.executable, "-u", str(STAGE9_PATH), "train",
            "--episodes", str(episodes),
            "--out-dir", str(ACTIVE_DIR),
        ]
        if resume_effective:
            cmd += ["--resume"]
        if not baseline:
            cmd += ["--random-baseline-episodes", "0"]
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        p_train_console().parent.mkdir(parents=True, exist_ok=True)
        handle = open_console_log(p_train_console())
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            env=_child_env(),
        )
        handle.close()
        self.server.train_proc = proc  # type: ignore[attr-defined]
        self._send_json(
            HTTPStatus.OK,
            {
                "running": True, "pid": proc.pid,
                "episodes": episodes, "baseline": baseline,
                "resume": resume_effective,
            },
        )

    def _train_stop(self) -> None:
        state = read_train_state()
        killed = False
        state_pid: int | None = None
        if state is not None:
            state_pid = int(state.get("pid", 0))
            if state_pid and _pid_alive(state_pid):
                os.kill(state_pid, signal.SIGTERM)
                deadline = time.time() + 3.0
                while _pid_alive(state_pid) and time.time() < deadline:
                    time.sleep(0.05)
                killed = not _pid_alive(state_pid)
        proc = getattr(self.server, "train_proc", None)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.poll() is None:
                proc.kill()
            killed = True
            # 杀启动器后，工作进程若还活着会继续写状态文件：扫尾补刀
            deadline = time.time() + 8.0
            while time.time() < deadline:
                orphan = read_train_state()
                if orphan is None:
                    break
                orphan_pid = int(orphan.get("pid", 0))
                if not orphan_pid or not _pid_alive(orphan_pid):
                    break
                os.kill(orphan_pid, signal.SIGTERM)
                time.sleep(0.2)
                killed = True
        # 只在确实结束掉训练时清理状态文件；杀不掉就保留，避免"假停机"
        if killed or state_pid is None or not _pid_alive(state_pid):
            if p_state().exists():
                try:
                    p_state().unlink(missing_ok=True)
                except OSError:
                    pass
        self._send_json(
            HTTPStatus.OK,
            {"stopped": killed, "pid": state_pid},
        )

    def _teach_start(self, body: dict[str, Any]) -> None:
        if self._manual_teach_conflict():
            return
        teach = teach_status(getattr(self.server, "teach_proc", None))
        if teach.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"已有教学在运行（pid={teach.get('pid')}），先停止再启动"},
            )
            return
        current = train_status(getattr(self.server, "train_proc", None))
        if current.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"练习（训练）进行中（pid={current.get('pid')}），先停止再教学"},
            )
            return
        astat = auto_status(getattr(self.server, "auto_proc", None))
        if astat.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"自动循环进行中（pid={astat.get('pid')}），先停止再教学"},
            )
            return
        episodes = int(body.get("episodes", 10))
        if episodes < 1 or episodes > 1000:
            raise ValueError("教学 episodes must be between 1 and 1000")
        aug_k = int(body.get("aug_k", stage9.TEACH_AUG_K))
        if aug_k not in (1, 2, 4, 8):
            raise ValueError("aug_k must be one of 1/2/4/8")
        depth = int(body.get("depth", stage9.TEACHER_DEPTH))
        if depth not in (2, 3):
            raise ValueError("depth must be 2 or 3")
        if not p_readout().exists():
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "当前权重还未生成，请先在该权重下完成一次练习（训练）"},
            )
            return

        cmd = [
            sys.executable, "-u", str(STAGE9_PATH), "teach",
            "--episodes", str(episodes),
            "--aug-k", str(aug_k),
            "--depth", str(depth),
            "--out-dir", str(ACTIVE_DIR),
        ]
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        p_teach_console().parent.mkdir(parents=True, exist_ok=True)
        handle = open_console_log(p_teach_console())
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            env=_child_env(),
        )
        handle.close()
        self.server.teach_proc = proc  # type: ignore[attr-defined]
        self._send_json(
            HTTPStatus.OK,
            {"running": True, "pid": proc.pid, "episodes": episodes},
        )

    def _teach_stop(self) -> None:
        killed = False
        target_pid: int | None = None
        proc = getattr(self.server, "teach_proc", None)
        if proc is not None:
            target_pid = int(proc.pid)
            if proc.poll() is None:
                killed = _kill_process_tree(target_pid)
            else:
                killed = True
        else:
            # 服务重启后的兜底：按命令行特征找教学进程，整树杀掉
            pids = _find_teach_pids()
            if pids:
                target_pid = pids[0]
                killed = _kill_process_tree(target_pid)
        self._send_json(
            HTTPStatus.OK,
            {"stopped": killed, "pid": target_pid},
        )

    def _auto_start(self, body: dict[str, Any]) -> None:
        if self._manual_teach_conflict():
            return
        astat = auto_status(getattr(self.server, "auto_proc", None))
        if astat.get("running"):
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": f"已有自动循环在运行（pid={astat.get('pid')}），先停止再启动"},
            )
            return
        if train_status(getattr(self.server, "train_proc", None)).get("running"):
            self._send_json(
                HTTPStatus.CONFLICT, {"error": "练习进行中，先停止再启动自动循环"},
            )
            return
        if teach_status(getattr(self.server, "teach_proc", None)).get("running"):
            self._send_json(
                HTTPStatus.CONFLICT, {"error": "教学进行中，先停止再启动自动循环"},
            )
            return

        cycles = int(body.get("cycles", 5))
        teach_eps = int(body.get("teach_episodes", 10))
        practice_eps = int(body.get("practice_episodes", 50))
        # 教学与课程对齐默认开启：每轮教学 15→3 局、老师深 2→3（方向1）
        teach_schedule = bool(body.get("teach_schedule", True))
        aug_k = int(body.get("aug_k", stage9.TEACH_AUG_K))
        if not (1 <= cycles <= 1000):
            raise ValueError("cycles must be between 1 and 1000")
        if not (1 <= teach_eps <= 100):
            raise ValueError("teach_episodes must be between 1 and 100")
        if not (1 <= practice_eps <= 10000):
            raise ValueError("practice_episodes must be between 1 and 10000")
        if aug_k not in (1, 2, 4, 8):
            raise ValueError("aug_k must be one of 1/2/4/8")

        cmd = [
            sys.executable, "-u", str(STAGE9_PATH), "auto",
            "--cycles", str(cycles),
            "--teach-episodes", str(teach_eps),
            "--practice-episodes", str(practice_eps),
            "--aug-k", str(aug_k),
            "--out-dir", str(ACTIVE_DIR),
        ]
        if not teach_schedule:
            cmd.append("--no-teach-schedule")
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        p_auto_console().parent.mkdir(parents=True, exist_ok=True)
        handle = open_console_log(p_auto_console())
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_DIR, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=flags, env=_child_env(),
        )
        handle.close()
        self.server.auto_proc = proc  # type: ignore[attr-defined]
        self._send_json(
            HTTPStatus.OK,
            {"running": True, "pid": proc.pid,
             "cycles": cycles, "teach_episodes": teach_eps,
             "practice_episodes": practice_eps},
        )

    def _auto_stop(self) -> None:
        killed = False
        target_pid: int | None = None
        proc = getattr(self.server, "auto_proc", None)
        if proc is not None:
            target_pid = int(proc.pid)
            if proc.poll() is None:
                killed = _kill_process_tree(target_pid)
            else:
                killed = True
        else:
            pids = _find_auto_pids()
            if pids:
                target_pid = pids[0]
                killed = _kill_process_tree(target_pid)
        # auto 写的 train_state 一并清理，避免状态误判
        try:
            st = read_train_state()
            if st and st.get("mode") == "auto":
                p_state().unlink(missing_ok=True)
        except OSError:
            pass
        self._send_json(HTTPStatus.OK, {"stopped": killed, "pid": target_pid})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动果蝇玩 2048 的网页互动台。"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--nodes", type=Path, default=stage9.DEFAULT_NODES)
    parser.add_argument("--edges", type=Path, default=stage9.DEFAULT_EDGES)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="读出层权重 readout.npz（默认自动查找 outputs/fly2048/readout.npz）",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="训练日志 training_log.jsonl（默认自动查找）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 统一工作目录：core 脚本与所有相对路径（outputs/...）都以项目根为基准
    os.chdir(PROJECT_DIR)
    lab = Fly2048Lab(args)
    server = HTTPServer((args.host, args.port), Fly2048LabHandler)
    server.lab = lab  # type: ignore[attr-defined]
    server.args = args  # type: ignore[attr-defined]
    print(f"果蝇 2048 互动台：http://{args.host}:{args.port}/")
    print(
        f"大脑就绪：{lab.brain.n_neurons} 神经元 / {len(lab.brain.edges)} 连接，"
        f"16 个感觉输入 → 中心神经元 index={lab.brain.center_index}"
    )
    if lab.weights_loaded and lab.weights_path is not None:
        print(f"读出层权重：{lab.weights_path}")
    else:
        print("读出层权重：未找到 readout.npz，使用随机初始化（未训练）")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止果蝇 2048 互动台 ...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())