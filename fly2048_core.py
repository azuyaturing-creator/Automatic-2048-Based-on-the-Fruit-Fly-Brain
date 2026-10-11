"""让 Brian2 连接组子图玩 2048 —— Level 2 学习版（A2C + MLP 读出）。

脑回路（固定不变）：
    16 个棋盘格 → 16 个上游感觉神经元（泊松放电，格值越大放电越快）
    → 连接组 LIF 子图（权重来自真实连接组数据，不可学习）
    → 读出层（唯一可学习部分）：特征 → 64 ReLU 共享 trunk → 4 方向 Actor + V(s) Critic

决策特征：最近 5 窗归一化放电率 + 5 窗总放电强度 + 3 维棋盘摘要。

学习算法（A2C）：
    每个 100ms 脑循环 = 一步决策；
    奖励 = log2 缩放的合并得分（课程塑形）+ 布局健康度增量，非法移动给负奖励；
    Monte Carlo 折扣回报 + Critic 逐状态基线 + 熵系数退火（带熵下限）
    + softmax 温度退火 + 课程学习。

用法：
    uv run python fly2048_core.py train --episodes 300
    uv run python fly2048_core.py play --weights outputs/fly2048/readout.npz
    uv run python fly2048_core.py random --episodes 300
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import brian2 as b2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
DEFAULT_NODES = Path("outputs/subgraphs/DNge104_R_12781_nodes.csv")
DEFAULT_EDGES = Path("outputs/subgraphs/DNge104_R_12781_edges.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/fly2048")
STAGE5_PATH = Path(__file__).with_name("05_brian2_lif_sim.py")

CENTER_BODY_ID = 12781
INPUT_COUNT = 16           # 棋盘 4x4 = 16 格，一对一驱动 16 个感觉神经元
WINDOW_MS = 100.0          # 一步决策 = 100ms 脑循环
DT_MS = 0.1

BASE_RATE_HZ = 8.0         # 编码基准：rate = BASE * (1 + log2(格值))
MAX_RATE_HZ = 120.0

N_ACTIONS = 4
ACTION_NAMES = ["up", "right", "down", "left"]   # 与 Game2048 的动作序一致
K_MAP = {0: 1, 1: 2, 2: 3, 3: 0}                 # 把动作方向旋转到"向左压缩"
K_BACK = {1: 3, 2: 2, 3: 1, 0: 0}

GAMMA = 0.99                # 折扣因子
INVALID_REWARD = -1.0       # 非法移动惩罚
STEP_COST = -0.05           # 每步小成本，促使尽早合并
TEMP_START = 2.0            # softmax 温度初值（高探索）
TEMP_MIN = 0.3
# 快退火：约 100 局降到 0.4（0.984^100≈0.2，2.0*0.2=0.4），策略更快果断
TEMP_DECAY = 0.984          # 每局温度衰减
LR_DEFAULT = 0.01
CRITIC_LR = 0.02            # Critic（V 值头）学习率
VALUE_COEF = 0.5            # Critic 损失权重（保留兼容）
# 熵正则：随训练进度从 ENTROPY_START 线性退火到 ENTROPY_END（不到零），
# 温度触底后熵项仍能阻止策略坍缩；单步策略熵低于 ENTROPY_FLOOR_H 时
# 临时把熵系数放大 ENTROPY_KICK 倍，把过早固化的策略"踢"回探索
ENTROPY_START = 0.05
ENTROPY_END = 0.01
ENTROPY_FLOOR_H = 0.35      # 策略熵下限（nats；ln2≈0.69，ln4≈1.39）
ENTROPY_KICK = 3.0
BASELINE_EMA = 0.95         # EMA 回报均值（仅日志用，A2C baseline 由 Critic 给出）
TEACH_LR = 0.05             # 模仿教学（监督交叉熵）学习率
TEACHER_DEPTH = 3           # 老师 expectimax 深度（standalone teach 默认；动作层数）
TEACHER_TOP_ACTIONS = 2     # 每层 max 节点保留的候选动作数（剪枝）
TEACHER_SAMPLE_EMPTY = 4    # 每个 chance 节点采样的空格数（空格多时随机抽）
TEACH_MARGIN = 0.01         # 最佳/次佳价值相对差距阈值，低于此视为"两可"，跳过不教
MAX_STEPS_PER_EPISODE = 1500

# ── 方向1：教学强度与课程对齐（auto 调度；先验衰减点对齐课程节点）──────────
# 每轮教学局数随"轮中点课程进度"从 15 线性减到 3：早期多带、后期自主；
# 老师深度 2→3：深度4 实测约 750ms/决策（深度3 约45ms），过慢故封顶3。
TEACH_N_EARLY = 15
TEACH_N_LATE = 3
TEACH_DEPTH_EARLY = 2
TEACH_DEPTH_LATE = 3

# ── 方向4：棋盘 D4 对称数据增强 ────────────────────────────────────────────
# 2048 对旋转/翻转严格对称；同一逻辑局面的 K 个对称呈现各跑一次脑、
# 各维护独立 3 窗历史，老师动作按几何置换后分别做交叉熵（梯度取 1/K 平均）。
# K ∈ {1,2,4,8}：1=关闭；2=[恒等,R180]；4=四个旋转；8=旋转+翻转（最慢）。
# 大脑连接组本身不对称，增强的意义是让读出层对同一逻辑局面学到几何一致的动作。
# 成本：每教学步 K 次脑仿真（单次约 98ms），K=4 时一局约 1.5 分钟。
TEACH_AUG_K = 4

# 好局门控重放：某局得分进入近 REPLAY_WINDOW 局的 top REPLAY_TOP_FRAC 时，
# 以 REPLAY_LR_FACTOR 倍学习率把该局轨迹再更新一遍（好局多巩固，又不至过拟合单局运气）
REPLAY_WINDOW = 25
REPLAY_TOP_FRAC = 0.2
REPLAY_MIN_HISTORY = 5
REPLAY_LR_FACTOR = 0.5

# advantage 裁剪：风暴局（1500 步纯负回报）会把 EMA baseline 拖到很负，
# 正常局的 advantage 随之被放大几十倍导致发散；裁剪到 ±ADV_CLIP 保稳定
ADV_CLIP = 20.0
# critic TD 误差裁剪：未裁剪时 V 一旦越过 G，(G−V) 随权重正反馈指数放大
# （1099 神经元/3303 特征下实测 2 局内溢出到 inf 并经共享 trunk 污染 actor）；
# 与 advantage 同幅裁剪，切断发散回路，V 仍以每步有界增量趋近 G
CRITIC_TD_CLIP = 20.0
# 但仅裁 TD 标量不够：_backward_critic 中 d_h = v2·td 经共享 w1 反传，
# v2 与 w1 仍可互相指数放大（加 TD 裁剪后探针实测第 2 局 V 仍达 2.5e7）。
# 对反传进 trunk 的 d_z1 做 L2 范数裁剪——只缩梯度长度、不改方向，
# 每步 ||Δw1|| 有界，正反馈回路无法复利累积。
TRUNK_GRAD_CLIP = 5.0
# V(s) 输出物理硬边界：γ=0.99 下单步奖励 r∈[-1.05, 16.6]
# （merge_mult≤1.5 × log2(2048)≈11，加 health±0.05；invalid=-1），
# 折扣回报理论界 r/(1-γ) ∈ [-105, 1660]，取宽松边。
# 真实 G 到不了边界，不影响学习，但从源头禁止 V 参与千万量级自激。
VALUE_CLIP_MIN = -500.0
VALUE_CLIP_MAX = 5000.0

# 时间维度：决策特征拼接最近 TIME_WINDOWS 个脑循环的放电率
# （当前窗 + 前 K-1 窗），让读出层拥有短期记忆；特征维 = K × n_neurons + K + 3
TIME_WINDOWS = 5

# 增强读出层：特征 → HIDDEN_SIZE(ReLU 共享 trunk) → 4 方向 Actor + 1 维 Critic。
# HIDDEN_SIZE=0 时退化为纯线性读出（用于加载旧版权重）
HIDDEN_SIZE = 64
# 布局健康度奖励权重：reward += HEALTH_W * (health_after - health_before)
HEALTH_W = 0.05

# 特征增强：归一化放电率只保留相对模式，丢失了"盘面规模"信号。
# 额外拼接：每窗 log 总放电数（K 维）+ 当前棋盘摘要（BOARD_EXTRA_DIM 维：
# log2 格值总和 / 最大格量级 / 空格比例）
BOARD_EXTRA_DIM = 3
SPIKE_TOTAL_SCALE = 8.0    # log1p(总spike数)/8 归一化（log1p(3000)≈8）
BOARD_LOG_SUM_SCALE = 16.0 * 11.0   # 16 格全 2048(log2=11) 时归一化到 1

# 课程学习（按训练进度 progress∈[0,1] 线性塑形）：
#   开局合并奖励放大 1+CURRICULUM_MERGE_BOOST（早期多奖励合并、快速形成基本策略）
#   步数成本早期更小（鼓励多探索布局），后期回到 STEP_COST
#   健康度奖励早期权重低、后期满权重（蛇形链是高阶技巧，后期再强调）
CURRICULUM_MERGE_BOOST = 0.5
CURRICULUM_STEP_MIN = 0.4
CURRICULUM_HEALTH_MIN = 0.25


def feature_dim(n_neurons: int, windows: int = TIME_WINDOWS) -> int:
    """读出层输入维度 = K 窗归一化放电率 + K 窗总放电强度 + 棋盘摘要。"""
    return windows * n_neurons + windows + BOARD_EXTRA_DIM


def load_stage_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage5 = load_stage_module("stage5_brian2_lif_sim_2048", STAGE5_PATH)


# --------------------------------------------------------------------------
# 2048 游戏环境（纯 numpy，无需额外依赖）
# --------------------------------------------------------------------------
class Game2048:
    """标准 4x4 2048。动作序 0=up 1=right 2=down 3=left。"""

    def __init__(self, rng: np.random.Generator, max_steps: int = MAX_STEPS_PER_EPISODE):
        self.rng = rng
        self.max_steps = max_steps
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.moves = 0

    def reset(self) -> None:
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.moves = 0
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return
        index = self.rng.integers(len(empty))
        row, col = empty[index]
        self.board[row, col] = 2 if self.rng.random() < 0.9 else 4

    @staticmethod
    def _merge_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
        """单行向左侧压缩合并，返回 (新行, 合并得分)。"""
        vals = row[row != 0].tolist()
        merged: list[int] = []
        gain = 0
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                merged.append(vals[i] * 2)
                gain += vals[i] * 2
                i += 2
            else:
                merged.append(vals[i])
                i += 1
        out = np.zeros(4, dtype=np.int64)
        out[: len(merged)] = merged
        return out, gain

    def move(self, action: int) -> tuple[bool, int]:
        """执行滑动，返回 (是否发生移动, 合并得分)。非法/无效移动不改变棋盘。"""
        old = self.board.copy()
        # 注意：np.rot90 返回视图，写入前必须拷贝，否则会写坏原棋盘
        board = np.rot90(self.board, K_MAP[action]).copy()
        gain = 0
        for r in range(4):
            row, g = self._merge_row_left(board[r])
            gain += g
            board[r] = row
        self.board = np.rot90(board, K_BACK[K_MAP[action]]).copy()
        moved = not np.array_equal(old, self.board)
        if moved:
            self.score += gain
            self.moves += 1
            self._spawn()
        return moved, gain

    def legal_actions(self) -> np.ndarray:
        """返回 4 个方向各自是否合法（压缩后棋盘会变 = 合法），供动作屏蔽。"""
        legal = np.zeros(N_ACTIONS, dtype=bool)
        for action in range(N_ACTIONS):
            board = np.rot90(self.board, K_MAP[action])
            for r in range(4):
                row, _ = self._merge_row_left(board[r])
                if not np.array_equal(row, board[r]):
                    legal[action] = True
                    break
        return legal

    def game_over(self) -> bool:
        if self.moves >= self.max_steps:
            return True
        if (self.board == 0).any():
            return False
        return not bool(self.legal_actions().any())

    @property
    def max_tile(self) -> int:
        return int(self.board.max())

    @property
    def board_flat(self) -> np.ndarray:
        return self.board.reshape(-1)

    def peek_move(self, action: int) -> tuple[np.ndarray, int]:
        """只模拟压缩合并、不生成新块、不改变当前棋盘。

        返回 (合并后棋盘, 合并得分)，供贪心老师做一步预判。
        """
        board = np.rot90(self.board, K_MAP[action]).copy()
        gain = 0
        for r in range(4):
            row, g = self._merge_row_left(board[r])
            gain += g
            board[r] = row
        board = np.rot90(board, K_BACK[K_MAP[action]]).copy()
        return board, gain

    def render(self) -> str:
        lines = ["┌──────┬──────┬──────┬──────┐"]
        for r in range(4):
            cells = "│".join(
                f"{int(self.board[r, c]) if self.board[r, c] else '.':^6}"
                for c in range(4)
            )
            lines.append(f"│{cells}│")
            if r < 3:
                lines.append("├──────┼──────┼──────┼──────┤")
        lines.append("└──────┴──────┴──────┴──────┘")
        return "\n".join(lines)


def apply_merge_board(board: np.ndarray, action: int) -> tuple[np.ndarray, int]:
    """对给定棋盘做一次压缩合并（不生成新块、不改原棋盘），返回 (新棋盘, 合并原始分)。"""
    work = np.rot90(board, K_MAP[action]).copy()
    gain = 0
    for r in range(4):
        row, g = Game2048._merge_row_left(work[r])
        gain += g
        work[r] = row
    return np.rot90(work, K_BACK[K_MAP[action]]).copy(), gain


def _board_heuristic(board: np.ndarray) -> float:
    """静态局面评估（log2 分值量纲，和训练奖励一致）。

    空格奖励（保灵活）+ 最大格在角落 + 行列单调性（取最佳朝向）
    + 相邻同值奖励（可立刻/近期合并的对齐，鼓励及时合并）
    - 相邻落差惩罚（平滑度，减少大格被小格切断）。
    """
    occupied = board > 0
    n_empty = 16 - int(occupied.sum())
    logs = np.zeros_like(board, dtype=float)
    logs[occupied] = np.log2(board[occupied].astype(float))

    empty_bonus = 0.4 * n_empty

    max_pos = np.unravel_index(int(np.argmax(logs)), logs.shape)
    corner_bonus = 0.0
    if max_pos in ((0, 0), (0, 3), (3, 0), (3, 3)):
        corner_bonus = 1.5 * float(logs[max_pos])

    # 单调性：每个行/列分别取两个朝向中较单调的一侧，只奖励正向
    mono = 0.0
    for axis in range(2):
        rolled = np.moveaxis(logs, axis, 0)
        diffs = rolled[:, :-1] - rolled[:, 1:]
        mono += float(np.clip(diffs.sum(axis=1), 0, None).sum())
        mono += float(np.clip(-diffs.sum(axis=1), 0, None).sum())

    # 相邻同值（含跨空格不算）：直接奖励，促使对齐相同方块
    align = 0.0
    for axis in range(2):
        rolled = np.moveaxis(board, axis, 0)
        for line in rolled:
            vals = line[line != 0]
            for i in range(len(vals) - 1):
                if vals[i] == vals[i + 1]:
                    align += float(np.log2(float(vals[i])))

    smooth_penalty = 0.0
    for axis in range(2):
        rolled = np.moveaxis(logs, axis, 0)
        smooth_penalty += float(np.abs(rolled[:, :-1] - rolled[:, 1:]).sum())

    return (
        empty_bonus + corner_bonus
        + 0.3 * mono + 0.5 * align
        - 0.05 * smooth_penalty
    )


# 老师采样随机源（固定种子保证可复现）
_teacher_rng = np.random.default_rng(7)


def _action_outcomes(board: np.ndarray):
    """生成所有合法动作的 (动作, 合并后棋盘, log2 合并分)。"""
    for action in range(N_ACTIONS):
        nb, gain = apply_merge_board(board, action)
        if np.array_equal(nb, board):
            continue
        yield action, nb, float(np.log2(max(1.0, float(gain))))


def _expectimax(board: np.ndarray, depth: int) -> float:
    """Max 节点：在剪枝后的合法动作中取最大「合并分 + chance 期望」。"""
    if depth <= 0:
        return _board_heuristic(board)
    outcomes = list(_action_outcomes(board))
    if not outcomes:
        return _board_heuristic(board)
    # 动作剪枝：先用"合并分+合并后启发式"浅排，只留 top N，控制深度3的开销
    if len(outcomes) > TEACHER_TOP_ACTIONS:
        outcomes.sort(
            key=lambda o: o[2] + _board_heuristic(o[1]), reverse=True
        )
        outcomes = outcomes[:TEACHER_TOP_ACTIONS]
    best = -np.inf
    for _, nb, lg in outcomes:
        best = max(best, lg + _chance_value(nb, depth))
    return best


def _chance_value(board_after: np.ndarray, depth: int) -> float:
    """Chance 节点：对随机新块（2:90%/4:10%）的位置采样求期望。"""
    empties = np.argwhere(board_after == 0)
    if len(empties) == 0:
        # 棋盘已满：无新块，直接继续下一层 max
        return _expectimax(board_after, depth - 1)
    n = min(TEACHER_SAMPLE_EMPTY, len(empties))
    if n < len(empties):
        idx = _teacher_rng.choice(len(empties), size=n, replace=False)
        positions = empties[idx]
    else:
        positions = empties
    total = 0.0
    for row, col in positions:
        step = 0.0
        for tile, prob in ((2, 0.9), (4, 0.1)):
            board2 = board_after.copy()
            board2[row, col] = tile
            step += prob * _expectimax(board2, depth - 1)
        total += step
    return total / float(n)


def teacher_choose(
    env: Game2048, legal: np.ndarray, depth: int = TEACHER_DEPTH
) -> tuple[int, float, float | None]:
    """高质量老师（深度 expectimax，默认 3 层动作预判）。

    depth 可随课程进度调整（auto 教学调度：早期 2、后期 3）。
    返回 (最佳动作, 最佳价值, 次佳价值)。次佳价值用于置信度门控。
    """
    scored: list[tuple[int, float]] = []
    for action in range(N_ACTIONS):
        if not legal[action]:
            continue
        board1, gain1 = env.peek_move(action)
        lg = float(np.log2(max(1.0, float(gain1))))
        scored.append((action, lg + _chance_value(board1, depth - 1)))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_action, best_val = scored[0]
    second_val = scored[1][1] if len(scored) > 1 else None
    return best_action, best_val, second_val


# --------------------------------------------------------------------------
# 棋盘 D4 对称性（教学数据增强用）
# --------------------------------------------------------------------------
# 动作序：0=up, 1=right, 2=down, 3=left。
# 每个变换 g(B) = rot90(fliplr(B) if flip else B, k)，动作置换 perm 满足
#     apply_merge(g(B), perm[a]) == g(apply_merge(B, a))
# 该表由 30 个真实中局棋盘做交换图经验枚举验证（见实现提交说明）。
# 顺序约定：K=1 取 [0]；K=2 取 [0,2]；K=4 取前 4；K=8 全取。
_D4_TRANSFORMS: list[tuple[bool, int, tuple[int, ...]]] = [
    (False, 0, (0, 1, 2, 3)),  # 恒等
    (False, 1, (3, 0, 1, 2)),  # 逆时针 90°
    (False, 2, (2, 3, 0, 1)),  # 180°
    (False, 3, (1, 2, 3, 0)),  # 逆时针 270°
    (True, 0, (0, 3, 2, 1)),   # 水平翻转
    (True, 1, (3, 2, 1, 0)),   # 水平翻转 + 逆时针 90°
    (True, 2, (2, 1, 0, 3)),   # 水平翻转 + 180°
    (True, 3, (1, 0, 3, 2)),   # 水平翻转 + 逆时针 270°
]
_D4_SUBSET_INDEX = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3), 8: tuple(range(8))}


def d4_subset(k: int) -> list[tuple[bool, int, tuple[int, ...]]]:
    """按增强倍数 K 返回 (是否水平翻转, rot90_k, 动作置换) 列表。"""
    if k not in _D4_SUBSET_INDEX:
        raise ValueError(f"TEACH_AUG_K 只支持 1/2/4/8，收到 {k}")
    return [_D4_TRANSFORMS[i] for i in _D4_SUBSET_INDEX[k]]


def transform_board_4x4(
    board: np.ndarray, use_flip: bool, k: int
) -> np.ndarray:
    """对 4×4 棋盘做 D4 变换（先可选水平翻转，再逆时针旋转 k×90°）。"""
    work = np.fliplr(board) if use_flip else board
    return np.asarray(np.rot90(work, k)).copy()


def transform_legal(
    legal: np.ndarray, perm: tuple[int, ...]
) -> np.ndarray:
    """动作合法性随几何变换置换：tlegal[perm[a]] = legal[a]。"""
    inv = [0] * N_ACTIONS
    for a in range(N_ACTIONS):
        inv[perm[a]] = a
    return legal[np.asarray(inv)]


def teach_schedule(
    cycle: int, cycles: int
) -> tuple[int, int]:
    """auto 第 cycle 轮（1-based）的教学调度：返回 (教学局数, 老师深度)。

    以该轮中点的课程进度 p=(cycle-0.5)/cycles 在线性曲线上取值，
    保证先验（教学）覆盖与课程节点对齐：基本合并阶段老师浅而密，
    蛇形链/角落阶段老师深而精。
    """
    p = (cycle - 0.5) / float(cycles)
    p = float(np.clip(p, 0.0, 1.0))
    n = TEACH_N_EARLY + (TEACH_N_LATE - TEACH_N_EARLY) * p
    depth = TEACH_DEPTH_EARLY if p < 0.5 else TEACH_DEPTH_LATE
    return max(TEACH_N_LATE, min(TEACH_N_EARLY, int(round(n)))), depth


# --------------------------------------------------------------------------
# 果蝇大脑：Brian2 连接组子图（固定线路）+ 可学习读出层
# --------------------------------------------------------------------------
def board_to_rates(board_flat: np.ndarray) -> np.ndarray:
    """棋盘编码：空格 0Hz，格值越大放电越快，封顶 MAX_RATE_HZ。"""
    rates = np.zeros(INPUT_COUNT, dtype=float)
    for i, value in enumerate(board_flat):
        if value > 0:
            rates[i] = min(MAX_RATE_HZ, BASE_RATE_HZ * (1.0 + math.log2(float(value))))
    return rates


def layout_health(board: np.ndarray) -> float:
    """棋盘布局健康度（越大越好），三项合成：

    空格数（0.1/格，最多 1.6）—— 棋盘越灵活越好；
    最大格在角落（+2.0）—— 高分局的基本盘；
    蛇形单调性（0-1）—— 沿蛇形路径格值不增的相邻对比例，
        衡量棋盘是否形成从大到小的"链"，避免大格被小格围死。
    """
    empty = float((board == 0).sum())
    max_val = float(board.max())
    corner = 0.0
    if max_val > 0 and max_val in (
        float(board[0, 0]), float(board[0, 3]),
        float(board[3, 0]), float(board[3, 3]),
    ):
        corner = 1.0
    # 蛇形展开：第一行左→右，第二行右→左 ...
    snake = np.concatenate(
        [board[0], board[1][::-1], board[2], board[3][::-1]]
    ).astype(float)
    diffs = np.diff(snake)
    # 只对非零段计分：空格(0)之间的"单调"没有信息，用有效相邻对做分母
    valid = (snake[:-1] > 0) | (snake[1:] > 0)
    if valid.sum() > 0:
        monotonic = float((diffs[valid] <= 0).sum()) / float(valid.sum())
    else:
        monotonic = 0.0
    return 0.1 * empty + 2.0 * corner + monotonic


class FlyBrain:
    """固定连接组子图：16 感觉输入 → 166 LIF → 每 100ms 返回逐神经元 spike 计数。"""

    def __init__(self, nodes_path: Path, edges_path: Path, seed: int) -> None:
        self.nodes = pd.read_csv(nodes_path)
        self.edges = pd.read_csv(edges_path)

        body_id_to_index = {
            int(body_id): i for i, body_id in enumerate(self.nodes["body_id"].astype(int))
        }
        self.n_neurons = len(self.nodes)
        self.center_index = body_id_to_index[CENTER_BODY_ID]

        input_indices, _ = stage5.select_input_targets(
            self.nodes, self.edges, CENTER_BODY_ID, INPUT_COUNT
        )
        self.input_indices = np.asarray(input_indices, dtype=int)

        source_indices = np.asarray(
            [body_id_to_index[int(v)] for v in self.edges["source"]], dtype=int
        )
        target_indices = np.asarray(
            [body_id_to_index[int(v)] for v in self.edges["target"]], dtype=int
        )
        normalized_weights, _ = stage5.normalize_weights(
            self.edges["weight"].to_numpy()
        )

        b2.prefs.codegen.target = "numpy"
        b2.start_scope()
        b2.defaultclock.dt = DT_MS * b2.ms
        np.random.seed(seed)
        b2.seed(seed)

        equations = """
        dv/dt = (v_rest - v) / tau : volt (unless refractory)
        scount : 1
        """
        self.neurons = b2.NeuronGroup(
            self.n_neurons,
            equations,
            threshold="v > v_threshold",
            reset="v = v_reset; scount += 1",
            refractory=stage5.REFRACTORY_MS * b2.ms,
            method="exact",
            namespace={
                "v_rest": stage5.V_REST_MV * b2.mV,
                "v_reset": stage5.V_RESET_MV * b2.mV,
                "v_threshold": stage5.V_THRESHOLD_MV * b2.mV,
                "tau": stage5.TAU_MS * b2.ms,
            },
        )
        self.neurons.v = stage5.V_RESET_MV * b2.mV

        self.recurrent = b2.Synapses(
            self.neurons, self.neurons, model="w : volt", on_pre="v += w"
        )
        self.recurrent.connect(i=source_indices, j=target_indices)
        self.recurrent.w = (
            normalized_weights * stage5.BASE_SYNAPTIC_WEIGHT_MV * b2.mV
        )

        self.sensory = b2.PoissonGroup(INPUT_COUNT, rates=0 * b2.Hz)
        self.input_synapses = b2.Synapses(
            self.sensory, self.neurons, model="w : volt", on_pre="v += w"
        )
        self.input_synapses.connect(
            i=np.arange(INPUT_COUNT), j=self.input_indices
        )
        self.input_synapses.w = stage5.STIMULUS_WEIGHT_MV * b2.mV

        self.network = b2.Network(
            self.neurons, self.recurrent, self.sensory, self.input_synapses
        )

    def run_step(self, step_rates: np.ndarray) -> np.ndarray:
        """跑一个 100ms 循环，返回 166 个神经元的 spike 计数（每步独立重置膜电位）。

        注意：参数名不能叫 rates，否则会与 PoissonGroup 内部变量重名，
        Brian2 会在 namespace 解析时刷警告。
        """
        step_rates = np.clip(step_rates, 0.0, MAX_RATE_HZ)
        self.neurons.v = stage5.V_RESET_MV * b2.mV
        self.neurons.scount = 0.0
        self.sensory.rates = step_rates * b2.Hz
        self.network.run(WINDOW_MS * b2.ms)
        return np.asarray(self.neurons.scount).copy()


def board_summary(board_flat: np.ndarray) -> np.ndarray:
    """棋盘标量摘要（3 维），补充脉冲特征丢失的绝对盘面信息：

    [0] log2 格值总和（归一化）—— 盘面整体规模/推进阶段；
    [1] 最大格量级 log2(max)/11 —— 高阶合并进度；
    [2] 空格比例 —— 局面灵活度。
    """
    occupied = board_flat > 0
    n_empty = int((~occupied).sum())
    if occupied.any():
        logs = np.log2(board_flat[occupied].astype(float))
        log_sum = float(logs.sum()) / BOARD_LOG_SUM_SCALE
        max_log = float(logs.max()) / 11.0
    else:
        log_sum = 0.0
        max_log = 0.0
    return np.array(
        [min(log_sum, 1.5), min(max_log, 1.5), n_empty / float(INPUT_COUNT)],
        dtype=float,
    )


class FeatureBuffer:
    """滚动特征缓冲：保留最近 K 个脑循环的放电模式与强度。

    每窗包含：
      - 归一化放电率（n_neurons 维，除以窗内 spike 总数 → 相对模式）；
      - log 总 spike 数（1 维 → 绝对强度/盘面规模）。
    另在最新窗附带当前棋盘摘要 board_summary（3 维）。

    reset 时历史帧填零（开局没有过去）；每步 push 当前窗计数，
    features() 按「旧→新」拼接，长度 = K × n_neurons + K + 3。
    """

    def __init__(self, n_neurons: int, windows: int = TIME_WINDOWS):
        self.n_neurons = n_neurons
        self.windows = windows
        self.frames: deque[np.ndarray] = deque(maxlen=windows)
        self.totals: deque[float] = deque(maxlen=windows)
        self.summary = np.zeros(BOARD_EXTRA_DIM, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.frames.clear()
        self.totals.clear()
        for _ in range(self.windows):
            self.frames.append(np.zeros(self.n_neurons))
            self.totals.append(0.0)
        self.summary = np.zeros(BOARD_EXTRA_DIM, dtype=float)

    @staticmethod
    def _normalize(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        if total <= 0:
            return np.zeros_like(counts, dtype=float)
        return counts.astype(float) / float(total)

    @staticmethod
    def _total_feature(counts: np.ndarray) -> float:
        return float(
            np.clip(
                np.log1p(float(counts.sum())) / SPIKE_TOTAL_SCALE, 0.0, 2.0
            )
        )

    def push(
        self, counts: np.ndarray, board_flat: np.ndarray | None = None
    ) -> np.ndarray:
        self.frames.append(self._normalize(counts))
        self.totals.append(self._total_feature(counts))
        if board_flat is not None:
            self.summary = board_summary(board_flat)
        return self.features()

    def features(self) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(list(self.frames)),
                np.asarray(self.totals, dtype=float),
                self.summary,
            ]
        )


class ReadoutAgent:
    """4 方向 Actor + Critic（A2C），可选一层 ReLU 共享隐层。

    线性模式 (hidden=0)：
        logits = W·f + bias； V(s) = v_w·f + v_b
    MLP 模式 (hidden>0)：
        h = ReLU(W1·f + b1)（Actor/Critic 共享 trunk）
        logits = W2·h + bias； V(s) = v2·h + v_b

    Actor 经动作屏蔽 softmax 选方向；Critic 逐状态估折扣回报，
    advantage = G − V(s)（比全局滑动平均 baseline 精确）。
    """

    def __init__(
        self, n_features: int, rng: np.random.Generator,
        hidden: int = HIDDEN_SIZE,
    ):
        self.rng = rng
        self.n_features = n_features
        self._alloc(hidden)

    def _alloc(self, hidden: int) -> None:
        """按结构分配参数；Critic 零初始化，起步 V=0 等价无基线。"""
        self.hidden = int(hidden)
        h_size, f_size = self.hidden, self.n_features
        if self.hidden > 0:
            # trunk 用 1/sqrt(F) 小尺度初始化（输入已归一化，避免开局 ReLU 饱和）
            self.w1 = self.rng.normal(
                0.0, 1.0 / max(1.0, f_size ** 0.5), size=(h_size, f_size)
            )
            self.b1 = np.zeros(h_size)
            self.w2 = self.rng.normal(0.0, 0.1, size=(N_ACTIONS, h_size))
            self.bias = self.rng.normal(0.0, 0.1, size=N_ACTIONS)
            self.v2 = np.zeros(h_size)
            self.v_b = 0.0
        else:
            self.w = self.rng.normal(0.0, 0.1, size=(N_ACTIONS, f_size))
            self.bias = self.rng.normal(0.0, 0.1, size=N_ACTIONS)
            self.v_w = np.zeros(f_size)
            self.v_b = 0.0

    def forward(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """前向，返回 (logits, value, cache)；cache 供反向传播复用中间量。"""
        if self.hidden > 0:
            z1 = self.w1 @ features + self.b1
            h = np.maximum(0.0, z1)
            logits = self.w2 @ h + self.bias
            # V 物理硬裁剪：即使权重意外变大，输出也不进入自激区
            value = float(np.clip(
                self.v2 @ h + self.v_b, VALUE_CLIP_MIN, VALUE_CLIP_MAX
            ))
            return logits, value, {"f": features, "z1": z1, "h": h}
        value = float(np.clip(
            self.v_w @ features + self.v_b, VALUE_CLIP_MIN, VALUE_CLIP_MAX
        ))
        logits = self.w @ features + self.bias
        return logits, value, {"f": features, "z1": None, "h": features}

    def logits(self, features: np.ndarray) -> np.ndarray:
        logits, _, _ = self.forward(features)
        return logits

    def value(self, features: np.ndarray) -> float:
        _, value, _ = self.forward(features)
        return value

    def masked_softmax(
        self, logits: np.ndarray, temperature: float,
        legal: np.ndarray | None = None,
    ) -> np.ndarray:
        """带动作屏蔽的 softmax：非法方向 logit 置 -inf，概率质量只在合法方向。"""
        if legal is None:
            legal = np.ones(N_ACTIONS, dtype=bool)
        masked = np.where(legal, logits, -np.inf)
        z = (masked - masked.max()) / max(temperature, 1e-3)
        exp = np.where(legal, np.exp(z), 0.0)
        total = exp.sum()
        return exp / total if total > 0 else legal.astype(float) / max(1, int(legal.sum()))

    def sample(
        self, features: np.ndarray, temperature: float,
        legal: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        """返回 (action, probs, cache)；cache 含前向中间量与该状态 V 值。"""
        logits, value, cache = self.forward(features)
        probs = self.masked_softmax(logits, temperature, legal)
        action = int(self.rng.choice(N_ACTIONS, p=probs))
        cache["value"] = value
        return action, probs, cache

    def greedy(
        self, features: np.ndarray, legal: np.ndarray | None = None
    ) -> tuple[int, np.ndarray]:
        logits = self.logits(features)
        probs = self.masked_softmax(logits, 1.0, legal)
        action = int(np.argmax(probs))
        return action, probs

    @staticmethod
    def _clip_trunk_grad(grad: np.ndarray) -> np.ndarray:
        """共享 trunk 梯度 L2 范数裁剪：截断幅度、保留方向；非有限值置零跳过。"""
        if not np.isfinite(grad).all():
            return np.zeros_like(grad)
        norm = float(np.linalg.norm(grad))
        if norm > TRUNK_GRAD_CLIP:
            grad = grad * (TRUNK_GRAD_CLIP / norm)
        return grad

    def _backward_actor(
        self, cache: dict[str, Any], d_logits: np.ndarray, lr: float
    ) -> None:
        """Actor（策略梯度/监督交叉熵）反向，梯度上升；共享 trunk 一并更新。"""
        if not np.isfinite(d_logits).all():
            return
        h = cache["h"]
        if self.hidden > 0:
            # d_h 必须在更新 w2 之前用旧权重计算
            d_h = self.w2.T @ d_logits
            self.w2 += lr * np.outer(d_logits, h)
            self.bias += lr * d_logits
            d_z1 = self._clip_trunk_grad(d_h * (cache["z1"] > 0))
            self.w1 += lr * np.outer(d_z1, cache["f"])
            self.b1 += lr * d_z1
        else:
            self.w += lr * np.outer(d_logits, h)
            self.bias += lr * d_logits

    def _backward_critic(
        self, cache: dict[str, Any], td_err: float, lr: float
    ) -> None:
        """Critic（0.5(G-V)^2）反向，梯度上升；共享 trunk 一并更新。"""
        # 有界 TD：防止 V 失配经 v2↔共享 trunk 正反馈指数爆炸
        td_err = float(np.clip(td_err, -CRITIC_TD_CLIP, CRITIC_TD_CLIP))
        if not np.isfinite(td_err):
            return
        h = cache["h"]
        if self.hidden > 0:
            d_h = self.v2 * td_err
            # critic 头步长同样裁范数：正常训练初期 h 很小远低于阈值，
            # 只有自激时才触发，防止 v2 自身无界累积
            d_v2 = self._clip_trunk_grad(td_err * h)
            self.v2 += lr * d_v2
            self.v_b += lr * td_err
            # 关键：进共享 w1 的梯度必须裁范数，否则 v2 越大→w1 步长越大→
            # h 越大→v2·h 越大，构成指数正反馈
            d_z1 = self._clip_trunk_grad(d_h * (cache["z1"] > 0))
            self.w1 += lr * np.outer(d_z1, cache["f"])
            self.b1 += lr * d_z1
        else:
            d_vw = self._clip_trunk_grad(td_err * h)
            self.v_w += lr * d_vw
            self.v_b += lr * td_err

    @staticmethod
    def _entropy_grad(probs: np.ndarray) -> tuple[np.ndarray, float]:
        """熵 H=-Σp log p 对 logits 的梯度：p_k(H - log p_k)，同时返回熵值。"""
        logp = np.zeros(N_ACTIONS)
        nz = probs > 0.0
        logp[nz] = np.log(probs[nz])
        entropy = float(-(probs[nz] * logp[nz]).sum())
        return probs * (entropy - logp), entropy

    def update(
        self,
        cache: dict[str, Any],
        action: int, probs: np.ndarray, advantage: float, lr: float,
    ) -> None:
        """纯监督/REINFORCE 更新（教学用）：∇log π · advantage。"""
        d_logits = -probs.copy()
        d_logits[action] += 1.0
        d_logits *= advantage
        self._backward_actor(cache, d_logits, lr)

    def update_ac(
        self,
        cache: dict[str, Any], action: int, probs: np.ndarray,
        target: float, actor_lr: float, critic_lr: float,
        entropy_coef: float = ENTROPY_END,
    ) -> float:
        """A2C 单步更新，返回该步 advantage（供日志统计）。

        Critic：V(s) 朝 Monte Carlo 目标 G 回归（MSE）；
        Actor：∇log π(a) · (G − V(s)) ＋ 熵正则项 ∇H；
        熵系数随训练退火（run_training 传入），且当策略熵低于
        ENTROPY_FLOOR_H 时临时放大 ENTROPY_KICK 倍，防止过早固化。
        """
        value = cache.get("value")
        if value is None:
            value = self.value(cache["f"])
        advantage = float(np.clip(target - value, -ADV_CLIP, ADV_CLIP))

        # Critic 更新（损失 0.5(G-V)^2，上升用正号）
        self._backward_critic(cache, target - float(value), critic_lr)

        # Actor 策略梯度
        d_logits = -probs.copy()
        d_logits[action] += 1.0
        d_logits *= advantage

        # 熵正则：温度触底后仍靠它维持探索；熵过低则加强"踢"一下
        d_h, entropy = self._entropy_grad(probs)
        coef = entropy_coef
        if entropy < ENTROPY_FLOOR_H:
            coef *= ENTROPY_KICK
        d_logits += coef * d_h

        self._backward_actor(cache, d_logits, actor_lr)
        return advantage

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "bias": self.bias,
            "v_b": np.asarray(self.v_b),
            "windows": np.asarray(TIME_WINDOWS),
            "hidden": np.asarray(self.hidden),
            "version": np.asarray(2),
        }
        if self.hidden > 0:
            payload.update(
                w1=self.w1, b1=self.b1, w2=self.w2, v2=self.v2,
            )
        else:
            payload.update(w=self.w, v_w=self.v_w)
        np.savez(path, **payload)

    def load(self, path: Path) -> None:
        data = np.load(path)
        file_hidden = int(data["hidden"]) if "hidden" in data else 0
        # 结构以权重文件为准：加载旧线性权重时本实例自动切换为线性模式
        if file_hidden != self.hidden:
            self._alloc(file_hidden)

        if file_hidden > 0:
            if "w1" not in data or data["w1"].shape[1] != self.n_features:
                raise ValueError(
                    f"{path} 的 MLP 权重维度与当前特征维 "
                    f"{self.n_features} 不匹配，无法加载"
                )
            self.w1 = data["w1"]
            self.b1 = data["b1"]
            self.w2 = data["w2"]
            self.bias = data["bias"]
            self.v2 = data["v2"] if "v2" in data else np.zeros(self.hidden)
            self.v_b = float(data["v_b"]) if "v_b" in data else 0.0
            return

        if "w" not in data:
            raise ValueError(f"{path} 缺少线性 Actor 权重 w，无法加载")
        old_w = data["w"]
        old_dim = old_w.shape[1]
        n_neurons = (
            (self.n_features - TIME_WINDOWS - BOARD_EXTRA_DIM) // TIME_WINDOWS
        )

        def embed_neuron_blocks(old_vec: np.ndarray) -> np.ndarray:
            """把旧 K_old 窗神经元块嵌入新特征（对齐到最近窗；新增维度置零）。"""
            if old_vec.ndim == 2:
                new_vec = np.zeros((N_ACTIONS, self.n_features))
            else:
                new_vec = np.zeros(self.n_features)

            def put(target_block: int, source: np.ndarray) -> None:
                start = target_block * n_neurons
                end = start + n_neurons
                if old_vec.ndim == 2:
                    new_vec[:, start:end] = source
                else:
                    new_vec[start:end] = source

            if old_dim == n_neurons:
                # 旧单窗权重 → 放到「最近窗」块，历史窗置零
                put(TIME_WINDOWS - 1, old_vec)
            elif old_dim % n_neurons == 0:
                old_k = old_dim // n_neurons
                if 1 <= old_k <= TIME_WINDOWS:
                    for j in range(old_k):
                        src = (
                            old_vec[:, j * n_neurons:(j + 1) * n_neurons]
                            if old_vec.ndim == 2
                            else old_vec[j * n_neurons:(j + 1) * n_neurons]
                        )
                        put(TIME_WINDOWS - old_k + j, src)
                else:
                    raise ValueError(f"权重窗数 {old_k} 超出当前 {TIME_WINDOWS}")
            else:
                raise ValueError(
                    f"权重维度不匹配: W 列数 {old_dim}，当前结构需要 "
                    f"{self.n_features}（{TIME_WINDOWS}×{n_neurons}"
                    f"+{TIME_WINDOWS}+{BOARD_EXTRA_DIM}），"
                    f"且无法按每窗 {n_neurons} 维解释旧权重"
                )
            return new_vec

        if old_dim == self.n_features:
            self.w = old_w
            self.v_w = data["v_w"] if "v_w" in data else np.zeros(self.n_features)
        else:
            # 跨特征版本热启动：旧神经元权重对齐到新窗块，
            # 总放电强度/棋盘摘要等新维度置零，行为与旧策略近似、新特征逐步学起
            self.w = embed_neuron_blocks(old_w)
            self.v_w = (
                embed_neuron_blocks(data["v_w"]) if "v_w" in data
                else np.zeros(self.n_features)
            )
        self.bias = data["bias"]
        self.v_b = float(data["v_b"]) if "v_b" in data else 0.0


# --------------------------------------------------------------------------
# 单局（一个 episode）
# --------------------------------------------------------------------------
def curriculum_factors(progress: float) -> tuple[float, float, float]:
    """课程学习塑形因子（progress∈[0,1]，0=训练最早期，1=训练结束）。

    返回 (合并奖励倍数, 步数成本倍数, 健康度权重倍数)：
    早期放大合并奖励、降低步数成本以快速建立"多合并"基本策略并多探索；
    后期全部回到 1.0，由真实奖励 + 布局健康度驱动高阶蛇形链策略。
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    merge_mult = 1.0 + CURRICULUM_MERGE_BOOST * (1.0 - progress)
    step_mult = CURRICULUM_STEP_MIN + (1.0 - CURRICULUM_STEP_MIN) * progress
    health_mult = CURRICULUM_HEALTH_MIN + (1.0 - CURRICULUM_HEALTH_MIN) * progress
    return merge_mult, step_mult, health_mult


def episode_pg(
    brain: FlyBrain,
    agent: ReadoutAgent,
    env: Game2048,
    temperature: float,
    episode: int = -1,
    live_path: Path | None = None,
    extra: dict[str, Any] | None = None,
    progress: float = 1.0,
) -> dict[str, Any]:
    """跑一局并收集 (cache, 动作, 概率, 奖励)，随后由 run_training 做 A2C 更新。

    progress 为课程学习进度（0→1），控制奖励塑形强度。
    """
    env.reset()
    trajectories: list[tuple[dict[str, Any], int, np.ndarray, float]] = []
    step_times: list[float] = []
    buffer = FeatureBuffer(brain.n_neurons)   # 滚动记忆，reset 时历史为零
    merge_mult, step_mult, health_mult = curriculum_factors(progress)

    def write_live(steps: int, score: int, moves: int, done: bool) -> None:
        """每 50 步写一次实时状态，供训练控制台显示"当前操作"。"""
        if live_path is None:
            return
        payload: dict[str, Any] = {
            "episode": episode,
            "steps": steps,
            "score": score,
            "moves": moves,
            "invalid": steps - moves,
            "done": done,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        live_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )

    write_live(0, env.score, env.moves, False)

    # 动作屏蔽：每步只在合法方向上采样，非法步从源头不发生，
    # 因此每步棋盘必然改变、没有无效移动风暴，硬上限只作安全兜底。
    while not env.game_over() and len(trajectories) < MAX_STEPS_PER_EPISODE:
        before = time.perf_counter()
        legal = env.legal_actions()
        encoded_rates = board_to_rates(env.board_flat)
        counts = brain.run_step(encoded_rates)
        # 当前窗入缓冲，输出含最近 K 窗放电模式/强度 + 棋盘摘要
        features = buffer.push(counts, env.board_flat)
        step_times.append((time.perf_counter() - before) * 1000.0)

        action, probs, cache = agent.sample(features, temperature, legal)
        health_before = layout_health(env.board)
        moved, gain = env.move(action)

        # 奖励（课程塑形）：
        #   合并得分 log2 缩放 × merge_mult（早期放大，快速建立合并倾向）
        #   + HEALTH_W × health_mult × 布局健康度增量（后期强调蛇形链）
        #   + STEP_COST × step_mult（早期成本更小、鼓励探索布局）
        if moved:
            health_after = layout_health(env.board)
            reward = merge_mult * np.log2(max(1.0, float(gain)))
            reward += HEALTH_W * health_mult * (health_after - health_before)
        else:
            reward = INVALID_REWARD
        reward += STEP_COST * step_mult
        trajectories.append((cache, action, probs, reward))
        if len(trajectories) % 50 == 0:
            write_live(len(trajectories), env.score, env.moves, False)

    write_live(len(trajectories), env.score, env.moves, True)

    # 计算折扣回报 G_t
    returns = np.zeros(len(trajectories))
    running = 0.0
    for t in range(len(trajectories) - 1, -1, -1):
        running = trajectories[t][3] + GAMMA * running
        returns[t] = running

    stats = {
        "score": env.score,
        "max_tile": env.max_tile,
        "moves": env.moves,
        "steps": len(trajectories),
        "invalid": 0,
        "returns_mean": float(returns.mean()) if len(returns) else 0.0,
        "returns": returns,
    }
    stats["step_ms_mean"] = float(np.mean(step_times)) if step_times else 0.0
    return {
        **stats,
        "trajectories": trajectories,
    }


def teach_episode(
    brain: FlyBrain,
    agent: ReadoutAgent,
    env: Game2048,
    lr: float,
    episode: int = -1,
    live_path: Path | None = None,
    extra: dict[str, Any] | None = None,
    depth: int = TEACHER_DEPTH,
    aug_k: int = TEACH_AUG_K,
) -> dict[str, Any]:
    """老师玩一局，果蝇读出层每步监督模仿（交叉熵）＋ D4 对称增强。

    - 老师只在真实棋盘上决策一次（指定 expectimax 深度）；
    - K 路对称呈现各自跑脑、各自维护 3 窗 FeatureBuffer（历史几何连续）；
    - 门控只判真实路一次；通过则 K 路同时更新，目标动作按几何置换，
      每路学习率 lr/K（总梯度幅度与单路一致，方向为 K 路平均）；
    - 果蝇动作不采样、不影响棋盘，棋盘始终按老师动作推进。
    """
    env.reset()
    steps = 0
    agree = 0          # 有把握的步骤中，更新前果蝇 argmax 与老师一致的步数
    confident = 0      # 老师认为"有把握"而实际施教的步骤数
    aug_updates = 0    # 实际发生的增强路更新总数（= 施教步 × K）
    step_times: list[float] = []
    transforms = d4_subset(aug_k)
    buffers = [FeatureBuffer(brain.n_neurons) for _ in transforms]

    def write_live(done: bool) -> None:
        if live_path is None:
            return
        payload: dict[str, Any] = {
            "phase": "teach",
            "episode": episode,
            "steps": steps,
            "score": env.score,
            "moves": env.moves,
            "invalid": 0,
            "done": done,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        live_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )

    write_live(False)
    while not env.game_over() and steps < MAX_STEPS_PER_EPISODE:
        before = time.perf_counter()
        legal = env.legal_actions()
        teacher_action, best_val, second_val = teacher_choose(
            env, legal, depth=depth
        )

        # K 路对称脑流：恒等路固定为第 0 路（一致率统计以它为准）
        ways: list[tuple[dict[str, Any], np.ndarray, int]] = []
        for (use_flip, rot_k, perm), buf in zip(transforms, buffers):
            tboard = transform_board_4x4(env.board, use_flip, rot_k)
            tflat = tboard.reshape(-1)
            counts = brain.run_step(board_to_rates(tflat))
            features = buf.push(counts, tflat)
            tlegal = transform_legal(legal, perm)
            logits, _, cache = agent.forward(features)
            probs = agent.masked_softmax(logits, 1.0, tlegal)
            ways.append((cache, probs, perm[teacher_action]))
        step_times.append((time.perf_counter() - before) * 1000.0)

        # 置信度门控：唯一合法方向必学；多个方向时，最佳与次佳的相对差距
        # 小于 TEACH_MARGIN 说明局面两可，不强行灌输，跳过本次更新
        is_confident = True
        if second_val is not None:
            denom = abs(best_val) + 1.0
            is_confident = (best_val - second_val) / denom >= TEACH_MARGIN
        if is_confident:
            confident += 1
            _, probs0, taction0 = ways[0]
            if int(np.argmax(probs0)) == taction0:
                agree += 1
            # K 路各做交叉熵反传，lr/K 平均；
            # 监督梯度 = one_hot(置换后老师动作) - probs（线性/MLP 统一反传）
            way_lr = lr / float(len(transforms))
            for cache, probs, taction in ways:
                agent.update(cache, taction, probs, 1.0, way_lr)
                aug_updates += 1

        # 棋盘始终按老师动作走（必然合法），保证局面推进
        env.move(teacher_action)
        steps += 1
        if steps % 50 == 0:
            write_live(False)

    write_live(True)
    return {
        "score": env.score,
        "max_tile": env.max_tile,
        "moves": env.moves,
        "steps": steps,
        "confident_steps": confident,
        "agreement": agree / confident if confident else 0.0,
        "aug_updates": aug_updates,
        "step_ms_mean": float(np.mean(step_times)) if step_times else 0.0,
    }


def write_results_summary(results_path: Path, log_path: Path) -> None:
    """从 training_log.jsonl 汇总当前实验成绩到 results.json（每次 checkpoint 刷新）。"""
    records: list[dict[str, Any]] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return
    last = records[-1]
    scores = [float(r.get("score", 0.0)) for r in records]
    tiles = [int(r.get("max_tile", 0)) for r in records]
    moves = [int(r.get("moves", 0)) for r in records]
    recent = scores[-100:]
    summary = {
        "episode": int(last.get("episode", len(records))),
        "score": float(last.get("score", 0.0)),
        "max_tile": int(last.get("max_tile", 0)),
        "moves": int(last.get("moves", 0)),
        "average_score": round(float(np.mean(scores)), 2),
        "average_score_recent100": round(float(np.mean(recent)), 2),
        "best_score": float(np.max(scores)),
        "best_max_tile": int(np.max(tiles)),
        "average_moves": round(float(np.mean(moves)), 2),
        "episodes_recorded": len(records),
        "updated_at": datetime.now().isoformat(),
    }
    results_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def run_training(
    brain: FlyBrain,
    agent: ReadoutAgent,
    rng: np.random.Generator,
    episodes: int,
    lr: float,
    baseline: float,
    temperature: float,
    log_path: Path,
    start_episode: int = 1,
    recent_init: list[float] | None = None,
    weights_path: Path | None = None,
    save_every: int = 5,
    extra: dict[str, Any] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    checkpoint_dir: Path | None = None,
    results_path: Path | None = None,
) -> tuple[float, float]:
    """训练循环（A2C：Critic 基线 + 熵正则 + 课程塑形），返回 (baseline, temperature)。

    每局结束后：
      1. 倒序计算 Monte Carlo 折扣回报 G_t（γ=0.99）；
      2. 逐状态 advantage = G_t − V(s)（Critic），Critic 朝 G_t 回归；
      3. Actor 做 ∇log π·adv ＋ 熵正则更新（熵系数按训练进度退火、带熵下限）；
      4. 课程进度 progress 控制本局奖励塑形强度；
      5. softmax 温度每局退火；每 save_every 局存一次 checkpoint。
    动作屏蔽保留：采样只在合法方向上进行，非法步从不发生。
    """
    env = Game2048(rng)
    live_path = log_path.with_name("live_episode.json")
    # recent_init 保留参数仅为兼容旧调用（main / run_auto）
    total = start_episode + episodes - 1
    for episode in range(start_episode, start_episode + episodes):
        local_index = episode - start_episode
        frac = local_index / max(1, episodes - 1) if episodes > 1 else 1.0
        progress = progress_start + (progress_end - progress_start) * frac
        result = episode_pg(
            brain, agent, env, temperature,
            episode=episode, live_path=live_path, extra=extra,
            progress=progress,
        )
        returns = result["returns"]
        # EMA 仅用于日志展示；A2C 的实际 baseline 是 Critic 的 V(s)
        ep_mean = float(returns.mean()) if len(returns) else 0.0
        baseline = BASELINE_EMA * baseline + (1.0 - BASELINE_EMA) * ep_mean

        # 熵系数随训练进度从 ENTROPY_START 退火到 ENTROPY_END（不为零），
        # 温度触底后仍能防策略坍缩；update_ac 内部另有熵下限自适应加码
        entropy_coef = (
            ENTROPY_END + (ENTROPY_START - ENTROPY_END) * (1.0 - progress)
        )

        # A2C 主更新：Critic 回归 G_t，Actor 用 advantage=G_t−V(s) + 熵正则
        adv_values: list[float] = []
        for (cache, action, probs, _), g in zip(result["trajectories"], returns):
            adv = agent.update_ac(
                cache, action, probs, float(g),
                lr, CRITIC_LR, entropy_coef,
            )
            adv_values.append(adv)

        temperature = max(TEMP_MIN, temperature * TEMP_DECAY)

        record = {
            "episode": episode,
            "score": result["score"],
            "max_tile": result["max_tile"],
            "moves": result["moves"],
            "invalid": result["invalid"],
            "step_ms_mean": round(result["step_ms_mean"], 3),
            "temperature": round(temperature, 4),
            "baseline": round(float(baseline), 4),
            "entropy_coef": round(entropy_coef, 4),
            "adv_mean": round(float(np.mean(adv_values)), 3) if adv_values else 0.0,
            "curriculum": round(progress, 3),
            # replayed 字段保留兼容控制台表格
            "replayed": 0,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 中途 checkpoint：每 save_every 局存一次权重，
        # 进程被意外杀掉时最多只丢 save_every 局，续训可从存档恢复；
        # checkpoint_dir 给定时额外按局号快照（长实验留痕），并刷新 results.json
        if weights_path is not None and save_every > 0 and episode % save_every == 0:
            agent.save(weights_path)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(
                    weights_path, checkpoint_dir / f"readout_ep{episode}.npz",
                )
            if results_path is not None:
                write_results_summary(results_path, log_path)

        if episode % 10 == 0 or episode == start_episode:
            print(
                f"episode {episode:>5}/{total} | "
                f"score {record['score']:>6} | max_tile {record['max_tile']:>5} | "
                f"moves {record['moves']:>4} | step {record['step_ms_mean']:.1f}ms | "
                f"T {record['temperature']:.3f} | ent {entropy_coef:.3f} | "
                f"cur {progress:.2f} | adv {record['adv_mean']:>6.2f}",
                flush=True,
            )
    if results_path is not None:
        write_results_summary(results_path, log_path)
    return baseline, temperature


def run_teach_batch(
    brain: FlyBrain,
    agent: ReadoutAgent,
    out_dir: Path,
    episodes: int,
    lr: float,
    base_seed: int,
    extra: dict[str, Any] | None = None,
    weights_path: Path | None = None,
    depth: int = TEACHER_DEPTH,
    aug_k: int = TEACH_AUG_K,
) -> int:
    """教学一个批次，返回本批之前已有的教学总局数（即起始编号-1）。

    独立 teach 命令与 auto 循环共用此函数。depth 控制老师搜索深度，
    aug_k 控制 D4 对称增强倍数（1=关闭）。
    """
    teach_log = out_dir / "teach_log.jsonl"
    live_path = out_dir / "live_episode.json"
    start_episode = 1
    if teach_log.exists():
        start_episode = (
            sum(
                1
                for line in teach_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            + 1
        )
    # 种子随教学局数偏移：每批都是全新棋盘，不重复课件
    env_rng = np.random.default_rng(base_seed + (start_episode - 1))
    env = Game2048(env_rng)
    print(
        f"开始教学 {episodes} 局（从第 {start_episode} 教学局起），"
        f"lr={lr}，深度={depth}，对称增强×{aug_k}，"
        f"game_seed={base_seed + start_episode - 1}",
        flush=True,
    )
    for i in range(episodes):
        episode = start_episode + i
        result = teach_episode(
            brain, agent, env, lr,
            episode=episode, live_path=live_path, extra=extra,
            depth=depth, aug_k=aug_k,
        )
        record = {
            "episode": episode,
            "score": result["score"],
            "max_tile": result["max_tile"],
            "moves": result["moves"],
            "steps": result["steps"],
            "confident_steps": result["confident_steps"],
            "agreement": round(result["agreement"], 4),
            "aug_updates": result["aug_updates"],
            "depth": depth,
            "aug_k": aug_k,
            "step_ms_mean": round(result["step_ms_mean"], 3),
        }
        with teach_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"teach {episode:>3} | score {record['score']:>6} | "
            f"max_tile {record['max_tile']:>5} | steps {record['steps']:>4} | "
            f"施教 {record['confident_steps']:>3} 步（×{aug_k}） | "
            f"一致率 {record['agreement'] * 100:5.1f}%",
            flush=True,
        )
    if weights_path is not None:
        agent.save(weights_path)
    return start_episode - 1


def _read_train_log_tail(log_path: Path) -> tuple[int, float, float, list[float]]:
    """读取训练日志尾部：返回 (末局号, baseline, temperature, 近期得分)。"""
    last_episode = 0
    baseline = 0.0
    temperature: float | None = None
    recent: list[float] = []
    if not log_path.exists():
        return 0, 0.0, TEMP_START, []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_episode = int(rec.get("episode", last_episode))
        baseline = float(rec.get("baseline", baseline))
        if rec.get("temperature") is not None:
            temperature = float(rec["temperature"])
        recent.append(float(rec.get("score", 0.0)))
    return (
        last_episode, baseline,
        temperature if temperature is not None else TEMP_START, recent,
    )


def run_auto(
    brain: FlyBrain,
    agent: ReadoutAgent,
    out_dir: Path,
    cycles: int,
    teach_episodes: int,
    practice_episodes: int,
    lr: float,
    base_seed: int,
    teach_schedule_on: bool = True,
    aug_k: int = TEACH_AUG_K,
) -> None:
    """自动循环：每轮「深度老师教学 N 局 → A2C 自主练习 M 局」，共 cycles 轮。

    教学默认与课程对齐（teach_schedule_on=True）：每轮局数 15→3 递减、
    老师深度 2→3 递增；关闭后每轮固定 teach_episodes 局、深度 TEACHER_DEPTH。
    D4 对称增强对两种模式都生效（aug_k=1 可关闭）。

    全程一个进程、权重持续累积；日志分别追加到 teach_log/training_log，
    live_episode 每步带 auto_phase/cycle，控制台实时显示所处轮次与阶段。
    """
    weights_path = out_dir / "readout.npz"
    # 练习日志必须与普通训练同一文件（.jsonl），表格/曲线才读得到；
    # 曾误写成 training_log.json 导致自动循环练习局"隐身"
    log_path = out_dir / "training_log.jsonl"
    live_path = out_dir / "live_episode.json"
    state_path = out_dir / "train_state.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_teach = out_dir / "readout_pre_teach.npz"
    if not pre_teach.exists():
        import shutil
        shutil.copyfile(weights_path, pre_teach)

    # 从训练日志尾部续接编号/温度/baseline，与续训语义一致
    last_ep, baseline, temperature, recent = _read_train_log_tail(log_path)
    next_episode = last_ep + 1
    if teach_schedule_on:
        plan = [
            teach_schedule(c, cycles) for c in range(1, cycles + 1)
        ]
        plan_desc = "，".join(
            f"第{c}轮({n}局/深{d})" for c, (n, d) in enumerate(plan, start=1)
        )
        print(
            f"自动循环开始：{cycles} 轮 ×（教学→练习 {practice_episodes}），"
            f"教学与课程对齐（对称增强×{aug_k}）：{plan_desc}；"
            f"练习从第 {next_episode} 局接续，T={temperature:.3f}",
            flush=True,
        )
    else:
        print(
            f"自动循环开始：{cycles} 轮 ×（教学 {teach_episodes}(深{TEACHER_DEPTH}) "
            f"→ 练习 {practice_episodes}），对称增强×{aug_k}，"
            f"练习从第 {next_episode} 局接续，T={temperature:.3f}",
            flush=True,
        )
    for cycle in range(1, cycles + 1):
        if teach_schedule_on:
            n_teach, teach_depth = teach_schedule(cycle, cycles)
        else:
            n_teach, teach_depth = teach_episodes, TEACHER_DEPTH
        state_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "mode": "auto",
                    "cycle": cycle, "cycles": cycles,
                    "teach_episodes": n_teach,
                    "teach_depth": teach_depth,
                    "aug_k": aug_k,
                    "practice_episodes": practice_episodes,
                    "started_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"━━━ 第 {cycle}/{cycles} 轮 · 教学阶段（{n_teach} 局，"
            f"深度 {teach_depth}，×{aug_k}）━━━", flush=True,
        )
        run_teach_batch(
            brain, agent, out_dir, n_teach, lr, base_seed,
            extra={
                "auto_phase": "teach", "cycle": cycle, "cycles": cycles,
            },
            weights_path=weights_path,
            depth=teach_depth, aug_k=aug_k,
        )

        print(f"━━━ 第 {cycle}/{cycles} 轮 · 练习阶段 ━━━", flush=True)
        # 课程进度跨轮连续：第 cycle 轮练习覆盖 [(cycle-1)/K, cycle/K]
        prog_start = (cycle - 1) / float(cycles)
        prog_end = cycle / float(cycles)
        baseline, temperature = run_training(
            brain, agent, np.random.default_rng(base_seed + 1000 + cycle),
            episodes=practice_episodes, lr=lr,
            baseline=baseline, temperature=temperature,
            log_path=log_path, start_episode=next_episode,
            recent_init=recent, weights_path=weights_path,
            save_every=5,
            extra={
                "auto_phase": "practice", "cycle": cycle, "cycles": cycles,
            },
            progress_start=prog_start,
            progress_end=prog_end,
        )
        next_episode += practice_episodes
        # 从刚追加的日志刷新近期得分，保证下一轮重放门控基于最新成绩
        _, _, _, recent = _read_train_log_tail(log_path)
        agent.save(weights_path)
        print(
            f"━━━ 第 {cycle} 轮完成，权重已保存 ━━━", flush=True,
        )

    live_path.write_text(
        json.dumps(
            {"phase": "auto-done", "updated_at": datetime.now().isoformat()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state_path.unlink(missing_ok=True)
    print(f"自动循环全部完成：{cycles} 轮，最终权重 {weights_path}")


def eval_greedy(
    brain: FlyBrain, agent: ReadoutAgent, rng: np.random.Generator, episodes: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """果蝇认真玩（greedy argmax），关掉探索，用于评估学习成绩。"""
    scores: list[int] = []
    max_tiles: list[int] = []
    moves: list[int] = []
    env = Game2048(rng)
    started = time.perf_counter()

    def report_progress(
        completed: int, current_episode: int, current_steps: int,
        current_score: int, current_max_tile: int,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback({
            "completed": completed,
            "total": episodes,
            "current_episode": current_episode,
            "current_steps": current_steps,
            "current_score": current_score,
            "current_max_tile": current_max_tile,
            "mean_score": float(np.mean(scores)) if scores else None,
            "max_score": int(max(scores)) if scores else None,
            "mean_max_tile": float(np.mean(max_tiles)) if max_tiles else None,
            "best_max_tile": int(max(max_tiles)) if max_tiles else None,
            "eval_elapsed_s": round(time.perf_counter() - started, 1),
            "eta_s": (
                round((time.perf_counter() - started) / completed * (episodes - completed))
                if completed else None
            ),
        })

    report_progress(0, 1, 0, 0, 0)
    for episode_index in range(episodes):
        env.reset()
        steps = 0
        buffer = FeatureBuffer(brain.n_neurons)
        while not env.game_over() and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            legal = env.legal_actions()
            encoded_rates = board_to_rates(env.board_flat)
            counts = brain.run_step(encoded_rates)
            features = buffer.push(counts, env.board_flat)
            action, _ = agent.greedy(features, legal)
            env.move(action)
            if steps % 25 == 0:
                report_progress(
                    episode_index, episode_index + 1, steps,
                    env.score, env.max_tile,
                )
        scores.append(env.score)
        max_tiles.append(env.max_tile)
        moves.append(env.moves)
        report_progress(
            episode_index + 1, episode_index + 2,
            0, 0, 0,
        )
    return {
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "best_max_tile": int(np.max(max_tiles)),
        "rate_256_plus": float(np.mean(np.asarray(max_tiles) >= 256)),
        "rate_512_plus": float(np.mean(np.asarray(max_tiles) >= 512)),
        "mean_moves": float(np.mean(moves)),
    }


def eval_random(rng: np.random.Generator, episodes: int) -> dict[str, Any]:
    """随机策略基线：不经过大脑，纯随机动作，用于对比学习效果。"""
    scores: list[int] = []
    max_tiles: list[int] = []
    moves: list[int] = []
    env = Game2048(rng)
    for _ in range(episodes):
        env.reset()
        while not env.game_over():
            env.move(int(rng.integers(N_ACTIONS)))
        scores.append(env.score)
        max_tiles.append(env.max_tile)
        moves.append(env.moves)
    return {
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "best_max_tile": int(np.max(max_tiles)),
        "mean_moves": float(np.mean(moves)),
    }


# --------------------------------------------------------------------------
# 报告与学习曲线
# --------------------------------------------------------------------------
def write_report(
    out_dir: Path, trained: dict[str, Any], random_stats: dict[str, Any]
) -> None:
    lines = [
        "# 果蝇玩 2048 · 评估报告",
        "",
        "策略            | 平均得分 | 最高得分 | 平均最大格 | 最佳最大格 | 平均步数",
        "--------------- | -------- | -------- | ---------- | ---------- | --------",
        f"果蝇(学习后)     | {trained['mean_score']:.1f} | {trained['max_score']} "
        f"| {trained['mean_max_tile']:.1f} | {trained['best_max_tile']} "
        f"| {trained['mean_moves']:.1f}",
        f"随机基线         | {random_stats['mean_score']:.1f} | {random_stats['max_score']} "
        f"| {random_stats['mean_max_tile']:.1f} | {random_stats['best_max_tile']} "
        f"| {random_stats['mean_moves']:.1f}",
        "",
    ]
    if random_stats["mean_score"] > 0:
        ratio = trained["mean_score"] / random_stats["mean_score"]
        lines.append(f"学习效果：平均得分是随机基线的 {ratio:.2f} 倍")
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


def plot_learning_curve(log_path: Path, out_path: Path) -> None:
    """从训练日志画学习曲线（score 与 max_tile 的滑动平均）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes: list[int] = []
    scores: list[int] = []
    tiles: list[int] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            episodes.append(record["episode"])
            scores.append(record["score"])
            tiles.append(record["max_tile"])

    def rolling(values: list[int], window: int) -> list[float]:
        result: list[float] = []
        acc: list[int] = []
        for value in values:
            acc.append(value)
            if len(acc) > window:
                acc.pop(0)
            result.append(sum(acc) / len(acc))
        return result

    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(episodes, scores, alpha=0.3, color="#0FDC78", lw=1)
    ax1.plot(episodes, rolling(scores, 25), color="#02743B", lw=2, label="score (rolling 25)")
    ax1.set_ylabel("score")
    ax1.legend()
    ax2.plot(episodes, tiles, alpha=0.3, color="#27D2BF", lw=1)
    ax2.plot(episodes, rolling(tiles, 25), color="#0F766E", lw=2, label="max tile (rolling 25)")
    ax2.set_ylabel("max tile")
    ax2.set_xlabel("episode")
    ax2.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)


# --------------------------------------------------------------------------
# 演示/试玩
# --------------------------------------------------------------------------
def play_once(
    brain: FlyBrain, agent: ReadoutAgent, rng: np.random.Generator,
    out_path: Path | None,
) -> None:
    env = Game2048(rng)
    env.reset()
    print("初始棋盘：")
    print(env.render())
    records: list[dict[str, Any]] = []
    buffer = FeatureBuffer(brain.n_neurons)
    while not env.game_over():
        legal = env.legal_actions()
        encoded_rates = board_to_rates(env.board_flat)
        counts = brain.run_step(encoded_rates)
        features = buffer.push(counts, env.board_flat)
        logits = agent.logits(features)
        action, probs = agent.greedy(features, legal)
        moved, gain = env.move(action)
        records.append({
            "step": env.moves,
            "board": env.board_flat.tolist(),
            "action": ACTION_NAMES[action],
            "moved": bool(moved),
            "gain": int(gain),
            "spike_total": int(counts.sum()),
            "probs": [round(float(p), 4) for p in probs],
        })
        print(
            f"{env.moves:>4} | 动作 {ACTION_NAMES[action]:<5} | "
            f"合并 {gain:>5} | 总spike {int(counts.sum()):>4} | "
            f"选择概率 {probs[action]:.3f}"
        )
        print(env.render())
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
    print(
        f"游戏结束：得分 {env.score} | 最大格 {env.max_tile} | 步数 {env.moves}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="让 Brian2 连接组子图玩 2048（Level 2 学习版）."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练读出层并评估")
    train_parser.add_argument("--episodes", type=int, default=300)
    train_parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    train_parser.add_argument("--final-eval-episodes", type=int, default=50,
                              help="训练后 greedy 评估局数（0 跳过）")
    train_parser.add_argument("--random-baseline-episodes", type=int, default=300,
                              help="随机基线局数（0 跳过）")
    train_parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    train_parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    train_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--seed", type=int, default=20261005)
    train_parser.add_argument("--resume", action="store_true",
                              help="在已有 readout.npz 与训练日志基础上继续训练")
    train_parser.add_argument("--save-every", type=int, default=5,
                              help="权重 checkpoint 间隔（局）")
    train_parser.add_argument("--checkpoint-dir", type=Path, default=None,
                              help="给定后每 save-every 局额外按局号快照到该目录")

    play_parser = subparsers.add_parser("play", help="加载权重试玩一局")
    play_parser.add_argument("--weights", type=Path, required=True,
                             help="readout.npz 权重文件")
    play_parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    play_parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    play_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    play_parser.add_argument("--seed", type=int, default=20261005)

    teach_parser = subparsers.add_parser("teach", help="expectimax 老师示范，读出层模仿 + D4 对称增强")
    teach_parser.add_argument("--episodes", type=int, default=10)
    teach_parser.add_argument("--lr", type=float, default=TEACH_LR)
    teach_parser.add_argument("--depth", type=int, default=TEACHER_DEPTH,
                              choices=[2, 3],
                              help="老师 expectimax 深度（4 过慢，约 0.75s/决策）")
    teach_parser.add_argument("--aug-k", type=int, default=TEACH_AUG_K,
                              choices=[1, 2, 4, 8],
                              help="D4 对称增强倍数：1 关闭/2 中心对称/4 旋转/8 旋转+翻转")
    teach_parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    teach_parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    teach_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    teach_parser.add_argument("--seed", type=int, default=20261005)

    random_parser = subparsers.add_parser("random", help="随机策略基线")
    random_parser.add_argument("--episodes", type=int, default=300)
    random_parser.add_argument("--seed", type=int, default=20261005)

    auto_parser = subparsers.add_parser(
        "auto", help="自动化循环：课程对齐教学（15→3局/深2→3）→ 练习 M 局，重复 K 轮",
    )
    auto_parser.add_argument("--cycles", type=int, default=5, help="循环轮数")
    auto_parser.add_argument("--teach-episodes", type=int, default=10,
                             help="每轮固定教学局数（仅 --no-teach-schedule 时生效）")
    auto_parser.add_argument("--practice-episodes", type=int, default=50,
                             help="每轮练习局数")
    auto_parser.add_argument("--no-teach-schedule", action="store_true",
                             help="关闭课程对齐调度，每轮固定教学局数与深度3")
    auto_parser.add_argument("--aug-k", type=int, default=TEACH_AUG_K,
                             choices=[1, 2, 4, 8],
                             help="教学 D4 对称增强倍数：1 关闭/2/4/8")
    auto_parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    auto_parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    auto_parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    auto_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    auto_parser.add_argument("--seed", type=int, default=20261005)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.command == "random":
        stats = eval_random(rng, args.episodes)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    brain = FlyBrain(args.nodes, args.edges, args.seed)
    print(
        f"大脑就绪：{brain.n_neurons} 神经元 / {len(brain.edges)} 连接，"
        f"16 个感觉输入 → 中心神经元 index={brain.center_index}"
    )

    if args.command == "train":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.out_dir / "training_log.jsonl"
        weights_path = args.out_dir / "readout.npz"

        # 续训：加载已有权重，从日志尾部继续编号，
        # 温度与 baseline 从日志末条恢复
        start_episode = 1
        init_baseline = 0.0
        init_temperature = TEMP_START
        last_temperature: float | None = None
        recent_init: list[float] = []
        if args.resume:
            if not weights_path.exists():
                print(f"没有可续训的权重：{weights_path}，请先完成一次 train。")
                return 1
            last_episode = 0
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    last_episode = int(rec.get("episode", last_episode))
                    init_baseline = float(rec.get("baseline", init_baseline))
                    if rec.get("temperature") is not None:
                        last_temperature = float(rec["temperature"])
                    recent_init.append(float(rec.get("score", 0.0)))
            start_episode = last_episode + 1
            # 优先沿用日志末局的真实温度（保证新旧退火策略切换不跳变）；
            # 无记录时退回按当前衰减公式重算
            if last_temperature is not None:
                init_temperature = max(TEMP_MIN, last_temperature)
            else:
                init_temperature = max(
                    TEMP_MIN, TEMP_START * TEMP_DECAY ** (start_episode - 1)
                )
            print(
                f"续训：从第 {start_episode} 局开始（权重 {weights_path.name}），"
                f"T={init_temperature:.3f}，baseline={init_baseline:.2f}"
            )
        elif log_path.exists():
            # 不勾续训=重新开始：旧日志绝不物理删除，轮转归档留底，
            # 避免误操作无法找回（之前 unlink 曾导致训练历史全部丢失）
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path.rename(
                log_path.with_name(f"training_log_reset_{stamp}.jsonl")
            )
            (args.out_dir / "eval_results.json").unlink(missing_ok=True)

        # 训练状态文件：供训练控制台（fly2048_lab.py）读取/停止本进程
        target_episodes = start_episode + args.episodes - 1
        state_path = args.out_dir / "train_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "episodes": target_episodes,
                    "out_dir": str(args.out_dir),
                    "started_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        agent = ReadoutAgent(feature_dim(brain.n_neurons), rng)
        if args.resume:
            agent.load(weights_path)
        start = time.perf_counter()
        run_training(
            brain, agent, rng,
            episodes=args.episodes,
            lr=args.lr,
            baseline=init_baseline,
            temperature=init_temperature,
            log_path=log_path,
            start_episode=start_episode,
            recent_init=recent_init,
            weights_path=weights_path,
            save_every=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            results_path=args.out_dir / "results.json",
        )
        elapsed = time.perf_counter() - start
        print(f"训练完成，用时 {elapsed / 60:.1f} 分钟")

        weights_path = args.out_dir / "readout.npz"
        agent.save(weights_path)
        print(f"权重已保存：{weights_path}")

        trained_stats: dict[str, Any] = {}
        live_path = args.out_dir / "live_episode.json"
        if args.final_eval_episodes > 0:
            # 阶段标记：评估中（供训练控制台显示）
            def write_eval_progress(progress: dict[str, Any]) -> None:
                live_path.write_text(
                    json.dumps(
                        {
                            "phase": "eval",
                            "updated_at": datetime.now().isoformat(),
                            **progress,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            print(f"greedy 评估 {args.final_eval_episodes} 局 ...")
            trained_stats = eval_greedy(
                brain, agent, rng, args.final_eval_episodes,
                progress_callback=write_eval_progress,
            )
        random_stats: dict[str, Any] = {}
        if args.random_baseline_episodes > 0:
            print(f"随机基线 {args.random_baseline_episodes} 局 ...")
            random_stats = eval_random(rng, args.random_baseline_episodes)
        # 评估结果落 JSON，供训练控制台常驻展示
        eval_payload: dict[str, Any] = {}
        if trained_stats:
            eval_payload["trained"] = trained_stats
        if random_stats:
            eval_payload["random"] = random_stats
        if eval_payload:
            eval_results_path = args.out_dir / "eval_results.json"
            eval_results_path.write_text(
                json.dumps(eval_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if trained_stats and random_stats:
            write_report(args.out_dir, trained_stats, random_stats)
        plot_learning_curve(log_path, args.out_dir / "learning_curve.png")
        print(f"学习曲线：{args.out_dir / 'learning_curve.png'}")

        live_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return 0

    if args.command == "play":
        agent = ReadoutAgent(feature_dim(brain.n_neurons), rng)
        agent.load(args.weights)
        demo_path = args.out_dir / "play_demo.jsonl"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if demo_path.exists():
            demo_path.unlink()
        play_once(brain, agent, rng, demo_path)
        print(f"演示记录：{demo_path}")
        return 0

    if args.command == "teach":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = args.out_dir / "readout.npz"
        if not weights_path.exists():
            print(f"没有可教的权重：{weights_path}，请先完成一次 train。")
            return 1

        # 教学前留一次权重快照，教坏可回退（run_teach_batch 只负责教与存）
        pre_teach = args.out_dir / "readout_pre_teach.npz"
        if not pre_teach.exists():
            import shutil
            shutil.copyfile(weights_path, pre_teach)

        agent = ReadoutAgent(feature_dim(brain.n_neurons), rng)
        agent.load(weights_path)
        run_teach_batch(
            brain, agent, args.out_dir, args.episodes, args.lr, args.seed,
            weights_path=weights_path,
            depth=args.depth, aug_k=args.aug_k,
        )
        (args.out_dir / "live_episode.json").unlink(missing_ok=True)
        print(f"教学完成，权重已保存：{weights_path}")
        return 0

    if args.command == "auto":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = args.out_dir / "readout.npz"
        agent = ReadoutAgent(feature_dim(brain.n_neurons), rng)
        if weights_path.exists():
            agent.load(weights_path)
        else:
            # 首次使用无权重：保存随机初值，run_auto 内建快照逻辑即可继续
            agent.save(weights_path)
        run_auto(
            brain, agent, args.out_dir,
            cycles=args.cycles,
            teach_episodes=args.teach_episodes,
            practice_episodes=args.practice_episodes,
            lr=args.lr, base_seed=args.seed,
            teach_schedule_on=not args.no_teach_schedule,
            aug_k=args.aug_k,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())