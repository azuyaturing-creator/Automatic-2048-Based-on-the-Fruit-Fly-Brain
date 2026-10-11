# Automatic 2048 Based on the Fruit Fly Brain

> 用**真实果蝇连接组**驱动一个脉冲神经网络玩 2048：
> 棋盘 16 格编码为感觉刺激 → MaleCNS 连接组子图（1452 个 LIF 神经元）放电扩散
> → 一个可学习的读出层把脑活动映射成 ↑→↓← 决策。
> 支持强化学习自练、expectimax 老师示范教学、以及**人在网页上亲自走棋当老师**。

这是一个本地科研/教学实验台：脑网络的拓扑与突触强度全部来自 Janelia FlyEM
公开发布的雄性果蝇 CNS 连接组数据，脉冲仿真由 Brian2 完成，不包含任何身体/运动仿真。

## 1. 它是怎么工作的

```text
2048 棋盘（4×4）
    │  每格 log2 数值 → 对应感觉神经元的泊松放电率（数值越大放电越快）
    ▼
16 个感觉输入神经元
    │  突触连接来自真实连接组（固定、不可学习）
    ▼
连接组 LIF 脉冲子图（默认 1452 神经元 / 2388 连接，Brian2 仿真）
    │  每个 100ms 脑循环统计一次各神经元放电率
    ▼
决策特征：最近 5 窗归一化放电率 + 5 窗总放电强度 + 3 维棋盘摘要（共 7268 维）
    │
    ▼
读出层（全系统唯一可学习部分）
    7268 → 64 ReLU 共享 trunk ──→ Actor：4 方向 softmax 概率（采样/贪心选动作）
                              └→ Critic：V(s) 状态价值基线
    ▼
执行方向 → 棋盘更新 → 奖励回流（A2C）
```

- **脑是固定的"湿件"**：连接组拓扑、突触权重、LIF 参数均不可学习，模拟先天神经回路。
- **只有读出层在学**：一个轻量 actor-critic MLP，负责"读懂"脑的放电模式。
- **奖励**：log2 缩放的合并得分（带课程塑形）+ 布局健康度增量；非法移动给负奖励。
- **学习算法**：Monte Carlo 折扣回报（γ=0.99）、Critic 基线减方差、温度/熵系数退火、
  课程学习；Critic 带 TD 裁剪、共享 trunk 梯度范数裁剪与输出物理界，防止自激发散。

## 2. 三种学习方式

| 方式 | 入口 | 说明 |
|---|---|---|
| **强化学习自练** | `train` / 网页控制台 | A2C 逐局自我对弈，温度从高到低退火 |
| **AI 老师示范** | `teach` / `auto` | expectimax 搜索（深度 3）给出老师动作，读出层做监督交叉熵模仿，支持 D4 对称增强（一局当 ×1/×2/×4/×8 局用） |
| **人类手动教学** | 网页互动台 | 你用方向键亲自走棋，每个合法的人类动作即时作为标签更新读出层；页面实时显示"学生原本想走哪、是否与你一致"和一致率统计 |

## 3. 仓库结构

| 文件 | 作用 |
|---|---|
| `01_check_token.py` | 验证 neuPrint 访问令牌（数据下载前置） |
| `02_fetch_neuron.py` | 从 MaleCNS（neuPrint `male-cns:v1.0`）抓取中枢神经元 DNge104_R 及其上下游 |
| `03_plot_skeleton.py` | 神经元骨架 3D 可视化（navis） |
| `04_build_subgraph.py` | 按规模参数构建连接组子图，导出 nodes/edges CSV |
| `05_brian2_lif_sim.py` | Brian2 LIF 脉冲网络的构建/归一化/仿真工具，被核心模块直接复用 |
| `fly2048_core.py` | 2048 环境 + 脑回路封装 + A2C/教学逻辑，提供全部命令行功能 |
| `fly2048_lab.py` | 网页互动台 + 训练控制台的 HTTP 服务器 |
| `fly2048_lab.html` | 互动台页面：棋盘、感觉放电热图、神经元放电栅格、方向概率、手动教学 |
| `train_console.html` | 训练控制台页面：启动/停止训练、学习曲线、权重 profile 管理 |
| `start_lab.bat` | Windows 一键启动互动台 |
| `outputs/subgraphs/` | **运行必需**：默认子图 `DNge104_R_12781_{nodes,edges}.csv`（已随仓库提供） |

## 4. 安装

需要 Windows / Linux / macOS + Python 3.12（无需 GPU、无需 Docker）。

推荐使用 [uv](https://docs.astral.sh/uv/)：

```powershell
uv sync
```

或使用标准虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## 5. 快速开始

### 5.1 网页互动台（推荐入口）

```powershell
# 方式一：一键脚本（后台启动服务并打开浏览器）
.\start_lab.bat

# 方式二：手动启动
.\.venv\Scripts\python.exe fly2048_lab.py
```

- 互动台：<http://127.0.0.1:8766/>
  实时观察棋盘、感觉输入放电、1452 个神经元的放电栅格和 4 方向概率；
  可让 AI 单步/连续走棋，也可进入**手动教学**模式用方向键走棋训练它。
- 训练控制台：<http://127.0.0.1:8766/console>
  在页面上新建权重 profile、启动/停止训练、查看学习曲线。

### 5.2 命令行

```powershell
# 强化学习自练 300 局（结束后自动评估、与随机基线对比）
.\.venv\Scripts\python.exe fly2048_core.py train --episodes 300

# expectimax 老师示范 10 局，D4 对称增强 ×4
.\.venv\Scripts\python.exe fly2048_core.py teach --episodes 10 --aug-k 4

# 自动循环：示范教学 ↔ 自主练习 交替进行
.\.venv\Scripts\python.exe fly2048_core.py auto --cycles 5

# 用训练好的权重贪心试玩一局
.\.venv\Scripts\python.exe fly2048_core.py play --weights outputs/fly2048/readout.npz

# 随机策略基线
.\.venv\Scripts\python.exe fly2048_core.py random --episodes 300
```

续训：`train --resume`；定期快照：`--save-every 5 --checkpoint-dir <dir>`。
权重、训练日志（`training_log.jsonl`）、教学日志（`human_teach_log.jsonl`）均写入
`--out-dir`（默认 `outputs/fly2048/`）。

### 5.3 重新生成连接组子图（可选）

仓库已附带默认可运行的子图 CSV。如需自己抓取/调整规模：

```powershell
# 1) 设置 neuPrint 令牌（https://neuprint.janelia.org 注册获取）
setx NEUPRINT_APPLICATION_CREDENTIALS "your-token"
# 2) 验证 → 抓取 → 出图 → 构建子图（--nodes/--edges 可被 train 等命令引用）
.\.venv\Scripts\python.exe 01_check_token.py
.\.venv\Scripts\python.exe 02_fetch_neuron.py
.\.venv\Scripts\python.exe 03_plot_skeleton.py
.\.venv\Scripts\python.exe 04_build_subgraph.py
```

## 6. 数据来源与致谢

- **连接组数据**：FlyEM **Male CNS connectome (male-cns:v1.0)**，
  Janelia Research Campus，协议 **CC-BY**。
  项目主页：<https://www.janelia.org/project-team/flyem/male-cns-connectome>
  本仓库 `outputs/subgraphs/` 中的 CSV 是该数据集的衍生作品，使用请同样遵守署名要求
  并引用 MaleCNS 论文（见上述项目页）。中枢神经元 `DNge104_R`（body ID 12781）
  及其上下游神经元均取自该数据集。
- **脉冲神经网络仿真**：[Brian2](https://briansimulator.org/)（CeCILL 2.1），
  Stimberg, Brette & Goodman。
- **神经形态分析/可视化**：[navis](https://github.com/navis-org/navis)（GPL-3.0-or-later）。
- **连接组查询客户端**：[neuprint-python](https://github.com/connectome-neuprint/neuprint-python)（BSD-3-Clause）。

上述库均通过包管理器安装、不在本仓库内分发；本项目代码不依赖 FlyGym / MuJoCo。

## 7. 免责声明

本项目是**连接组 + 脉冲网络 → 决策**的教学演示：

- 感觉编码（棋盘格值 → 泊松放电率）是人为设计的接口，不是果蝇真实视觉通路；
- 读出层是工程化的 MLP，果蝇脑中并不存在这样的"2048 决策区"；
- 结果用于展示真实神经结构上的信息处理与可塑性，不构成对果蝇认知能力的生物学结论。

## 8. 许可证

本项目自有代码采用 [MIT License](LICENSE)。
分发的连接组衍生数据（`outputs/subgraphs/`）按 **CC-BY** 单独授权，版权归 FlyEM / Janelia。
