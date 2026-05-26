# move_sampler 设计文档：自对弈开局温度采样

## 背景

[engine.py](file:///d:/projects/gomoku/ai/engine.py) 当前仅支持 argmax 落子策略，`strategy="sampling"` 分支预留
为 `NotImplementedError`（见 [engine_move_strategy_design.md](file:///d:/projects/gomoku/design/engine_move_strategy_design.md#L203-L209)）。贪婪
策略让自对弈轨迹高度同质化——同一模型在同一开局上永远走同一手，生成的训练样本
在开局空间的覆盖极窄，模型难以从自己的错误估值中学到新东西。

本设计在 **winner_path 软标签化**（[sample_creator_design.md §第三十九次追加](file:///d:/projects/gomoku/design/sample_creator_design.md#L367-L489)）落地后，
为自对弈数据生成链路引入"**开局阶段温度采样 + 之后 argmax**"的阶梯式探索策略，
让模型在开局阶段接触更多样的局面，从而在后续训练中下出更好的棋。

## 设计目标

1. 新增 [ai/move_sampler.py](file:///d:/projects/gomoku/ai/move_sampler.py) 模块，提供基于温度的加权采样
   函数，以及描述采样策略的 `TempConfig` 数据结构。
2. 扩展 [AIEngine.get_move](file:///d:/projects/gomoku/ai/engine.py#L66) 的 `strategy="sampling"` 分支，使
   其真正调用 `move_sampler` 完成落子；不再抛 `NotImplementedError`。
3. 在 [AIGameManager.start_ai_vs_ai / _play_single_game](file:///d:/projects/gomoku/ai/game_manager.py) 中完成"按
   步切换温度"与"按方启用"的编排，**不**在 engine 内部保留状态。
4. 在 [create_samples.py](file:///d:/projects/gomoku/ai/create_samples.py) 与 [train_main.py](file:///d:/projects/gomoku/ai/train_main.py)（cycle 模式）
   暴露 CLI 参数，**默认开启**温度采样；提供 `--no-temp-sampling` 关闭开关。
5. **GUI AI 内战**（[gui/ai_battle.py](file:///d:/projects/gomoku/gui/ai_battle.py)）不启用：通过
   `start_ai_vs_ai` 的 `temp_config=None` 默认值自然保持现状，无需改动。
6. 提供对照实验脚本，量化"温度采样 vs 全程 argmax"的开局多样性与胜方分布差异。

## 核心算法

### 温度采样公式

对合法位置概率向量 `p = [p_0, ..., p_{k-1}]`，给定温度 `T > 0`：

```
q_i = p_i^(1/T) / Σ_j p_j^(1/T)
```

然后从 `q` 中按权重加权随机抽一个下标。

- `T → 0`：最大位概率 → 1，等价 argmax。
- `T = 1`：`q == p`（原分布不变）。
- `T → ∞`：趋于均匀分布。

### 阶梯切换（方案 A）

给定 `temp_moves = N`（全局步数阈值）：

```
move_count = 该局已落子总数（黑白合计，从 0 起计）
if move_count < N and 本方已被 temp_side 选中：
    使用 temperature 采样
else:
    使用 argmax
```

### temp_side 四种取值语义

| 值 | 黑方 | 白方 | 备注 |
|---|---|---|---|
| `black` | 启用 | 关闭 | |
| `white` | 关闭 | 启用 | |
| `both` | 启用 | 启用 | 默认 |
| `random` | 每局开始前等概率抽一方启用，另一方关闭 | | 抽签在 `_play_single_game` 入口完成 |

## API 设计

### 1. 新模块 `ai/move_sampler.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random as _stdrandom

# ---- 常量 ----

TEMP_SIDE_BLACK = "black"
TEMP_SIDE_WHITE = "white"
TEMP_SIDE_BOTH = "both"
TEMP_SIDE_RANDOM = "random"
SUPPORTED_TEMP_SIDES = (TEMP_SIDE_BLACK, TEMP_SIDE_WHITE,
                       TEMP_SIDE_BOTH, TEMP_SIDE_RANDOM)

# ---- 数据类 ----

@dataclass
class TempConfig:
    """温度采样配置。

    Fields:
        temperature: 温度值，> 0。越大越随机；趋近 0 等价 argmax。
        temp_moves: 前 N 个全局步（move_count < N）启用温度采样，其后 argmax。
                    N = 0 表示"任何步都不用温度"（等同于关闭）。
        temp_side: 启用采样的一方，取值见 SUPPORTED_TEMP_SIDES。
        top_k: 可选截断，只在概率前 K 个中采样；None 表示不截断。
        seed: 可选随机种子；None 表示使用全局随机源（不可复现）。

    Raises (构造时):
        ValueError: temperature <= 0；temp_moves < 0；temp_side 不在白名单；
                    top_k 不为 None 且 <= 0。
    """
    temperature: float = 1.0
    temp_moves: int = 12
    temp_side: str = TEMP_SIDE_BOTH
    top_k: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None: ...

# ---- 采样函数 ----

def temperature_sample(
    valid_positions: List[Tuple[int, int]],
    valid_probs: List[float],
    temperature: float,
    top_k: Optional[int] = None,
    rng: Optional[_stdrandom.Random] = None,
) -> Tuple[int, int]:
    """对合法位置按温度加权采样并返回落点。

    Args:
        valid_positions: 合法（空位）坐标列表，长度 >= 1。
        valid_probs: 与 valid_positions 一一对应的模型概率。
        temperature: 温度值，> 0。
        top_k: 截断阈值，None 或 >= len(valid_positions) 表示不截断。
        rng: 随机源；None 时使用 random 模块全局源。

    Returns:
        (x, y) 选中的合法位置。

    Raises:
        ValueError: temperature <= 0；valid_positions 为空；长度不一致。
    """

# ---- 按步/按方决策 ----

def should_sample(
    move_count: int,
    current_player: int,
    temp_config: TempConfig,
    temp_side_this_game: str,
) -> bool:
    """判定当前步是否应该用温度采样。

    Args:
        move_count: 该局已落子总数（黑白合计，从 0 起计，当前步尚未落子）。
        current_player: 当前落子方，1=黑 / 2=白。
        temp_config: 温度配置。
        temp_side_this_game: 本局已解析的 temp_side（`random` 在对局开始前已解析为
                             `black` 或 `white`，传入此函数时应只取
                             black/white/both，永远不会是 `random`）。

    Returns:
        True → 该步用温度采样；False → 该步用 argmax。
    """

def resolve_temp_side(
    temp_side: str,
    rng: Optional[_stdrandom.Random] = None,
) -> str:
    """把 temp_side 解析为当局生效值。

    - `random` → 等概率返回 `black` 或 `white`；
    - 其他值 → 原样返回。

    Args:
        temp_side: TempConfig.temp_side 字段值。
        rng: 随机源，用于 `random` 的抽签。

    Raises:
        ValueError: temp_side 不在 SUPPORTED_TEMP_SIDES。
    """
```

### 2. 修改 `ai/engine.py`

```python
def get_move(
    self,
    board: List[List[int]],
    current_player: int,
    probs: List[float] = None,
    strategy: str = STRATEGY_ARGMAX,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    """...
    新增参数仅在 strategy="sampling" 时生效；argmax 分支行为完全不变。
    strategy="sampling" 不再抛 NotImplementedError，改为委托给
    ai.move_sampler.temperature_sample。
    """
```

行为约束：
- `strategy="argmax"` 分支签名、异常、返回值**零变更**。
- `strategy="sampling"` 且 `temperature <= 0` → 委托层统一 `ValueError`。
- 未知 `strategy` 仍抛 `ValueError`。

### 3. 修改 `ai/game_manager.py`

```python
@staticmethod
def _play_single_game(
    engine_black: AIEngine,
    engine_white: AIEngine,
    recorder: DataRecorder,
    game_num: int,
    name_prefix: str = "ai_vs_ai",
    temp_config: Optional["TempConfig"] = None,   # 新增
) -> int: ...

@staticmethod
def start_ai_vs_ai(
    engine1: AIEngine,
    engine2: AIEngine,
    num_games: int,
    recorder: DataRecorder,
    project_root: str = None,
    on_progress=None,
    temp_config: Optional["TempConfig"] = None,   # 新增
) -> List[dict]: ...

@staticmethod
def start_ai_vs_ai_random(
    engine1: AIEngine,
    engine2: AIEngine,
    num_games: int,
    recorder: DataRecorder,
    temp_config: Optional["TempConfig"] = None,   # 新增
) -> List[dict]: ...
```

- `temp_config=None`（默认）→ 保持现状（纯 argmax），所有既有调用方（GUI AI 内战、
  测试桩等）零改动。
- `temp_config` 非 None → 按步按方编排温度采样。
- `temp_side="random"` 在每局进入 `_play_single_game` 后立即通过
  `resolve_temp_side` 解析为 `black` / `white`，局内保持该值不变。

### 4. 修改 `ai/self_play_generator.py`

`run_self_play_and_evaluate` 新增透传参数：

```python
def run_self_play_and_evaluate(
    ...
    temp_config: Optional["TempConfig"] = None,
) -> dict: ...
```

传给 `AIGameManager.start_ai_vs_ai`；默认 `None` 时既有调用方零改动。

`SelfPlayGenerator.generate` 同步新增 `temp_config=None` 形参。

### 5. 修改 `ai/self_play_cycle.py`

`SelfPlayCycle.run` 新增：

```python
def run(
    self,
    ...
    temp_config: Optional["TempConfig"] = None,
) -> None: ...
```

透传到 `run_self_play_and_evaluate`。

### 6. 修改 `ai/create_samples.py` CLI

新增参数（作用于 `start_ai_vs_ai`）：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--no-temp-sampling` | flag | False | 禁用温度采样（等价 `temp_config=None`）|
| `--temperature` | float | 1.0 | 温度值 T，> 0 |
| `--temp-moves` | int | 12 | 前 N 步启用，其后 argmax |
| `--temp-side` | choice | both | `black` / `white` / `random` / `both` |
| `--temp-seed` | int | None | 随机种子（用于复现）|

参数互斥：`--no-temp-sampling` 与其他四个温度参数**不互斥**（但若用户显式传入其他参数，
以 `--no-temp-sampling` 为准，打印 warning 提示参数被忽略）。

### 7. 修改 `ai/train_main.py`（仅 cycle 模式）

与 `create_samples.py` 完全相同的 5 个 CLI 参数。参数解析后构造 `TempConfig` 并透
传至 `SelfPlayCycle.run`；**其他训练模式忽略这些参数**（已在互斥校验里加提示）。

### 8. GUI AI 内战

[gui/ai_battle.py](file:///d:/projects/gomoku/gui/ai_battle.py) 及相关调用**不传** `temp_config`，自动保持纯
argmax 行为——**本次不修改 GUI 文件**。

## 主流程伪代码

### move_sampler.temperature_sample

```
函数 temperature_sample(valid_positions, valid_probs, temperature, top_k=None, rng=None):
    校验 temperature > 0；valid_positions 非空；长度一致
    rng = rng or random 模块全局源

    # 1. top_k 截断（可选）
    若 top_k 非 None 且 top_k < len(valid_probs):
        取 valid_probs 中最大的 top_k 个保留，其余设 0（位置保持原对应关系）

    # 2. 温度变换：p_i^(1/T)
    inv_T = 1.0 / temperature
    weights = [p ** inv_T for p in valid_probs]

    # 3. 归一化；若全 0（模型把概率压在已落子位置）→ 均匀回退
    total = sum(weights)
    若 total <= 0:
        weights = [1.0] * len(valid_positions)

    # 4. 加权抽样
    idx = rng.choices(range(len(valid_positions)), weights=weights, k=1)[0]
    返回 valid_positions[idx]
```

### move_sampler.should_sample

```
函数 should_sample(move_count, current_player, temp_config, temp_side_this_game):
    若 temp_config.temp_moves <= 0:   # 等同关闭
        返回 False
    若 move_count >= temp_config.temp_moves:
        返回 False
    side = temp_side_this_game  # 已被 resolve_temp_side 规范化
    若 side == "both":
        返回 True
    若 side == "black" 且 current_player == 1:
        返回 True
    若 side == "white" 且 current_player == 2:
        返回 True
    返回 False
```

### AIEngine.get_move（sampling 分支新逻辑）

```
... 前半部分（board_to_vector、probs、valid_positions 收集）不变 ...

若 strategy == STRATEGY_ARGMAX:
    返回 _pick_argmax(valid_positions, valid_probs)   # 不变

若 strategy == STRATEGY_SAMPLING:
    从 ai.move_sampler import temperature_sample
    返回 temperature_sample(valid_positions, valid_probs,
                            temperature=temperature, top_k=top_k, rng=rng)

否则：
    raise ValueError(未知 strategy)
```

### AIGameManager._play_single_game（温度编排新逻辑）

```
函数 _play_single_game(engine_black, engine_white, recorder, game_num,
                       name_prefix="ai_vs_ai", temp_config=None):
    ... controller / recorder 创建不变 ...

    # 本局 temp_side（random → 抽签 black/white；其他原样）
    若 temp_config 非 None:
        rng = Random(temp_config.seed + game_num) 若 seed 非 None，否则 Random()
        temp_side_this_game = resolve_temp_side(temp_config.temp_side, rng)
    否则:
        rng = None
        temp_side_this_game = None

    move_count = 0
    while not controller.is_over():
        state = controller.get_state()
        board, player = state["board"], state["current_player"]
        engine = engine_black if player == 1 else engine_white

        board_vector = board_to_vector(board, player)
        probs = engine.model.predict(board_vector)

        # 决定本步策略
        use_temp = (temp_config 非 None and
                    should_sample(move_count, player, temp_config,
                                  temp_side_this_game))

        若 use_temp:
            x, y = engine.get_move(board, player, probs=probs,
                                   strategy=STRATEGY_SAMPLING,
                                   temperature=temp_config.temperature,
                                   top_k=temp_config.top_k,
                                   rng=rng)
        否则:
            x, y = engine.get_move(board, player, probs=probs)  # argmax

        recorder.record_move(current_game_id, board_vector, probs, (x, y))
        controller.make_move(x, y)
        move_count += 1

    ... 保存结果部分不变 ...
```

### create_samples.py CLI 构造 TempConfig

```
解析 CLI 参数
若 args.no_temp_sampling:
    若 用户显式传了 --temperature / --temp-moves / --temp-side / --temp-seed:
        打印 warning：已禁用温度采样，以上参数被忽略
    temp_config = None
否则:
    temp_config = TempConfig(
        temperature=args.temperature,
        temp_moves=args.temp_moves,
        temp_side=args.temp_side,
        top_k=None,                # 本次 CLI 不暴露 top_k
        seed=args.temp_seed,
    )

results = AIGameManager.start_ai_vs_ai(engine, engine, games, recorder,
                                       on_progress=on_progress,
                                       temp_config=temp_config)
```

## 主流程测试用例

### move_sampler 单元测试（`tests/ai/test_move_sampler.py`）

| # | 用例 | 断言 |
|---|------|------|
| 1 | `temperature_sample_T_eq_1_follows_distribution` | T=1 时大量采样频率贴近 p（χ² 宽松阈值）|
| 2 | `temperature_sample_small_T_behaves_like_argmax` | T=0.01 时连续采样 200 次全命中 argmax 位 |
| 3 | `temperature_sample_large_T_is_near_uniform` | T=100 时最高/最低位频率比接近 1（< 2）|
| 4 | `temperature_sample_invalid_temperature_raises` | T=0 / 负数 → `ValueError` |
| 5 | `temperature_sample_empty_positions_raises` | 空位置列表 → `ValueError` |
| 6 | `temperature_sample_top_k_respected` | top_k=2 且前 2 外有 p>0 → 外部位置永不被选中 |
| 7 | `temperature_sample_all_zero_probs_uniform_fallback` | 全零概率时均匀选择 |
| 8 | `temperature_sample_seed_reproducible` | 同 seed 的两个 Random → 连续采样序列完全一致 |
| 9 | `TempConfig_validates_fields` | 非法 T/temp_moves/temp_side/top_k → 构造时抛 `ValueError` |
| 10 | `should_sample_before_threshold_both` | move_count=0..N-1, side=both → 均为 True |
| 11 | `should_sample_at_or_after_threshold` | move_count=N → False；move_count=N+5 → False |
| 12 | `should_sample_side_filter` | side=black 时黑方 True / 白方 False；white 对称 |
| 13 | `should_sample_temp_moves_zero_equals_off` | temp_moves=0 → 任何情况均 False |
| 14 | `resolve_temp_side_random_splits_evenly` | 固定 seed 抽样 200 局，black/white 频率均衡（误差 < 10%）|
| 15 | `resolve_temp_side_passthrough` | black/white/both 原样返回 |
| 16 | `resolve_temp_side_invalid_raises` | 未知值 → `ValueError` |

### engine 扩展测试（追加到 `tests/ai/test_engine.py`）

| # | 用例 | 断言 |
|---|------|------|
| 17 | `sampling_strategy_no_longer_raises_not_implemented` | 传 strategy="sampling" 不再抛 `NotImplementedError` |
| 18 | `sampling_strategy_returns_valid_position` | 返回位置 (x,y) ∈ 合法空位集合 |
| 19 | `sampling_strategy_small_T_equals_argmax` | strategy="sampling", T=0.01 → 与 argmax 同位置 |
| 20 | `sampling_strategy_respects_rng_seed` | 传入同 seed 的 rng → 两次调用同位置 |
| 21 | `sampling_strategy_invalid_temperature_raises` | T=0 → `ValueError` |
| 22 | `argmax_path_unchanged_by_new_params` | strategy 缺省调用，新参数默认值 → 行为与旧代码完全一致 |

### game_manager 编排测试（追加到 `tests/ai/test_game_manager.py`）

| # | 用例 | 断言 |
|---|------|------|
| 23 | `default_no_temp_config_uses_argmax_only` | temp_config=None → engine.get_move 调用始终 strategy=argmax（Mock 统计）|
| 24 | `temp_config_applies_only_before_threshold` | temp_moves=5 → 前 5 步 strategy=sampling，第 6 步起 strategy=argmax |
| 25 | `temp_side_black_only_affects_black` | side=black → 仅黑方调用用 sampling；白方全程 argmax |
| 26 | `temp_side_random_resolved_once_per_game` | side=random, seed 固定 → 同一局内 side 不变；不同局可不同 |
| 27 | `gui_path_unaffected` | 模拟 GUI 调用（不传 temp_config）→ 行为与改动前完全一致 |
| 28 | `start_ai_vs_ai_random_supports_temp_config` | `start_ai_vs_ai_random` 同样接受并生效 temp_config |

### CLI 端到端测试

| # | 文件 | 用例 | 断言 |
|---|---|---|---|
| 29 | `tests/ai/test_create_samples.py` | `test_cli_default_enables_temp_sampling` | 不传 `--no-temp-sampling` 时，CLI 内部构造的 TempConfig 非 None（可通过 mock 注入探针验证）|
| 30 | `tests/ai/test_create_samples.py` | `test_cli_no_temp_sampling_disables` | 传 `--no-temp-sampling` → TempConfig 为 None |
| 31 | `tests/ai/test_create_samples.py` | `test_cli_temp_side_parse` | `--temp-side random` → TempConfig.temp_side == "random" |
| 32 | `tests/ai/test_create_samples.py` | `test_cli_invalid_temperature_fails` | `--temperature 0` → argparse/TempConfig 报错，进程非 0 退出 |
| 33 | `tests/ai/test_train_main.py` | `test_cycle_mode_accepts_temp_flags` | cycle 模式解析新参数并透传到 SelfPlayCycle.run |
| 34 | `tests/ai/test_train_main.py` | `test_non_cycle_mode_ignores_temp_flags` | cycle 模式外传入温度参数时打印 warning 并不影响训练 |

### 回归保障

- 既有所有 `tests/ai/` 单测（engine、game_manager、create_samples、self_play_*、
  train_main、GUI AI 内战相关）应在本次改动后**全部通过**，无 xfail / skip 新增。
- 用户偏好：所有失败测试必须修复或明确处置，不得以"与改动无关"为由放过。

## 对照实验（手工验证流程）

### 目标

手工验证"温度采样开启 vs 关闭"确实对自对弈数据多样性产生了期望的影响。**本项目不提供脚本**，由开发者按以下流程手工操作，根据观察结果判断 `temperature` / `temp_moves` 是否需要微调。

### 命令行示例

以 `cnn_v3.1` 模型、50 局自对弈为例：

```bash
# A. argmax baseline（关闭温度采样）
python -m ai.create_samples --model cnn_v3.1 --games 50 \
    --name temp_compare_argmax --no-temp-sampling

# B. 温度采样（默认 T=1.0, temp_moves=12, both）
python -m ai.create_samples --model cnn_v3.1 --games 50 \
    --name temp_compare_sampling --temp-seed 42
```

两批数据分别落在 `ai/ai_data/temp_compare_argmax/` 与 `ai/ai_data/temp_compare_sampling/`。

### 预期观察点

| 维度 | 观察方式 | 预期现象 |
|---|---|---|
| 开局多样性 | 用 `ai/sample_visualizer.py` 查看前 1/5/10 步的落子点 | 温度采样下**唯一开局序列数**明显多于 argmax（argmax 倾向于少数几条固定线） |
| 胜负分布 | 统计 `results` 数组中黑 / 白 / 平局占比 | 温度采样下分布更均衡；argmax 可能长期一边倒 |
| 平均步数 | 对每局 moves 取平均 | 温度采样下方差更大；argmax 步数集中 |
| 棋盘位置熵 | 对前 12 步每格落子频率计算 Shannon 熵 | 温度采样显著更高 |

### 调参判定

- 若 **A 与 B 的开局多样性差异过小**（<2×）→ 考虑增大 `--temperature`（例如 1.2）或 `--temp-moves`（例如 18）。
- 若 **B 的胜率波动过大、训练 loss 不收敛** → 考虑减小 `--temperature`（例如 0.8）或仅对黑方采样（`--temp-side black`）。
- 若 **两批几乎无差异** → 检查模型输出 softmax 概率是否已接近 one-hot（此时温度采样退化为 argmax，符合预期，无需调参）。

### 非目标

本次**不提供自动化对比脚本**（如 `temp/compare_temperature_vs_argmax.py`）。如后续确有量化需求，再补充 ad-hoc 脚本。

## 涉及文件

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `design/move_sampler_design.md` | 新增 | 本设计文档 |
| `ai/move_sampler.py` | 新增 | `TempConfig` 数据类、`temperature_sample` / `should_sample` / `resolve_temp_side` 函数 |
| `ai/engine.py` | 修改 | `get_move` 新增 `temperature` / `top_k` / `rng` 形参；`sampling` 分支委托给 `move_sampler.temperature_sample` |
| `ai/game_manager.py` | 修改 | `_play_single_game` / `start_ai_vs_ai` / `start_ai_vs_ai_random` 新增 `temp_config` 形参；按步按方编排 |
| `ai/self_play_generator.py` | 修改 | `run_self_play_and_evaluate` 与 `SelfPlayGenerator.generate` 新增 `temp_config` 透传 |
| `ai/self_play_cycle.py` | 修改 | `SelfPlayCycle.run` 新增 `temp_config` 透传 |
| `ai/create_samples.py` | 修改 | 新增 5 个 CLI 参数并构造 `TempConfig` |
| `ai/train_main.py` | 修改 | cycle 模式新增 5 个 CLI 参数并透传 |
| `tests/ai/test_move_sampler.py` | 新增 | 用例 1-16 |
| `tests/ai/test_engine.py` | 修改 | 用例 17-22；原用例 `test_sampling_raises_not_implemented` 须调整为新语义 |
| `tests/ai/test_game_manager.py` | 修改 | 用例 23-28 |
| `tests/ai/test_create_samples.py` | 修改 | 用例 29-32 |
| `tests/ai/test_train_main.py` | 修改 | 用例 33-34 |
| `temp/compare_temperature_vs_argmax.py` | ~~新增~~ 取消 | 改为在本文档「对照实验」节记录手工验证流程，不写脚本 |
| `README.md` | 修改 | `create_samples.py` 与 `train_main.py` cycle 模式用法补充温度采样参数示例 |

## 非目标（本次不做）

- top-k 截断仅在 `move_sampler` 层实现并测试，**CLI 不暴露** `--top-k`。
- Dirichlet 噪声。
- 温度的连续衰减（方案 B）。
- 为 PvE / 评估 / battle.py 开启温度采样。

---

# 第四十一次追加：GUI AI 内战启用温度采样 + Replay 展示

> 对应需求：`requirements/key_points.md` 第四十一次追加。
> 详细设计：`design/gui_temp_sampling_replay_design.md`。

原设计目标中"GUI AI 内战不启用"在本次追加中变更为：

- **GUI AI 内战默认启用温度采样**，使用与 CLI 一致的默认参数（`temperature=1.0, temp_moves=12, temp_side="both"`），通过 `BattleSession` 透传 `temp_config`。
- **非目标**中"为 PvE / 评估 / battle.py 开启温度采样"仍然成立——PvE 人机对战保持 argmax 不变。
- 新增 `DataRecorder.record_move()` 扩展字段（`strategy`/`temperature`/`top_k`/`sampling_probs`），用于记录每步的采样信息。
- 新增 `ReplayViewer` Sampling 热力图 + 策略文本展示，在回放界面可视化温度采样细节。
