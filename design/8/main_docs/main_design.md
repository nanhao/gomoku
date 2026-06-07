# 主设计文档：battle.py 支持温度采样参数

| 字段 | 值 |
|------|-----|
| intention_id | 8 |
| PRD | requirements/8/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

本次变更仅在 `ai/battle.py` 新增 5 个 CLI 温度采样参数，并打通从 CLI → `TempConfig` 构造 → `AIGameManager.start_ai_vs_ai_random()` 的传递链路。后端（`TempConfig`、`AIGameManager`）已就绪，无需修改。

**修改范围**：1 个已有文件，无新增文件。

交互图见 [battle_temp_flow.puml](battle_temp_flow.puml)。

---

## 2. 模块变更

### 2.1 变更清单

| 文件 | 变更类别 | 说明 |
|------|---------|------|
| `ai/battle.py` | CLI 扩展 + 接口扩展 | 新增 5 个温度采样 CLI 参数；`run_battle()` 签名新增 `temp_config` 参数 |

### 2.2 run_battle() 签名变更

**当前签名**：

```
run_battle(ai1_name: str, ai2_name: str, num_games: int)
```

**变更后签名**：

```
run_battle(ai1_name: str, ai2_name: str, num_games: int,
           temp_config: Optional[TempConfig] = None)
```

| 新参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `temp_config` | `Optional[TempConfig]` | `None` | 温度采样配置；`None` 时透传给 `AIGameManager`，底层解释为全程 argmax；默认启用温度采样由 `main()` 保证 |

**行为**：
- 若 `temp_config` 不为 `None`：直接透传给 `AIGameManager.start_ai_vs_ai_random()`
- 若 `temp_config` 为 `None`：透传 `None` 给 `AIGameManager`，底层解释为全程 argmax

> **注意**：`run_battle()` 自身不做 `None → 默认 TempConfig` 的语义转换。
> 默认启用温度采样的职责在 `main()` 中完成——CLI 不传任何温度参数时，
> `main()` 自动构造默认 `TempConfig`（T=1.0, moves=12, both）传入。
> 若调用方需要全程 argmax，可直接传 `temp_config=None` 或传 `--no-temp-sampling`。

### 2.3 main() CLI 变更

新增 5 个 argparse 参数，参数定义完全对齐 `ai/create_samples.py` 的现有模式：

| CLI 参数 | argparse 类型 | 默认值 | choices | 说明 |
|----------|--------------|--------|---------|------|
| `--no-temp-sampling` | `store_true` | `False` | — | 禁用温度采样，全程 argmax |
| `--temperature` | `float` | `1.0` | — | 温度值 T > 0 |
| `--temp-moves` | `int` | `12` | — | 前 N 步启用温度采样，≥ 0 |
| `--temp-side` | `str` | `TEMP_SIDE_BOTH` | `list(SUPPORTED_TEMP_SIDES)` | 启用采样的一方（默认 `both`） |
| `--temp-seed` | `int` | `None` | — | 随机种子（可选） |

**TempConfig 构造逻辑**（在 `main()` 中，argparse 解析之后，**在 `try` 块之内**）：

```
import warnings  # 新增导入

# ...（argparse 解析）...

try:
    if args.no_temp_sampling:
        # 提示用户：其他温度参数被忽略（与 create_samples.py 行为一致）
        if (args.temperature != 1.0 or args.temp_moves != 12
                or args.temp_side != "both" or args.temp_seed is not None):
            warnings.warn(
                "已启用 --no-temp-sampling，以下参数被忽略："
                "--temperature / --temp-moves / --temp-side / --temp-seed"
            )
        temp_config = None
    else:
        temp_config = TempConfig(
            temperature=args.temperature,
            temp_moves=args.temp_moves,
            temp_side=args.temp_side,
            top_k=None,
            seed=args.temp_seed,
        )

    run_battle(args.ai1, args.ai2, args.games, temp_config=temp_config)
except ValueError as e:
    print(f"错误: {e}")
    sys.exit(1)
```

### 2.4 新增导入

`ai/battle.py` 新增导入：

```python
import math      # 标准库，用于 finite_float 校验 inf/nan
import warnings  # 标准库，用于 --no-temp-sampling 时提示被忽略的参数

from ai.move_sampler import TempConfig, SUPPORTED_TEMP_SIDES, TEMP_SIDE_BOTH
```

---

## 3. 数据流

```
ai/battle.py CLI
    │
    │  argparse 解析 5 个温度参数（全部可选，有默认值）
    │
    ▼
main()
    │
    │  if args.no_temp_sampling:
    │      temp_config = None
    │  else:
    │      temp_config = TempConfig(temperature=..., temp_moves=...,
    │                               temp_side=..., top_k=None, seed=...)
    │
    ▼
run_battle(ai1, ai2, games, temp_config)
    │
    │  → AIGameManager.start_ai_vs_ai_random(engine1, engine2,
    │        num_games, recorder, temp_config=temp_config)
    │
    ▼
AIGameManager.start_ai_vs_ai_random()
    │
    │  → play_single_game(..., temp_config=temp_config)
    │      ├─ 若 temp_config ≠ None → 按配置启用温度采样
    │      └─ 若 temp_config = None → 全程 argmax
    │
    ▼
对战结果输出（格式不变）
```

> **关于 `temp-seed` 的可复现性**：`AIGameManager.play_single_game()` 内部使用
> `seed + game_num` 为每局派生独立的随机种子（参见
> `ai/game_manager.py`）。这意味着：
> - 同一 `temp-seed` 的多局对战：每局有独立但确定性的随机序列
> - 同一 `temp-seed` + 同一局号：落子序列完全相同
> - 不同 `temp-seed` 或不同局号：随机序列不同
> - 这是 `AC-5.6`（同一 seed 运行两次结果完全一致）得以成立的关键机制。

---

## 4. 参数校验

| 校验点 | 策略 | 触发方 |
|--------|------|--------|
| `--temperature` 为负数 | argparse `type=float` 不拦截负数 → 依赖 `TempConfig.__post_init__` 的 `temperature > 0` 校验，抛 `ValueError` | TempConfig 构造 |
| `--temperature` 为 `inf`/`nan` | `type=float` 接受特殊浮点值 → **在 argparse 层使用自定义 `type` 函数拒绝**（因 PRD §3.2 排除修改 `TempConfig`，校验必须在 `battle.py` 内部完成） | argparse（自定义 `type`） |
| `--temp-moves` 为负数 | 同上，依赖 `TempConfig.__post_init__` 的 `temp_moves >= 0` 校验 | TempConfig 构造 |
| `--temp-side` 非法值 | argparse `choices` 参数自动拦截，报错退出（退出码 2） | argparse |
| `--temp-seed` 非整数 | argparse `type=int` 自动拦截 | argparse |
| `--no-temp-sampling` 与其他温度参数共存 | 不报错，`--no-temp-sampling` 优先，`temp_config = None`；通过 `warnings.warn` 提示用户被忽略的参数（与 `create_samples.py` 一致） | main() |

> **设计决策**：`temperature` 和 `temp_moves` 的合法性校验委托给 `TempConfig.__post_init__`，而非在 argparse 中重复校验。理由：① 避免重复校验逻辑；② 与 `create_samples.py` 保持一致；③ 即使通过 Python API 直接调用 `run_battle()` 传入非法值也能被拦截。
>
> **inf/nan 决策**：`argparse type=float` 会接受 `inf`/`-inf`/`nan` 等特殊浮点值，而 `TempConfig.__post_init__` 的 `temperature > 0` 检查无法拦截。
> PRD §3.2 明确排除修改 `TempConfig`，因此校验在 `battle.py` 的 argparse 层完成：
> 使用自定义 `type` 函数——若 `float(x)` 成功但 `not math.isfinite(x)`，则抛出 `argparse.ArgumentTypeError`；
> 否则返回 `float(x)`。实现示例：
>
> ```python
> import math
> def finite_float(value):
>     f = float(value)
>     if not math.isfinite(f):
>         raise argparse.ArgumentTypeError(f"temperature 必须是有限浮点数，收到: {value}")
>     return f
> ```

---

## 5. 向后兼容

| 场景 | 兼容性 |
|------|--------|
| 不传任何温度参数的现有调用 | 行为变化：由全程 argmax 变为默认启用温度采样（T=1.0, moves=12, both） |
| 传 `--no-temp-sampling` 的调用 | 行为与当前完全一致（全程 argmax） |
| `--ai1` / `--ai2` / `--games` 参数 | 行为不变 |
| 输出格式 | 不变（每局结果 + 统计摘要） |
| intention_id=6 的 `battle.py` 调用 | 兼容——不传温度参数时按默认温度采样执行，对战仍正常完成并输出胜率 |

---

## 6. 错误处理

| 场景 | 处理方式 |
|------|---------|
| `--temperature 0` 或负数 | `TempConfig.__post_init__` 抛 `ValueError`，`main()` 中 `except ValueError` 捕获，打印错误信息并以退出码 1 退出 |
| `--temp-moves -5` | 同上，`TempConfig.__post_init__` 抛 `ValueError` |
| `--temp-side abc` | argparse `choices` 校验失败，argparse 自动报错退出（退出码 2） |
| `--temp-seed abc` | argparse `type=int` 自动报错退出（退出码 2） |
| 模型加载失败 | 现有 `except ValueError` 逻辑不变，退出码 1 |

> **实现注意**：`TempConfig` 构造必须在 `try` 块之内（`try` 块需从 `TempConfig` 构造开始），
> 否则 `TempConfig.__post_init__` 抛出的 `ValueError` 不会被 `except ValueError` 捕获，
> 导致程序直接崩溃而非优雅退出。参见 §2.3 中 TempConfig 构造伪代码的位置——
> 已明确标注"在 `try` 块之内"。