# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/unit/test_battle_temp.py` | 单元测试 | `run_battle()` 签名变更、`main()` argparse 参数解析、TempConfig 构造逻辑 |

> 本次变更仅涉及 CLI 参数解析和 TempConfig 构造，不涉及新的业务逻辑，无需集成测试。温度采样的端到端行为已在 `tests/ai/test_game_manager_temp.py` 中覆盖。

---

## 2. 单元测试用例

### 2.1 run_battle() — temp_config 参数

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| UT-1.1 | `run_battle()` 不传 temp_config → 透传 None | `run_battle("random", "random", 1)` | 内部调用 `start_ai_vs_ai_random` 时传入的 `temp_config` 为 `None`（`run_battle()` 不做 `None → 默认 TempConfig` 转换，默认启用由 `main()` 保证） |
| UT-1.2 | 显式传入 temp_config | `cfg = TempConfig(temperature=2.0, temp_moves=6, temp_side="black"); run_battle("random", "random", 1, temp_config=cfg)` | 内部调用 `start_ai_vs_ai_random` 时传入的 `temp_config` 等于 `cfg` |
| UT-1.3 | `run_battle()` 显式传 temp_config=None → 透传 None | `run_battle("random", "random", 1, temp_config=None)` | 内部调用 `start_ai_vs_ai_random` 时传入的 `temp_config` 为 `None`（透传，不做语义转换） |
| UT-1.4 | 传入 temp_moves=0 的 TempConfig | `cfg = TempConfig(temperature=1.0, temp_moves=0, temp_side="both"); run_battle("random", "random", 1, temp_config=cfg)` | 内部调用 `start_ai_vs_ai_random` 时传入的 `temp_config` 为 `cfg`（temp_moves=0 等价于禁用温度采样） |

### 2.2 main() — argparse 参数解析

| 用例编号 | 用例描述 | 命令行参数模拟 | 预期结果 |
|----------|---------|--------------|---------|
| UT-2.1 | 不传任何温度参数 | `--ai1 random --ai2 random --games 1` | `args.no_temp_sampling` 为 `False`；`args.temperature` 为 `1.0`；`args.temp_moves` 为 `12`；`args.temp_side` 为 `"both"`；`args.temp_seed` 为 `None` |
| UT-2.2 | 传入 `--temperature 2.0` | `--ai1 random --ai2 random --games 1 --temperature 2.0` | `args.temperature` 为 `2.0` |
| UT-2.3 | 传入 `--temp-moves 6` | `--ai1 random --ai2 random --games 1 --temp-moves 6` | `args.temp_moves` 为 `6` |
| UT-2.4 | 传入 `--temp-side black` | `--ai1 random --ai2 random --games 1 --temp-side black` | `args.temp_side` 为 `"black"` |
| UT-2.5 | 传入 `--temp-side random` | `--ai1 random --ai2 random --games 1 --temp-side random` | `args.temp_side` 为 `"random"` |
| UT-2.6 | 传入 `--temp-seed 42` | `--ai1 random --ai2 random --games 1 --temp-seed 42` | `args.temp_seed` 为 `42` |
| UT-2.7 | 传入 `--no-temp-sampling` | `--ai1 random --ai2 random --games 1 --no-temp-sampling` | `args.no_temp_sampling` 为 `True` |
| UT-2.8 | 组合参数 | `--ai1 random --ai2 random --games 1 --temperature 0.5 --temp-moves 8 --temp-side white --temp-seed 99` | 所有参数正确解析 |
| UT-2.9 | `--temp-side abc` 报错 | `--ai1 random --ai2 random --games 1 --temp-side abc` | `SystemExit`，退出码 2，stderr 含 "invalid choice" |
| UT-2.10 | `--temp-seed abc` 报错 | `--ai1 random --ai2 random --games 1 --temp-seed abc` | `SystemExit`，退出码 2，stderr 含类型错误 |
| UT-2.11 | `--temperature inf` 报错 | `--ai1 random --ai2 random --games 1 --temperature inf` | `SystemExit`，退出码 2，stderr 含 "必须是有限浮点数" |
| UT-2.12 | `--temperature nan` 报错 | `--ai1 random --ai2 random --games 1 --temperature nan` | `SystemExit`，退出码 2，stderr 含 "必须是有限浮点数" |
| UT-2.13 | `--help` 输出含新参数 | `--help` | stdout 含 `--no-temp-sampling`、`--temperature`、`--temp-moves`、`--temp-side`、`--temp-seed` 的帮助文本 |

### 2.3 main() — TempConfig 构造逻辑

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| UT-3.1 | 默认构造 TempConfig | `args.no_temp_sampling=False`，其余参数为默认值 | 构造的 `TempConfig` 字段为 T=1.0, moves=12, side=both, seed=None |
| UT-3.2 | `--no-temp-sampling` 时不构造 TempConfig | `args.no_temp_sampling=True` | `temp_config` 为 `None`，不调用 `TempConfig()` 构造器 |
| UT-3.3 | `--temperature 0` 抛异常 | `args.temperature=0`（需手动将 `args.temperature` 设置为 0 模拟） | `TempConfig.__post_init__` 抛 `ValueError`，消息含 "temperature 必须 > 0" |
| UT-3.4 | `--temperature -1` 抛异常 | `args.temperature=-1` | `TempConfig.__post_init__` 抛 `ValueError`，消息含 "temperature 必须 > 0" |
| UT-3.5 | `--temp-moves -5` 抛异常 | `args.temp_moves=-5` | `TempConfig.__post_init__` 抛 `ValueError`，消息含 "temp_moves 必须 >= 0" |

### 2.4 main() — 错误处理

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| UT-4.1 | TempConfig 构造失败时退出（try 块覆盖校验） | 传入非法温度参数（如 `--temperature -1`） | 捕获 `ValueError`，打印错误信息，退出码 1。**验证点**：`main()` 中 `try` 块从 `TempConfig` 构造开始（而非仅包裹 `run_battle()`），确保构造阶段的 `ValueError` 也被捕获 |
| UT-4.2 | 模型加载失败仍正常处理 | 传入不存在的 AI 名称 | 现有 `except ValueError` 逻辑捕获，退出码 1（不受本次变更影响） |

---

## 3. 验收用例集（与 PRD §5 对齐）

### AC-1 参数解析验证（对应 PRD §5 AC-1）

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-1.1 "不传任何温度参数" | UT-2.1、UT-3.1 |
| AC-1.2 "传入 --temperature" | UT-2.2 |
| AC-1.3 "传入 --temp-moves" | UT-2.3 |
| AC-1.4 "传入 --temp-side black" | UT-2.4 |
| AC-1.5 "传入 --temp-side random" | UT-2.5 |
| AC-1.6 "传入 --temp-seed" | UT-2.6 |
| AC-1.7 "传入 --no-temp-sampling" | UT-2.7、UT-3.2 |
| AC-1.8 "组合多个温度参数" | UT-2.8 |
| AC-1.9 "temperature 0 报错" | UT-3.3 |
| AC-1.10 "temperature -1 报错" | UT-3.4 |
| AC-1.11 "temp-moves -5 报错" | UT-3.5 |
| AC-1.12 "temp-side invalid 报错" | UT-2.9 |
| AC-1.13 "--help 含新参数" | UT-2.13 |

### AC-2 默认行为验证（对应 PRD §5 AC-2）

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-2.1 "默认启用温度采样" | UT-3.1（`main()` 构造默认 TempConfig） |
| AC-2.2 "默认采样行为" | 端到端验证由 `tests/ai/test_game_manager_temp.py` 覆盖 |

### AC-3 关闭温度采样验证（对应 PRD §5 AC-3）

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-3.1 "--no-temp-sampling 关闭" | UT-2.7、UT-3.2 |
| AC-3.2 "--temp-moves 0 关闭" | UT-1.4 |
| AC-3.3 "关闭时温度参数被忽略" | UT-3.2（`--no-temp-sampling` 时 `temp_config=None`） |

### AC-4 向后兼容验证（对应 PRD §5 AC-4）

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-4.1 "现有调用方式不报错" | UT-2.1 |
| AC-4.2 "现有参数不受影响" | UT-2.1（`--ai1`/`--ai2`/`--games` 仍正确解析） |
| AC-4.3 "输出格式不变" | 端到端验证（函数签名不变，返回值格式不变） |
| AC-4.4 "intention_id=6 调用兼容" | UT-2.1（不传温度参数正常运行） |

### AC-5 温度采样端到端验证（对应 PRD §5 AC-5）

端到端验证由 `tests/ai/test_game_manager_temp.py` 覆盖，本次新增测试聚焦于参数传递正确性。

### AC-6 边缘场景验证（对应 PRD §5 AC-6）

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-6.1 "temperature 极小正数" | UT-3.1（`TempConfig` 接受极小正数，不抛异常） |
| AC-6.2 "temp-moves 极大值" | UT-2.3（argparse 不拦截，`TempConfig` 不拦截） |
| AC-6.3 "temperature 极大值" | UT-2.2（`TempConfig` 接受极大值，不抛异常） |
| AC-6.4 "同时传 --no-temp-sampling 和 --temp-moves 0" | UT-2.7 + UT-2.3 组合验证 |

---

## 4. Mock 策略

### 4.1 run_battle() 单元测试

Mock `AIGameManager.start_ai_vs_ai_random` 和 `get_model`，仅验证调用时传入的 `temp_config` 参数值是否正确。不执行实际对战。

### 4.2 CLI 参数测试

使用 `argparse.ArgumentParser.parse_args` 直接测试参数解析，不启动完整对战流程。对校验失败的测试通过 `pytest.raises(SystemExit)` 捕获退出。

### 4.3 TempConfig 构造测试

直接构造 `TempConfig` 实例验证字段值，或传入非法参数验证 `ValueError`。这部分已在 `tests/ai/test_move_sampler.py` 中覆盖，本次仅验证参数从 argparse 到 TempConfig 的映射正确性。

### 4.4 不 Mock 的模块

- `TempConfig`：真实构造，验证参数映射正确性
- `argparse`：真实解析，验证参数组合行为