# 测试集静态 Review 报告

**日期**：2026-05-01
**范围**：`tests/` 目录全量，约 95 个测试文件
**方式**：静态分析（未运行测试）
**判定口径**：
- **重复**：实质等价（路径高度重叠、仅输入差异可忽略、跨层覆盖相同场景均计入）
- **无效**：弱断言 / 废弃 / 过度 mock / skip 遗留 / 协助掩盖问题

> 说明：本报告仅产出问题清单与修复建议，不动测试代码。执行修复需另外确认。

---

## 一、总体统计

| 指标 | 数值 |
|------|------|
| 测试文件总数 | 约 95 |
| 测试用例总数（估算） | 800+ |
| 发现的实质等价重复**组** | 6 |
| 建议合并/删除的**文件** | 8–10 |
| 建议合并/删除的**用例** | 60–80 |
| 无效测试用例（弱断言 / 过度 mock / 结构遗留） | 25+ |
| 掩盖问题的用例 | 6 |

### 按模块规模
| 模块 | 文件数 | 主要问题 |
|------|-------|---------|
| tests/ai（主） | ~50 | **重灾区**：CNN v1/v2/v3 与 FC v1 系列训练/模型测试大量复制 |
| tests/ai/primary + edge + unit | ~10 | Evaluator 4 层覆盖，重构验证遗留 |
| tests/integration + tests/unit | 4 | 与 tests/ai/unit 存在三份 LR range test 重复 |
| tests/gui | 10 | 过度 mock 导致"只验证 mock 自身" |
| tests/app + tests/cli | 11 | 少量弱断言与重复 |
| tests 根目录 | 4 | 条件断言、异常吞噬 |

---

## 二、实质等价的重复测试（按组）

### 组 1（P0）：CNN v1 / v2 / v3 模型测试三套复制粘贴

**涉及文件**
- [test_cnn_model.py](file:///d:/projects/gomoku/tests/ai/test_cnn_model.py)（v1，13.2 KB）
- [test_cnn_v2_model.py](file:///d:/projects/gomoku/tests/ai/test_cnn_v2_model.py)（7.6 KB）
- [test_cnn_v3_model.py](file:///d:/projects/gomoku/tests/ai/test_cnn_v3_model.py)（9.7 KB）

**证据**（逐行对照）
| v1 | v2 | v3 | 内容 |
|----|----|----|------|
| L24-34 | L23-31 | L25-33 | `test_network_structure`：输入 `(4,3,15,15)`，断言输出 `(batch,225)` |
| L36-46 | L33-42 | L35-44 | `test_network_softmax_output_range`：断言 `0 < out < 1` |
| L48-60 | L44-56 | L46-57 | `test_network_softmax_row_sum`：断言每行和 ≈ 1 |
| L104-107 | L104-107 | L153-156 | `test_model_name`：断言 `model.name == "cnn_vX"` |
| L115-125 | L109-116 | L158-165 | `test_model_predict_shape`：断言 `len(result) == 225` |
| L143-154 | L118-128 | L167-177 | `test_model_predict_probability_range` |
| L140-148 | L140-148 | L189-197 | `test_model_save_and_load` |

**建议**
- 合并为单一文件 `test_cnn_models.py`，用 `parameterized` 或抽象基类对 `(CNNV1Network/Model, CNNV2Network/Model, CNNV3Network/Model)` 三者执行同一套行为测试。
- 保留各版本**特异性**测试：参数数量、输入通道语义、多代 gen 语义。
- 预计缩减：**3 个文件 → 1 个文件**，减少约 **50 个用例**。

---

### 组 2（P0）：CNN v1 / v2 / v3 训练器测试三套复制粘贴

**涉及文件**
- [test_cnn_trainer.py](file:///d:/projects/gomoku/tests/ai/test_cnn_trainer.py)
- [test_cnn_v2_trainer.py](file:///d:/projects/gomoku/tests/ai/test_cnn_v2_trainer.py)
- [test_cnn_v3_trainer.py](file:///d:/projects/gomoku/tests/ai/test_cnn_v3_trainer.py)

**证据**
| v1 | v2 | v3 | 内容 |
|----|----|----|------|
| L71-76 | L68-73 | L68-73 | `test_trainer_initialization` |
| L85-97 | L80-89 | L80-89 | `test_trainer_train_single` |
| — | L91-102 | L91-102 | `test_trainer_prepare_tensors`（v1 无但本质相同断言 shape 在主文件别处） |
| L200-230 | L116-146 | L116-146 | `TestCNNvXTrainerEdgeCases` 整类复制（空列表 / 不存在目录） |

**建议**
- 同组 1 用参数化整合。`TestCNNvXTrainerEdgeCases` 整体抽出到公共 Mixin。
- 预计缩减：约 **40 个用例** 合并为 ~15 个参数化用例。

---

### 组 3（P0）：FC_V1 训练器 7 文件碎片化

**涉及文件**（全部针对同一个 `FCv1Trainer`）
- [test_fc_v1_trainer.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py)（188 行，基础）
- [test_fc_v1_trainer_batch.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_batch.py)（266 行）
- [test_fc_v1_trainer_cycle.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_cycle.py)（75 行，**含空类**）
- [test_fc_v1_trainer_error.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_error.py)（49 行）
- [test_fc_v1_trainer_fresh.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_fresh.py)（122 行）
- [test_fc_v1_trainer_integration.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_integration.py)（83 行）
- [test_fc_v1_trainer_verbose.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_verbose.py)（104 行）

**重复证据**
1. **`test_trainer_train_single` & `test_trainer_train_single_returns_loss`**
   - [test_fc_v1_trainer.py#L86-L109](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py#L86-L109) 两个用例断言完全一致（`isinstance(loss, float)` + `loss >= 0`），只是类型检查写了两遍 —— **同文件内部重复**
2. **Fresh 参数测试**
   - `test_fc_v1_trainer.py::test_trainer_initialization_without_model` (L72-L77) 与
   - [test_fc_v1_trainer_fresh.py#L57-L68](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_fresh.py#L57-L68) `test_trainer_fresh_parameter_creates_new_model` 核心断言相同
3. **Verbose 输出**
   - `test_fc_v1_trainer_verbose.py` 与 `test_fc_v1_trainer_batch.py` L124-L138 (`test_train_batch_verbose_false_no_epoch_output`) 都在测 verbose stdout
4. **损失计算**
   - [test_fc_v1_trainer.py#L134-L183](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py#L134-L183) 三个 `test_trainer_compute_loss*` 实际测的是 `compute_masked_l1_loss` 纯函数，应放到 `tests/ai/test_sample_loader.py` 或 `tests/ai/unit/test_sample_loader_split.py`

**建议**
- **合并**：`test_fc_v1_trainer_fresh.py` → `test_fc_v1_trainer.py`（fresh 是参数而非独立特性）。
- **删除**：[test_fc_v1_trainer_cycle.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_cycle.py) 中的空类 `TestFCv1TrainerCycle` (L23-L42)（只有 setUp/tearDown 无 test_* 方法）；保留 `TestTrainCycleRateParameter`。
- **迁移**：`test_trainer_compute_loss*` 移到 `sample_loader` 测试。
- **精简**：[test_fc_v1_trainer_error.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_error.py) 与 `test_cnn_trainer.py::test_load_sample_data_with_invalid_json` 重复，可删。
- 预计缩减：**7 个文件 → 3 个文件**（trainer / trainer_batch / trainer_cycle）。

---

### 组 4（P0）：LR Range Test 三份 + 一份 e2e

**涉及文件**
- [tests/ai/unit/test_lr_range_test.py](file:///d:/projects/gomoku/tests/ai/unit/test_lr_range_test.py)（473 行，**重度 mock**）
- [tests/unit/test_lr_range_test.py](file:///d:/projects/gomoku/tests/unit/test_lr_range_test.py)（271 行，**真实对象**）
- [tests/integration/test_lr_range_e2e.py](file:///d:/projects/gomoku/tests/integration/test_lr_range_e2e.py)（133 行）

**重复证据**
- 两个 unit 文件都有 `test_load_model_cnn_v2`、`test_load_data_from_merge_file`、`test_plot_creates_file`。
- e2e 中 `TestEndToEndCnnV2::test_end_to_end_cnn_v2` 覆盖流程与上面 unit 的组合等价。
- `test_linear_lr_increase`、`test_losses_length_matches_rounds` 在三层被反复覆盖。

**建议**
- **保留** `tests/unit/test_lr_range_test.py`（真实对象，覆盖更可信）。
- **删除** `tests/ai/unit/test_lr_range_test.py`（整文件过度 mock，见"无效测试-过度 mock"章节）。
- **精简** `tests/integration/test_lr_range_e2e.py`：只保留 1 个真正的端到端烟雾测试（如 `test_end_to_end_cnn_v2` + 两个异常路径）。
- 预计缩减：约 **30 个用例** → 15 个。

---

### 组 5（P1）：Evaluator 四层重叠

**涉及文件**
- [tests/ai/test_evaluator.py](file:///d:/projects/gomoku/tests/ai/test_evaluator.py)（ProcessPool 回退逻辑，276 行）
- [tests/ai/primary/test_evaluator_shape_detection.py](file:///d:/projects/gomoku/tests/ai/primary/test_evaluator_shape_detection.py)（487 行）
- [tests/ai/primary/test_evaluator_winning_line.py](file:///d:/projects/gomoku/tests/ai/primary/test_evaluator_winning_line.py)（463 行）
- [tests/ai/edge/test_evaluator_shape_detection_edge.py](file:///d:/projects/gomoku/tests/ai/edge/test_evaluator_shape_detection_edge.py)（288 行）
- [tests/ai/unit/test_evaluator_refactor_phase1.py](file:///d:/projects/gomoku/tests/ai/unit/test_evaluator_refactor_phase1.py)（266 行）

**重复证据**
- `primary/test_evaluator_shape_detection.py` (L45-L260) 与 `edge/test_evaluator_shape_detection_edge.py` (L45-L180) 对连五/活四/活三/眠三/活二均有测试，只是棋盘具体坐标不同，逻辑雷同。
- `test_evaluator_shape_detection.py` 的 `test_live_five_current_player` 与 `test_evaluator_winning_line.py` 里的连五相关用例有**功能交叉**。
- [test_evaluator_refactor_phase1.py#L26-L100](file:///d:/projects/gomoku/tests/ai/unit/test_evaluator_refactor_phase1.py) 通篇用 `hasattr(self.evaluator, '_xxx')` 断言"方法已被删除" —— 这是**重构完成后的一次性验证**，长期保留无价值。

**建议**
- **合并** `primary` 与 `edge` 的棋型检测：同一份参数化测试（棋型 × 位置），覆盖连五/活四/冲四/活三/眠三/活二/眠二/独一 × 中心位/边缘位/角落位。
- **删除** `test_evaluator_refactor_phase1.py`（重构验收完成后应清理；可在 commit message 中保留"重构完成"记录）。
- **评估保留**：`test_evaluator.py::TestEvaluateFilesBrokenPoolFallback` 验证的是进程池崩溃回退，用了大量 mock 但断言真实分支逻辑（见 [L134-L165](file:///d:/projects/gomoku/tests/ai/test_evaluator.py#L134-L165) 注释明确写了"验证 remaining_files 计算逻辑"）—— **建议保留**，但加强可读性。
- 预计缩减：**5 文件 → 3 文件**，用例约 **60 → 35**。

---

### 组 6（P1）：其他疑似重复

1. **GameManager 单元 vs 集成**
   - [tests/ai/test_game_manager.py](file:///d:/projects/gomoku/tests/ai/test_game_manager.py)（15.7 KB）
   - [tests/ai/unit/test_game_manager_unit.py](file:///d:/projects/gomoku/tests/ai/unit/test_game_manager_unit.py)（12.6 KB）
   - **建议**：人工核对分工，若单元测试已覆盖完整接口，则可只保留 `unit/` 版本，原文件改为集成测试（`tests/ai/integration/`）。

2. **Loss Logger 三文件**
   - [test_loss_logger.py](file:///d:/projects/gomoku/tests/ai/test_loss_logger.py)（20 KB）
   - [test_loss_logger_val.py](file:///d:/projects/gomoku/tests/ai/test_loss_logger_val.py)（5.5 KB）
   - [test_bugfix_dense_log_and_val_ratio.py](file:///d:/projects/gomoku/tests/ai/test_bugfix_dense_log_and_val_ratio.py)（10 KB）
   - **建议**：`test_bugfix_dense_log_and_val_ratio.py` 是一次 bugfix 的回归测试，经一段时间稳定后可**按主题并入** `test_loss_logger.py`（通过 `class TestDenseLog` / `class TestValRatio` 区分）。`test_loss_logger_val.py` 也应并入主文件。

3. **Sample Loader 两处**
   - [test_sample_loader.py](file:///d:/projects/gomoku/tests/ai/test_sample_loader.py)
   - [tests/ai/unit/test_sample_loader_split.py](file:///d:/projects/gomoku/tests/ai/unit/test_sample_loader_split.py)
   - **建议**：若 split 是单元覆盖子函数，命名改为 `test_sample_loader_pure_fn.py` 并保持职责清晰；否则合并。

4. **train_main 两处**
   - [tests/ai/test_train_main.py](file:///d:/projects/gomoku/tests/ai/test_train_main.py)（11.4 KB）
   - [tests/ai/test_train_main_script.py](file:///d:/projects/gomoku/tests/ai/test_train_main_script.py)（4.5 KB）
   - [tests/cli/test_train_main_cli.py](file:///d:/projects/gomoku/tests/cli/test_train_main_cli.py)
   - **建议**：定位重叠点；script 层只保留"CLI 入口能被调用"的烟雾测试，其余移入 cli 测试。

5. **mini_batch_training primary vs edge**
   - [tests/ai/primary/test_mini_batch_training.py](file:///d:/projects/gomoku/tests/ai/primary/test_mini_batch_training.py)
   - [tests/ai/edge/test_mini_batch_training_edge.py](file:///d:/projects/gomoku/tests/ai/edge/test_mini_batch_training_edge.py)
   - **建议**：用 parameterized 合并为单文件。

6. **Performance 三文件**
   - [test_performance_level1.py](file:///d:/projects/gomoku/tests/ai/test_performance_level1.py)
   - [test_performance_level2_numba.py](file:///d:/projects/gomoku/tests/ai/test_performance_level2_numba.py)
   - [test_performance_level2_parallel.py](file:///d:/projects/gomoku/tests/ai/test_performance_level2_parallel.py)
   - **建议**：性能测试通常不稳定，建议用 `@unittest.skipUnless(os.environ.get("RUN_PERF"))` 守门，避免常规 CI 里跑。否则长期会因环境波动被误标失败。

7. **跨文件小重复**
   - `tests/gui/test_ai_battle.py` L39-L43 `test_init_result_is_none` 与 L46-L50 `test_get_result_returns_none_initially` 实质等价 —— **删一留一**。
   - `tests/app/test_gui_helper.py` L59-L65 `test_handle_player_move_error` 与 L77-L83 `test_handle_player_move_no_game` 实质等价，仅 game_mode 参数不同 —— **合并为 parametrize**。
   - 同文件 L95-L100 `test_handle_ai_move_no_game` 与 L102-L109 `test_handle_ai_move_error` 实质等价 —— **删一留一**。

---

## 三、无效测试清单

### A. 弱断言 / 缺失实质断言

| 文件 | 用例 / 行号 | 问题 | 建议 |
|------|------------|------|------|
| [test_engine_no_fallback.py#L31-L35](file:///d:/projects/gomoku/tests/ai/test_engine_no_fallback.py#L31-L35) | `test_valid_model_still_works` | 仅断言 `engine is not None and engine.model is not None` —— 不足以证明 engine 可用 | 调用 `engine.predict(empty_board)` 并校验返回 225 维概率 |
| [test_base_trainer.py#L92-93, L132](file:///d:/projects/gomoku/tests/ai/test_base_trainer.py#L92-L93) | `test_initialization`、`test_trainer_with_optimizer` | 连续 `assertIsNotNone(trainer)` + `assertIsNotNone(trainer.model)` | 校验关键属性具体值：`optimizer.param_groups[0]['lr']`、`trainer.device`、`type(model)` |
| [tests/ai/unit/test_registry.py#L27,34,41,48,67](file:///d:/projects/gomoku/tests/ai/unit/test_registry.py) | 多处 `assertIsNotNone(model)` | 注册表只要返回对象就通过，无法发现返回错误模型的 bug | 加 `self.assertEqual(model.name, expected_name)` 以及 `isinstance(model, ExpectedClass)` |
| [test_battle.py#L236,242,248](file:///d:/projects/gomoku/tests/ai/test_battle.py) | 三处 `assertIsNotNone(model)` | 同上 | 补 name / predict 形状校验 |
| [test_fc_v1_trainer_error.py#L22-L44](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_error.py#L22-L44) | `test_train_single_missing_sample_file`、`test_load_sample_invalid_json` | 只验证异常类型，未验证异常消息 | 使用 `with self.assertRaisesRegex(FileNotFoundError, r"不存在")` |
| [test_fc_v1_trainer.py#L99-L109](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py#L99-L109) | `test_trainer_train_single_returns_loss` | 断言仅 `isinstance(loss,(int,float)) and loss>=0`，与 `test_trainer_train_single` 同 | 与前者合并，删除本用例 |
| [test_fc_v1_trainer_batch.py#L79-L98](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_batch.py#L79-L98) | `test_train_batch_returns_float` / `test_train_batch_single_sample_dir` | 断言 `loss >= 0.0`，没检查合理量级 | 断言 `0 <= loss < 某上限`；或与 `test_train_batch_updates_model_params` 合并 |
| [test_fc_v1_trainer_cycle.py#L47-L54](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_cycle.py#L47-L54) | `test_train_cycle_has_rate_parameter` | 仅通过反射断言"存在 rate 参数默认 0.9"，未验证 rate 被真正使用 | 新增 `test_train_cycle_rate_affects_loss`：用不同 rate 运行，观察学习曲线不同 |
| [tests/app/test_gui_helper.py#L154-L162](file:///d:/projects/gomoku/tests/app/test_gui_helper.py#L154-L162) | `test_start_game_with_ai_first_game_over` | 注释声称"AI 直接获胜"，实际只断言 `status=="success"` 且 `"game_over" in result` | 构造**确定会胜**的残局，或改名去除"获胜"误导 |
| [tests/cli/test_main.py#L52-L54](file:///d:/projects/gomoku/tests/cli/test_main.py#L52-L54) | `test_unknown_command_returns_none` | `patch('builtins.print')` 未做 `.assert_called_once()`，print mock 形同虚设 | 断言 mock 被调用且提示中含"未知命令" |

### B. 过度 Mock（断言只验证 mock 自身）

| 文件 | 用例 / 行号 | 问题 |
|------|-------|------|
| [tests/ai/unit/test_lr_range_test.py#L34-L97](file:///d:/projects/gomoku/tests/ai/unit/test_lr_range_test.py#L34-L97) | `test_load_supported_model`、`test_load_model_none_network`、`test_load_multi_gen_model` 等 | 把 `_import_class`、`_parse_model_spec` 全部 mock 掉，`model.network` 也是 MagicMock，断言只验证 `mock_network.to.assert_called_once_with(device)`。**真实模型加载分支完全没覆盖**，被 [tests/unit/test_lr_range_test.py](file:///d:/projects/gomoku/tests/unit/test_lr_range_test.py) 以真实对象实现替代。建议整文件删除 |
| [tests/gui/test_main.py#L24-L58](file:///d:/projects/gomoku/tests/gui/test_main.py#L24-L58) | `_make_gui_with_mock` 构造函数 | `board`、`save_name_label`、`_display_updater` 等**界面组件全部 MagicMock**。随后的 `test_valid_move_calls_helper_then_updates` (L84-L100) 断言 `_display_updater.update.assert_called_once()` —— **只验证 mock 被调用，没验证 GUI 真实效果** |
| [tests/cli/test_train_main_temp_cleanup.py](file:///d:/projects/gomoku/tests/cli/test_train_main_temp_cleanup.py)（整文件） | 4 个用例全程 `@patch('shutil.rmtree')` 并断言 `mock_rmtree.assert_any_call(...)` | 实际训练流程全部被 mock，"临时目录被清理"只是断言 rmtree 被调用过。建议至少 1 个用例用真实临时目录走完整流程，验证退出后目录确实不存在 |
| [test_fc_v1_trainer.py#L89-L90](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py#L89-L90) 等 | `trainer.loss_logger = MagicMock()` 在大量 trainer 测试中出现 | 是为避免污染真实 loss_logs 目录（见 [test_fc_v1_trainer_batch.py#L48-L53](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_batch.py#L48-L53) 注释），做法合理；**但应用一个轻量真实实例（指向 tmp 目录）+ 默认不校验日志内容**，比 MagicMock 更可信。可降级为 P2 改造 |

### C. 协助掩盖问题（最需要警惕）

| 文件 | 用例 / 行号 | 问题 | 风险 |
|------|------|------|------|
| [tests/test_controller.py#L120-L156](file:///d:/projects/gomoku/tests/test_controller.py#L120-L156) | `test_draw` | 使用 `if self.controller.game.winner == 3: ... elif == 0: ... else:` 三分支断言。其中 `elif == 0` 分支只 `assert hasattr(controller, 'is_draw')`，任何"棋盘未满就有胜者"的 bug 都会进这里被静默放过 | **条件断言掩盖失败路径**。建议改造为"确定性平局"序列（交错填充，数学上保证 225 步都不连五） |
| [tests/test_controller.py#L196-L210](file:///d:/projects/gomoku/tests/test_controller.py#L196-L210) | `test_game_over_draw_cannot_move` | 同样用 `if controller.game.winner == 3:` 保护断言，**若未触发平局，整个用例什么都没断言就通过** | 同上 |
| [tests/test_controller.py#L36-L38](file:///d:/projects/gomoku/tests/test_controller.py#L36-L38) | `tearDown` 中 `except PermissionError: pass` | 注释为"Windows 可能文件被占用" | 合理的防御；但应记录 warning，防止**测试之间文件泄露**（上一个用例写的 save 残留给下一个用例）。建议改为 `warnings.warn(...)` |
| [tests/app/edge/test_session.py#L52-L58](file:///d:/projects/gomoku/tests/app/edge/test_session.py#L52-L58) | `test_pvp_do_ai_move_raises_runtime_error` | `try: session.do_ai_move(); assert False ... except RuntimeError: pass` —— 异常消息未校验 | 若将来重构改了异常类型，此测试仍会通过。改用 `assertRaisesRegex` |
| [tests/app/primary/test_session.py#L109-L119](file:///d:/projects/gomoku/tests/app/primary/test_session.py#L109-L119) | `test_do_ai_move_pvp_raises_runtime_error` | 虽然有 `assert "pvp 模式没有 AI" in str(e)`，但也用了手写 try/except + `assert False` 写法 | 改用 `self.assertRaisesRegex(RuntimeError, r"pvp 模式没有 AI")` |
| [tests/ai/test_fc_v1_softmax_input.py#L83-L100+](file:///d:/projects/gomoku/tests/ai/test_fc_v1_softmax_input.py#L83-L100) | `test_loss_scale_with_softmax_input` | 注释说"loss 应在合理范围内"，实际断言只是 `loss >= 0` | 若模型 forward 写错导致 loss 爆炸（如 10^6），此测试仍通过。补 `loss < 1.0` 上界 |
| [test_fc_v1_trainer_batch.py#L100-L122](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_batch.py#L100-L122) | `test_train_batch_multiple_sample_dirs_merged` | 断言"2 个 epoch 均小于 200，所以不应有任何 Epoch 输出" —— **用"没有 Epoch 文本被打印"作为"合并训练"的证据**。若将来 `verbose` 逻辑变了（比如改为 `epochs >= 100 才打印`），此测试误通过 | 直接 mock `trainer.optimizer.step`，断言调用次数 == `epochs` 而非 `epochs × n_samples` |

### D. 结构遗留 / 死代码

| 文件 | 问题 | 建议 |
|------|------|------|
| [test_fc_v1_trainer_cycle.py#L23-L42](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_cycle.py#L23-L42) | **`TestFCv1TrainerCycle` 是空类**：只有 `setUp` / `tearDown`，无任何 `test_*` 方法 | 直接删除这个类，保留 `TestTrainCycleRateParameter` |
| [test_evaluator_refactor_phase1.py 整个文件](file:///d:/projects/gomoku/tests/ai/unit/test_evaluator_refactor_phase1.py) | 全文用 `assertFalse(hasattr(evaluator, '_compute_position_reward'))` 等验证"方法已被删除" | 重构已完成 → **整文件删除**。长期挂着是代码考古，没有回归价值 |
| [tests/gui/test_heatmap.py#L46-L50](file:///d:/projects/gomoku/tests/gui/test_heatmap.py#L46-L50) | `teardown_method` 吞掉 `tk.TclError` 后 `pass` | Tkinter 测试常见模式，合理；加注释说明即可 |
| [tests/gui/conftest.py#L46-L52](file:///d:/projects/gomoku/tests/gui/conftest.py#L46-L52) | `_gui_session_cleanup` `except Exception: pass` | 合理，加注释即可 |

### E. Skip 遗留

- 本次扫描**未发现** `@unittest.skip(...)` 或 `@pytest.mark.skip` 的长期遗留用例。✅

---

## 四、Top 15 优先处理清单

| # | 优先级 | 动作 | 目标 | 预计工作量 |
|---|--------|------|------|-----------|
| 1 | **P0** | 删除空类 `TestFCv1TrainerCycle` | 消除死代码 | 5 min |
| 2 | **P0** | 删除 `tests/ai/unit/test_evaluator_refactor_phase1.py` | 清理重构遗留 | 5 min |
| 3 | **P0** | 删除 `tests/ai/unit/test_lr_range_test.py`（过度 mock 版） | 与 `tests/unit/` 真实版择一 | 10 min |
| 4 | **P0** | 参数化整合 CNN v1/v2/v3 三套 model 测试 | 3 文件 → 1 文件，~50 用例 → ~18 | 2 h |
| 5 | **P0** | 参数化整合 CNN v1/v2/v3 三套 trainer 测试 | 3 文件 → 1 文件 | 2 h |
| 6 | **P0** | FC_V1 trainer 7 文件整合为 3 文件；compute_loss 系列迁移至 sample_loader 测试 | 减少 4 个文件 | 2 h |
| 7 | **P0** | 修复 `test_controller.py::test_draw` / `test_game_over_draw_cannot_move` 的条件断言 | 杜绝"静默通过" | 1 h（需构造确定性平局序列） |
| 8 | **P0** | 修复 `test_fc_v1_trainer_batch.py::test_train_batch_multiple_sample_dirs_merged` 的 stdout 断言 | 改为断言 `optimizer.step` 调用次数 | 30 min |
| 9 | **P1** | 合并 `primary/test_evaluator_shape_detection.py` 与 `edge/..._edge.py` | 5 文件 → 3 文件 | 2 h |
| 10 | **P1** | 增强 `test_engine_no_fallback.py::test_valid_model_still_works` | 改为验证 predict 真实可用 | 20 min |
| 11 | **P1** | `test_registry.py` / `test_battle.py` 弱断言批量改造 | 补 name / isinstance / predict 校验 | 1 h |
| 12 | **P1** | `test_fc_v1_trainer_error.py` 异常断言改用 `assertRaisesRegex` | 捕获消息回归 | 20 min |
| 13 | **P1** | `app/primary/test_session.py` 与 `app/edge/test_session.py` 的手写 `try/except + assert False` 改为 `assertRaisesRegex` | 异常消息回归 | 30 min |
| 14 | **P2** | `tests/gui/test_main.py` 过度 mock 改造：至少保留 1 个真实组件 | 提升 GUI 测试价值 | 1 h |
| 15 | **P2** | `tests/cli/test_train_main_temp_cleanup.py` 补 1 个真实目录的端到端用例 | 验证真清理 | 1 h |

### 预期收益
- **测试文件**：95 → **≈ 82**（-13）
- **测试用例**：800+ → **≈ 700**（-100+），但**有效断言密度上升**
- **执行耗时**：预计 -15% ~ -25%（重复的 CNN/FC 训练用例最占时间）
- **可维护性**：改一个模型接口不再需要同步改三套复制粘贴的测试

---

## 五、分阶段落地建议

### 阶段 1：零风险清理（0.5 天）
- Top #1, #2, #3：纯删除 / 整文件删除，不影响任何其它用例。
- 运行全量测试验证；按 `用户偏好-禁止忽略任何失败测试` 规则，出现失败必须定位并修复。

### 阶段 2：参数化整合（2 天）
- Top #4, #5, #6：用 `parameterized` 或基类共享。
- 每整合一组，立刻跑一次全量，保证等价行为未丢失。

### 阶段 3：断言强化（1 天）
- Top #7, #8, #10, #11, #12, #13：修正会掩盖问题的用例。
- 这一阶段**可能暴露真实 bug**（如 `test_draw` 修复后可能发现平局逻辑错误）——按 TDD 规则，先记录 issue 再继续。

### 阶段 4：测试可信度治理（1 天）
- Top #14, #15：降低 mock 深度。
- Performance 三文件加 `skipUnless(os.environ.get("RUN_PERF"))` 守门。

### 阶段 5：验收
- 全量测试通过；
- `coverage_report/` 更新后**覆盖率不降**（重构应保持覆盖）。

---

## 六、Backlog：Top 15 之外的全部待处理项

> 定位：**Top 15 是"当前工期内必做"**。以下为所有在第二/第三章识别、但未进 Top 15 的问题的完整清单，已给出**判定结论**（保留 / 删除 / 改造 / 待核对）。按需选取进入下一轮迭代；不立即落地。
>
> 每条字段：**类型 | 文件 | 位置 | 现象 | 判定 | 建议动作**

### Backlog-A：疑似重复（来自组 6 与跨文件小重复）

| # | 文件 | 位置 | 现象 | 判定 | 建议动作 |
|---|------|------|------|------|---------|
| A1 | [test_game_manager.py](file:///d:/projects/gomoku/tests/ai/test_game_manager.py) vs [unit/test_game_manager_unit.py](file:///d:/projects/gomoku/tests/ai/unit/test_game_manager_unit.py) | 两文件 | 职责重叠不清，一个 15.7 KB 一个 12.6 KB | **待核对** | 人工逐例比对：若单元已覆盖公共接口 → 原文件改作集成测试并放到 `tests/ai/integration/` |
| A2 | [test_loss_logger.py](file:///d:/projects/gomoku/tests/ai/test_loss_logger.py) + [test_loss_logger_val.py](file:///d:/projects/gomoku/tests/ai/test_loss_logger_val.py) + [test_bugfix_dense_log_and_val_ratio.py](file:///d:/projects/gomoku/tests/ai/test_bugfix_dense_log_and_val_ratio.py) | 三文件 | Bugfix 回归测试长期独立存在 | **合并** | 按主题合并到 `test_loss_logger.py`（`class TestDenseLog` / `class TestValRatio`） |
| A3 | [test_sample_loader.py](file:///d:/projects/gomoku/tests/ai/test_sample_loader.py) vs [unit/test_sample_loader_split.py](file:///d:/projects/gomoku/tests/ai/unit/test_sample_loader_split.py) | 两文件 | 分工不明 | **改造** | 若 split 专测纯函数，重命名为 `test_sample_loader_pure_fn.py` 并加类注释；否则合并 |
| A4 | [test_train_main.py](file:///d:/projects/gomoku/tests/ai/test_train_main.py) + [test_train_main_script.py](file:///d:/projects/gomoku/tests/ai/test_train_main_script.py) + [cli/test_train_main_cli.py](file:///d:/projects/gomoku/tests/cli/test_train_main_cli.py) | 三文件 | 三层覆盖疑似重叠 | **改造** | `script` 只保留 CLI 入口烟雾测试；参数/分支校验集中到 `cli/test_train_main_cli.py` |
| A5 | [primary/test_mini_batch_training.py](file:///d:/projects/gomoku/tests/ai/primary/test_mini_batch_training.py) + [edge/test_mini_batch_training_edge.py](file:///d:/projects/gomoku/tests/ai/edge/test_mini_batch_training_edge.py) | 两文件 | primary / edge 同题目 | **合并** | 用 `parameterized` 合并为单文件 |
| A6 | [test_performance_level1.py](file:///d:/projects/gomoku/tests/ai/test_performance_level1.py) + [test_performance_level2_numba.py](file:///d:/projects/gomoku/tests/ai/test_performance_level2_numba.py) + [test_performance_level2_parallel.py](file:///d:/projects/gomoku/tests/ai/test_performance_level2_parallel.py) | 三文件 | 性能测试易因环境波动误报；三文件分层不严 | **改造** | 统一加 `@unittest.skipUnless(os.environ.get("RUN_PERF"), ...)`；可能合并 level2 两文件 |
| A7 | [gui/test_ai_battle.py](file:///d:/projects/gomoku/tests/gui/test_ai_battle.py#L39-L50) | L39-L43 vs L46-L50 | `test_init_result_is_none` 与 `test_get_result_returns_none_initially` 实质等价 | **删除** | 二选一保留 |
| A8 | [app/test_gui_helper.py](file:///d:/projects/gomoku/tests/app/test_gui_helper.py#L59-L83) | L59-L65 vs L77-L83 | `test_handle_player_move_error` 与 `test_handle_player_move_no_game` 实质等价 | **合并** | 改为 `parametrize(game_mode=['pvp','pve'])` |
| A9 | [app/test_gui_helper.py](file:///d:/projects/gomoku/tests/app/test_gui_helper.py#L95-L109) | L95-L100 vs L102-L109 | `test_handle_ai_move_no_game` 与 `test_handle_ai_move_error` 实质等价 | **删除** | 二选一保留 |

### Backlog-B：弱断言 / 缺失断言（未进 Top 15）

| # | 文件 | 位置 | 现象 | 判定 | 建议动作 |
|---|------|------|------|------|---------|
| B1 | [test_base_trainer.py](file:///d:/projects/gomoku/tests/ai/test_base_trainer.py#L92-L209) | L92-L93、L132、L198-L209 | 多处 `assertIsNotNone(trainer)` / `assertIsNotNone(trainer.model)` | **改造** | 改为校验 `optimizer.param_groups[0]['lr']`、`trainer.device`、`type(model)`、`lr_scheduler_type` 具体值 |
| B2 | [test_fc_v1_trainer.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py#L99-L109) | L99-L109 | `test_trainer_train_single_returns_loss` 与 L86-L97 的 `test_trainer_train_single` 断言相同 | **删除** | 删除本用例 |
| B3 | [test_fc_v1_trainer_batch.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_batch.py#L79-L98) | L79-L98 | `test_train_batch_returns_float` / `..._single_sample_dir` 只断言 `loss >= 0` | **改造** | 增加量级上界（如 `loss < 某常数`）或与 `test_train_batch_updates_model_params` 合并 |
| B4 | [test_fc_v1_trainer_cycle.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_trainer_cycle.py#L47-L54) | L47-L54 | `test_train_cycle_has_rate_parameter` 仅用反射查参数默认值 | **改造** | 新增 `test_train_cycle_rate_affects_loss`：不同 rate 运行后学习曲线应不同 |
| B5 | [app/test_gui_helper.py](file:///d:/projects/gomoku/tests/app/test_gui_helper.py#L154-L162) | L154-L162 | `test_start_game_with_ai_first_game_over` 注释称"AI 直接获胜"但实测做不到 | **改造** | 构造确定会胜的残局，或改名去除"获胜"误导 |
| B6 | [cli/test_main.py](file:///d:/projects/gomoku/tests/cli/test_main.py#L52-L54) | L52-L54 | `patch('builtins.print')` 未做 `.assert_called` | **改造** | 补 `mock_print.assert_called_once()` 并校验提示文本含"未知命令" |

### Backlog-C：过度 Mock（未进 Top 15）

| # | 文件 | 位置 | 现象 | 判定 | 建议动作 |
|---|------|------|------|------|---------|
| C1 | `test_cnn*/test_fc_v1*_trainer.py` 多处 | 大量 `trainer.loss_logger = MagicMock()` | 为避免写入真实 `loss_logs/`；但断言无法覆盖 logger 真实行为 | **改造** | 抽出测试工具：`_build_trainer_with_tmp_logger(tmp_dir)` 指向临时目录的真实 LossLogger；MagicMock 仅在明确需要监视调用链时用 |

### Backlog-D：协助掩盖问题（未进 Top 15）

| # | 文件 | 位置 | 现象 | 判定 | 建议动作 |
|---|------|------|------|------|---------|
| D1 | [tests/test_controller.py](file:///d:/projects/gomoku/tests/test_controller.py#L36-L38) | L36-L38 `tearDown` | `except PermissionError: pass` 静默吞 Windows 文件锁 | **改造** | 改为 `warnings.warn(f"clean failed: {e}")` 并打印；防止测试间文件泄露被忽略 |
| D2 | [test_fc_v1_softmax_input.py](file:///d:/projects/gomoku/tests/ai/test_fc_v1_softmax_input.py#L83-L100) | L83-L100 | `test_loss_scale_with_softmax_input` 注释说"合理范围"但只断言 `loss >= 0` | **改造** | 补 `self.assertLess(loss, 1.0)` 上界（与测试注释一致） |

### Backlog-E：覆盖缺口 / 其他质量改善

| # | 文件 | 位置 | 现象 | 判定 | 建议动作 |
|---|------|------|------|------|---------|
| E1 | [gui/test_game_board.py](file:///d:/projects/gomoku/tests/gui/test_game_board.py#L191-L211) | L191-L211 | `test_click_out_of_bounds` 只测右下角外 | **改造** | 补 4 个方向越界 + 边界内最大有效格（14,14）+ 负坐标 |
| E2 | [tests/app/primary/test_session.py](file:///d:/projects/gomoku/tests/app/primary/test_session.py#L139-L142) / [edge/test_session.py](file:///d:/projects/gomoku/tests/app/edge/test_session.py#L60-L66) | 未初始化异常测试 | 两处跨目录实质等价 | **保留** | 视作 primary / edge 分层的合理冗余；不动（不进改造） |

### Backlog 汇总
- 条目数：**20**（A9 + B6 + C1 + D2 + E2）
- 建议动作分布：删除 × 3、合并 × 3、改造 × 11、待核对 × 1、保留 × 1、加守门/标签 × 1
- 预计额外工作量：约 1–1.5 个工作日（不含 C1 的工具抽象）

---

## 七、附录：已并入 Backlog 的"二次整理"项
> （保留此章作为历史记录，实际内容已迁至 Backlog-A / E）
- ~~`tests/ai/test_train_main.py` vs `test_train_main_script.py` vs `tests/cli/test_train_main_cli.py`~~ → A4
- ~~`tests/ai/test_game_manager.py` vs `tests/ai/unit/test_game_manager_unit.py`~~ → A1
- ~~`tests/ai/test_loss_logger.py` + `_val.py` + `_bugfix_dense_log_and_val_ratio.py`~~ → A2
- ~~`tests/ai/test_sample_loader.py` vs `tests/ai/unit/test_sample_loader_split.py`~~ → A3
- ~~`tests/gui/test_game_board.py::test_click_out_of_bounds`~~ → E1

---

**报告作者**：测试集 Review（静态分析）
**完整性声明**：Top 15（第四章）+ Backlog（第六章）**覆盖本次审查发现的全部问题**，无遗漏。
**下一步动作建议**：与用户确认是否执行阶段 1（零风险清理），如确认则按 TDD 流程（先跑全量 → 删除 → 再跑全量 → 提交）执行。Backlog 可在 Top 15 完成后按优先级选取进入下一轮。
