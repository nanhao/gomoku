# 测试套件优化设计方案

**意图ID**: 18
**创建日期**: 2026-06-22
**核心约束**: 所有 1659 个测试用例每次必须全部运行，不允许任何跳过/过滤机制
**目标**: 将全量测试单次运行时间压缩至约 1 分钟

---

## 1. 问题分析

### 1.1 现状

| 指标 | 数值 |
|------|------|
| 测试文件数 | 97 |
| 测试用例总数 | 1659 |
| 仅收集用例耗时 | ~7 秒 |
| 全量运行实测耗时 | 122.47s（约 2:02） |

### 1.2 耗时根因分析

**A. 真实模型训练 (最核心瓶颈)**

部分测试使用了远超出实际需要的 epochs：

| 文件 | 当前 epochs | 实际意图 | 浪费程度 |
|------|------------|----------|----------|
| `test_base_trainer.py` | 400, 200 | 收敛性验证（过拟合单样本） | 保留 |
| `test_base_trainer_val.py` | 200×2 | 验证 loss/val_loss 分离逻辑 | **严重浪费** |
| `test_base_trainer_train_auto.py` | 1000, 50×4 | 验证参数透传 | **严重浪费** |
| `test_fc_v1_trainer.py` | 200 | 验证单步训练流程 | **浪费** |
| `test_mini_batch_training.py` | 200 | 验证 mini batch 调度 | **浪费** |
| `test_loss_logger.py` | 1000×2 | 验证日志记录 | **严重浪费** |
| `test_fc_v1_softmax_input.py` | 50 | 验证 softmax 处理 | 部分浪费 |
| `test_graceful_shutdown.py` | 10×4 | 验证中断处理 | 部分浪费 |

**B. PyTorch 模型实例化**
- 多个测试在 `setUp` 中反复创建 CNN/FC 模型
- `test_cnn_models.py` 对 v1~v6 每个版本都实例化 Network + Model

**C. GUI 测试**
- Tkinter 窗口创建/销毁开销大

**D. 性能优化的正确性验证**
- `test_performance_level1.py` — 正确性测试（Mock），快
- `test_performance_level2_numba.py` — JIT 正确性 + 一个 10000 次循环的性能对比
- `test_performance_level2_parallel.py` — 并行结果一致性，快

---

## 2. 优化策略

**只有一个方向：降低单测耗时。不跳过任何一个测试。**

```
优化方向
└── A. 降低单测耗时
    ├── A1. 减少训练 epochs
    ├── A2. Mock 替代真实训练（流程验证类测试）
    ├── A3. Fixture 作用域优化（减少重复初始化）
    └── A4. 精简鸡肋测试（删除低价值用例）
```

---

## 3. 核心策略详述

### 3.1 A1：减少训练 epochs

**原则**：收敛性验证保留大 epochs，流程验证降到最低。仅改动有真实训练的测试。

| 文件 | 当前 epochs | 改为 | 原因 | 状态 |
|------|------------|------|------|------|
| `test_mini_batch_training.py` | 10×2 | **5** | 验证 mini batch 调度 | ✅ 已改 |
| `test_fc_v1_softmax_input.py` | 50 | **5** | 验证 softmax 收敛 | ✅ 已改 |
| `test_graceful_shutdown.py` | 10×4 | **2** | 验证中断处理 | ✅ 已改 |

> **放弃改动的文件及原因**：
> - `test_base_trainer_val.py`：训练已被 `_forward_backward_step` mock，epochs=200 是触发 LOSS_LOG_INTERVAL 断言所必需
> - `test_fc_v1_trainer.py`：当前 epochs 已在 1-5 范围内，无需再降
> - `test_loss_logger.py`：无任何训练代码，纯日志测试，极快
> - `test_base_trainer_train_auto.py`：已完全 mock `train_batch`/`train_mini_batch`，无真实训练

### 3.2 A2：Mock 替代真实训练

> 原计划目标文件在实际代码中已全部使用 mock，无需额外改动。跳过。

### 3.3 A3：Fixture 作用域优化

将频繁创建的资源升级作用域：

| Fixture/资源 | 当前 | 改为 | 原因 |
|-------------|------|------|------|
| `temp_project` (ai/conftest.py) | function | module | 多个测试共享同一临时目录 |
| FCv1Model/RandomModel 实例 | 每个 setUp 创建 | module | 无状态模型可复用 |

### 3.4 A4：精简鸡肋测试

经过对 99 个测试文件的逐对交叉审查，发现以下可删除的低价值测试：

#### 3.4.1 签名检查测试 — 脆性高、价值低

**文件**: `tests/ai/test_evaluator.py` 中的 `TestEvaluatorSignatureChanges` 类（3 个测试）

用 `inspect.signature()` 检查函数参数名是否与预期一致。这类测试不验证行为只验证实现细节，一旦正常重构改名就会炸，且功能测试本身就能发现签名变化。

```python
# 示例：这测的是"参数名叫 history"而非"参数为 history 时行为正确"
def test_compute_all_rewards_has_only_history_param(self):
    sig = inspect.signature(Evaluator._compute_all_rewards)
    self.assertEqual(non_self_params, ['history'])
```

**操作**: 删除 `TestEvaluatorSignatureChanges` 整个类。保留同文件中的 `TestEvaluateFilesBrokenPoolFallback`（进程池崩溃回退逻辑，高价值）。

#### 3.4.2 空壳测试类 — 纯噪音

**文件**: `tests/ai/test_sample_creator_winner_path.py`

两个类文档自述"已移除"但代码还在：
- `TestWinnerPathOutputOneHot` — 零测试方法
- `TestWinnerPathInvalidPosition` — 零测试方法

**操作**: 删除这两个空壳类。

#### 3.4.3 错误处理测试重复

**文件**: `tests/integration/test_lr_range_e2e.py` 中的 `TestInvalidInputs` 类（2 个测试）

- `test_invalid_model_name` — 无效模型名 → ValueError
- `test_nonexistent_data_path` — 不存在路径 → FileNotFoundError

这两个错误处理在 `tests/unit/test_lr_range_test.py` 中已有完全等价的覆盖（`TestLoadModel.test_load_model_invalid_name` 和 `TestLoadData.test_load_data_nonexistent_path`）。

**操作**: 删除 `TestInvalidInputs` 类，或考虑删除整个 `test_lr_range_e2e.py`（核心断言在单元测试中都已覆盖）。

#### 3.4.4 伪集成测试

**文件**: `tests/ai/test_fc_v1_trainer_integration.py`（83 行，仅 1 个测试）

测试名叫 `test_train_single_improves_model`（"训练改善模型"），但实际断言只检查 `loss >= 0` 和输出长度=225——完全不验证 loss 有没有下降。单元测试 `test_fc_v1_trainer.py` 已覆盖同样的内容。

**操作**: 删除整个文件。如果确实需要"训练改善模型"的端到端验证，应写一个真正断言 loss_before > loss_after 的测试。

#### 3.4.5 节省汇总

| 操作 | 影响文件 | 节省 | 状态 |
|------|---------|------|------|
| 删除签名检查类 | `test_evaluator.py` | ~49 行，3 个测试 | ✅ |
| 删除空壳类 | `test_sample_creator_winner_path.py` | ~10 行，0 个测试 | ✅ |
| 删除重复错误处理 | `test_lr_range_e2e.py` | ~29 行，2 个测试 | ✅ |
| 删除伪集成测试 | `test_fc_v1_trainer_integration.py` | 83 行，1 个测试 | ✅ |
| **合计** | | **~172 行，约 6 个测试用例** | |

> 注：均不影响功能覆盖。删除的都是重复覆盖或零覆盖的测试。

### 3.5 附带修复：生产代码中的 subprocess 参数类型错误

**文件**: `ai/iterative_train.py` 第 252 行

```python
# 修复前（整数混入 subprocess 参数列表）
"--temp-moves", 100

# 修复后（与其他所有数值参数一致，使用字符串）
"--temp-moves", "100"
```

> 该 bug 导致 5 个测试因 `TypeError: " ".join()` 失败（`call_args` 列表中混入 int 无法 join）。

### 3.6 A5：第二轮鸡肋测试精简（基于 --durations 分析）

**原则**：\"迁移已完成\"型验证一律删除；纯属性赋值检查删除；重复覆盖只保留最强版本；大数据量缩小到够用即可。

#### 3.6.1 kl_div 迁移验证测试 — 迁移窗口已关闭

**文件**: `tests/ai/test_kl_div_migration.py`

| 测试 | 理由 | 操作 |
|------|------|------|
| T3-01~03: 检查 3 个文件是否物理删除 | 文件删了就不会重现 | 删除 |
| T3-04: 遍历源码搜索 SampleInspector 残留 | 同上，高 I/O | 删除 |
| T6-03: 遍历 models/ 搜索 Softmax 残留 | 同上 | 删除 |
| T6-04: 遍历源码搜索 compute_masked_l1_loss | 同上 | 删除 |
| T7-02: 读源码检查 nn.KLDivLoss | 源码文本扫描，行为测试 T7-01 已覆盖 | 删除 |
| T9-03: 全零 target 调用 compute_masked_kl_loss | 与 T9-02 完全重复 | 删除 |

> **保留**: T6-01/02（LogSoftmax 输出行为）、T7-01（训练 loss 有限）、T9-01/02（数值稳定性）——真正的行为回归测试。

#### 3.6.2 并行评测中的配置赋值测试

**文件**: `tests/ai/test_performance_level2_parallel.py`

| 测试 | 理由 | 操作 |
|------|------|------|
| `test_max_workers_stored` | 测试 Python 赋值语法 | 删除 |
| `test_default_max_workers_positive` | 默认值 ≥1 已被串行/并行测试隐式覆盖 | 删除 |
| `test_max_workers_one_is_serial` | 仅验证属性赋值，行为由串行测试覆盖 | 删除 |
| `test_worker_can_be_pickled_by_processpool` | 主并行测试已证明 worker 可 pickle，额外 spawn Pool 浪费 | 删除 |
| `test_serial_mode_result_matches_original` | 合并到 `test_serial_mode_produces_samples` | 合并 |
| `test_invalid_record_does_not_affect_others_serial` | 保留并行版即可覆盖 | 删除 |

#### 3.6.3 样本可视化中的弱断言测试

**文件**: `tests/ai/test_sample_visualizer.py`

| 测试 | 理由 | 操作 |
|------|------|------|
| `test_cli_help_exits_zero` | 测试 argparse --help，Python 标准库保证 | 删除 |
| `test_different_seed_selects_different_subset` | 弱断言（极小概率误报），同 seed 测试已约束行为 | 删除 |
| `test_same_seed_selects_same_subset` | 30→12 样本 | 精简 |
| `test_indices_selects_exact_samples` | 20→8 样本 | 精简 |
| `test_new_format_multiple_samples` | 6→3 样本 | 精简 |
| T17-1/2/3 参数验证 | alpha/上界/下界/mode 合并为 parametrize | 合并 |

#### 3.6.4 样本生成 CLI smoke 精简

**文件**: `tests/ai/test_sample_generator.py`

| 测试 | 修改 | 理由 |
|------|------|------|
| `test_T10_cli_generate_huo3` | --num-samples 20→5 | CLI 管道验证 5 个样本足够 |

#### 3.6.5 对战测试精简

**文件**: `tests/ai/test_battle.py`

| 测试 | 修改 | 理由 |
|------|------|------|
| `test_random_side_assignment` | 20→8 局 | 全同色概率 0.78% 已够低 |
| `test_draw_game_both_not_won` | 删除 | 与 `test_ai_won_flags_consistent` 重复 |

**文件**: `tests/app/test_battle.py`

| 测试 | 修改 | 理由 |
|------|------|------|
| temp_config 属性检查 ×2 | 合并为一个测试 | 避免重复 patch 开销 |

#### 3.6.6 第二轮节省汇总

| 类别 | 操作 | 预计节省 |
|------|------|----------|
| 迁移验证 | 删除 7 个 | ~2s + 消除 I/O 扫描 |
| 并行评测 | 删除 5 个 + 合并 2 个 | ~8s |
| 样本可视化 | 删除 2 个 + 精简 6 个 | ~8s |
| 样本生成 | 精简 1 个 | ~3s |
| 对战测试 | 删除 1 个 + 精简 1 个 | ~2s |
| **合计** | | **~23s** |

### 3.7 A6：第三轮鸡肋测试精简（全面扫描）

**原则**：测试 Python 语言特性的一律删除；已废弃参数/常量的验证删除；`assert True` 弱断言删除或改写；可合并的微测试合并。

#### 3.7.1 整文件删除（14 个测试）

| 文件 | 测试数 | 理由 |
|------|--------|------|
| `tests/app/test_exceptions.py` | 4 | 测 ModelLoadError 继承 Exception、str()、异常链等——全是 Python 自带机制 |
| `tests/ai/unit/test_iterative_train_dynamic_regex.py` | 10 | 与 `test_iterative_train.py` TestWinRateParsing 功能等价 |

#### 3.7.2 已废弃功能/参数验证（4 个测试）

| 文件 | 测试 | 理由 |
|------|------|------|
| `tests/cli/test_train_main_cli.py` | TestSelfPlayParamRemoved（2）+ TestRemovedCycleParam（1） | --self-play/--merge-output/--cycle 永远不会加回来 |
| `tests/ai/unit/test_iterative_train_constants.py` | test_deleted_constants_removed | 已删除常量不会再重现 |

#### 3.7.3 源码文本扫描测试（1 个）

| 文件 | 测试 | 理由 |
|------|------|------|
| `tests/ai/unit/test_iterative_train_constants.py` | test_no_hardcoded_model_specs_in_print | 用 inspect.getsource 扫源码文本——code review 的事 |

#### 3.7.4 重复覆盖（4 个测试）

| 文件 | 测试 | 理由 |
|------|------|------|
| `tests/integration/test_lr_range_e2e.py` | TestEndToEndCnnV2（2 个） | 与 `test_lr_range_test.py` 单元测试高度重叠，文件变空壳后整体删除 |
| `tests/gui/test_ai_battle.py` | test_get_available_ais_delegates_to_registry | 与 `test_mode_selector.py` 同逻辑完全重复 |
| `tests/gui/test_gomoku_gui.py` | test_is_over_method_exists | 方法重命名是很久前的重构，无回归风险 |
| `tests/ai/unit/test_iterative_train.py` | test_loop_always_continues | 逻辑已被 test_promote_moves_gen5 覆盖 |

#### 3.7.5 assert True 弱断言（7 个测试）

| 文件 | 测试 | 理由 |
|------|------|------|
| `tests/gui/test_replay_browser.py` | TestReplayBrowser（3 个） | callable() 检查无业务价值 |
| `tests/gui/test_replay_viewer.py` | test_draw_board_grid/test_draw_pieces/test_draw_heatmaps | `assert True`——等于没测 |
| `tests/gui/test_heatmap.py` | test_clear | `assert True` |

#### 3.7.6 GUI 琐碎测试合并（约 14 个测试 → 4 个）

| 文件 | 合并内容 | 变化 |
|------|---------|------|
| `tests/gui/test_ai_battle.py` | init_result + get_result → test_initial_state<br>num_games_stored + recorder_none + on_complete_none + root + status_label + progress_label → test_initial_window_state<br>run_ai_vs_ai_is_static + manager_has_run_method → 删除 | 10→3 |
| `tests/gui/test_game_board.py` | test_init + test_constants + test_canvas_size → test_initial_state | 3→1 |
| `tests/gui/test_mode_selector.py` | test_select_pve_black + white + different_ai → test_select_pve_creates_dialog<br>test_init + test_constants → test_initial_state | 5→2 |

#### 3.7.7 第三轮节省汇总

| 类别 | 操作 | 预计节省 |
|------|------|----------|
| 整文件删除 | 2 文件 14 测试 | ~1s |
| 废弃参数/常量 | 4 测试 | ~1s |
| 源码扫描 | 1 测试 | ~0s |
| 重复覆盖 | 4 测试 + 1 整文件 | ~3s |
| 弱断言 | 7 测试 | ~0.5s |
| GUI 合并 | 约 10 测试合并为 4 | ~1s |
| **合计** | | **~6.5s** |

---

## 4. pytest.ini 配置

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v -s
timeout = 60
filterwarnings =
    ignore::pytest.PytestUnknownMarkWarning
```

> 注意：`addopts` 中不包含任何 `-m` 过滤。默认运行始终执行全部测试。

---

## 5. 预期收益

### 5.1 优化前后对比（预估）

| 优化项 | 预估减少 | 说明 |
|--------|----------|------|
| A1: 减少 epochs | 40-60% 训练耗时 | train_batch 调用次数降低 10-100 倍 |
| A2: Mock 训练 | 20-30% 训练耗时 | 调度类测试从秒级降到毫秒级 |
| A3: Fixture 复用 | 5-10% 总耗时 | 减少重复模型创建 |
| A4: 精简鸡肋测试 | ~1% 总耗时 | 删除约 6 个低价值用例 |

**综合预估**：A1-A6 叠加可降低总耗时约 30-40%。若优化后仍在 2 分钟以上，可再评估是否引入并行。

> **实际结果（Phase 1 完成）**：1721 测试，0 失败，耗时 136.85s（约 2:17）。详见 §6。

> **Phase 1.5 目标**：在 Phase 1 基础上再节省 ~23s，目标 ~110s。

> **实际结果（Phase 1.5 完成）**：1703 测试，0 失败，耗时 127.75s（约 2:08）。

> **Phase 2.5 目标**：在 Phase 1.5 基础上再节省 ~6.5s，消除约 32 个低价值测试用例。

> **实际结果（Phase 2.5 完成）**：1659 测试，0 失败，耗时 122.47s（约 2:02）。累计删除/合并约 68 个低价值测试。

### 5.2 分阶段实施

| 阶段 | 内容 | 改动量 | 收益 | 状态 |
|------|------|--------|------|------|
| **Phase 1** | A1: 减少 epochs + A4: 精简鸡肋测试 + 附带修复 | 中 | 降低单测耗时 30-40% | ✅ 完成 |
| **Phase 1.5** | A5: 第二轮鸡肋测试精简（删除~15个+精简~6个） | 中 | 额外 ~23s | ✅ 完成 |
| **Phase 2** | A6: 第三轮鸡肋测试精简（删除~32个+合并~10个） | 中 | 额外 ~6.5s | ✅ 完成 |
| **Phase 3** | A3: fixture 优化 + pytest.ini | 小 | 额外 5-10% | 🔜 待执行 |

---

## 6. 实施计划（任务清单）

### Phase 1（已完成）
- [x] 1.1 降低高 epochs 测试的训练轮数（按 3.1 节表格，3 个文件）
- [ ] 1.2 将流程验证类测试改为 mock 训练（按 3.2 节，实际已 mock，跳过）
- [x] 1.3 删除签名检查测试（`TestEvaluatorSignatureChanges`，3 个用例）
- [x] 1.4 删除空壳类（`TestWinnerPathOutputOneHot`、`TestWinnerPathInvalidPosition`）
- [x] 1.5 删除重复错误处理测试（`TestInvalidInputs`，2 个用例）
- [x] 1.6 删除伪集成测试（`test_fc_v1_trainer_integration.py`）
- [x] 1.7 修复 `iterative_train.py` subprocess 参数类型（`int` → `str`）
- [x] 1.8 全量测试验证：1721 通过，0 失败，136.85s

### Phase 1.5（已完成）
- [x] 1.9 删除 kl_div 迁移验证测试（7 个，`test_kl_div_migration.py`）
- [x] 1.10 删除并行评测配置赋值测试（3 个，`test_performance_level2_parallel.py`）
- [x] 1.11 删除 worker pickle 测试（1 个，同上）
- [x] 1.12 合并 serial 测试 + 删除 serial 错误隔离（2 个，同上）
- [x] 1.13 删除可视化弱断言测试（2 个，`test_sample_visualizer.py`）
- [x] 1.14 精简可视化数据量（3 个，同上）
- [x] 1.15 合并参数验证为 parametrize（3 个，同上）
- [x] 1.16 精简 CLI smoke 样本数（1 个，`test_sample_generator.py`）
- [x] 1.17 精简对战局数 + 删除重复测试（`test_battle.py`）
- [x] 1.18 合并 app battle temp_config 测试（`test_battle.py`）
- [x] 1.19 全量测试验证：1703 通过，0 失败，127.75s

### Phase 2（已完成）
- [x] 1.20 删除整文件 `test_exceptions.py`（4 个）
- [x] 1.21 删除整文件 `test_iterative_train_dynamic_regex.py`（10 个）
- [x] 1.22 删除废弃参数测试（--self-play/--merge-output/--cycle，3 个）
- [x] 1.23 删除已删常量验证 + 源码扫描测试（2 个）
- [x] 1.24 删除重复 e2e `test_lr_range_e2e.py`（2 个 + 整文件）
- [x] 1.25 删除重复 registry 委托测试 + is_over 方法名检查（2 个）
- [x] 1.26 删除弱断言测试：replay_browser（3）+ replay_viewer（3）+ heatmap（1）+ loop_always_continues（1）
- [x] 1.27 合并 GUI 琐碎测试：ai_battle（10→3）+ game_board（3→1）+ mode_selector（5→2）
- [x] 1.28 全量测试验证：1659 通过，0 失败，122.47s

### Phase 3（待执行）
- [ ] 3.1 创建 `pytest.ini`
- [ ] 2.2 升级 fixture 作用域（function → module/session）
- [ ] 2.3 全量测试最终验证

---

## 7. 风险与注意事项

1. **收敛性测试保留**：`test_can_overfit_single_sample` 和 `test_can_overfit_batch` 保持 epochs=400/200，不做 mock
2. **epochs 减少后断言失效**：部分测试可能依赖 epochs=N 时的特定行为（如 loss 下降幅度），需逐个验证（已验证无问题）
3. **不引入并行执行**：并行测试可能引入竞态条件、资源冲突和非确定性失败，风险大于收益，本方案不采用
4. **subprocess 参数类型**：`iterative_train.py` 中 `--temp-moves` 误传 int 而非 str，导致 5 个测试因 `" ".join(call_args)` 失败，已修复

---

## 8. 附录

### 8.1 测试文件完整清单见 [test_inventory.md](test_inventory.md)

### 8.2 当前需降低 epochs 的完整列表

| 文件 | 当前 epochs | 改为 | 严重程度 |
|------|------------|------|----------|
| `test_base_trainer_train_auto.py` | 1000, 50×4 | mock | 高 |
| `test_loss_logger.py` | 1000×2 | 10 | 高 |
| `test_base_trainer_val.py` | 200×2 | 10 | 高 |
| `test_fc_v1_trainer.py` | 200 | 5 | 中 |
| `test_mini_batch_training.py` | 200 | 5 | 中 |
| `test_fc_v1_softmax_input.py` | 50 | 5 | 低 |
| `test_graceful_shutdown.py` | 10×4 | 2 | 低 |

---

**文档结束**
