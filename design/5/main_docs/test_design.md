# 测试设计

---

## 1. 整文件删除的测试

| 文件 | 说明 |
|------|------|
| `tests/ai/test_line_tracer.py` | LineTracer 的全部测试已无测试对象 |
| `tests/ai/test_self_play_cycle.py` | SelfPlayCycle 的全部测试已无测试对象 |
| `tests/ai/primary/test_evaluator_winning_line.py` | 五连追溯奖励的全部测试已无测试对象 |

---

## 2. 需适配更新的测试文件

### 2.1 tests/ai/test_evaluator.py

**无修改**。该文件测试 Evaluator 的通用功能（文件路径映射、串行/并行评价），不涉及 LineTracer 或 decay_rate。

### 2.2 tests/ai/test_train_main.py

**需要大幅修改**。当前多数测试用例围绕 `--cycle` 模式。

| 用例 | 操作 |
|------|------|
| `test_cycle_mode_parses_rounds_and_games`（第 23 行） | 删除（`--cycle` 不存在） |
| `test_cycle_mode_with_batch_size`（第 38 行） | 删除（`--cycle` 不存在） |
| `test_no_mode_fails`（第 51 行） | 修改：不指定 `--train-from-dir` 报错 |
| `test_fresh_flag`（第 57 行） | 删除（依赖 `--cycle`） |
| `test_lr_scheduler_args`（第 70 行） | 删除（依赖 `--cycle`） |
| `test_default_model_is_fc_v1`（第 88 行） | 删除（依赖 `--cycle`） |
| `test_train_from_dir_json_file`（第 104 行） | 保留 |
| `test_train_from_dir_with_merge`（第 129 行） | 保留 |
| `test_train_from_dir_empty_raises`（第 161 行） | 保留 |
| `test_train_from_dir_with_mini_batch`（第 175 行） | 保留 |
| `test_train_from_dir_mini_batch_enables_dense_log`（第 201 行） | 保留 |
| `test_train_from_dir_no_batch_no_dense_log`（第 226 行） | 保留 |
| `test_cycle_mode_accepts_temp_flags`（第 262 行） | 删除（温度采样参数不存在） |
| `test_cycle_mode_no_temp_sampling_flag`（第 293 行） | 删除 |
| `test_non_cycle_mode_ignores_temp_flags`（第 313 行） | 修改：改为测试 `--train-from-dir` 模式 |
| `TestTrainMainArgParsing` 整个类（第 20 行） | 删除（全部用例依赖 `--cycle`） |
| `TestTrainMainTempSamplingFlags` 整个类（第 259 行） | 删除（全部用例依赖温度采样） |

**修改细节**：
- `test_no_mode_fails`：当前验证 "不指定任何模式报错" → 改为验证 "不指定 --train-from-dir 报错"
- `test_non_cycle_mode_ignores_temp_flags`：温度参数不再存在，改为验证 --train-from-dir 正常工作（无需处理温度参数）

### 2.3 tests/ai/unit/test_runners.py

**需要修改**。

| 用例 | 操作 |
|------|------|
| import 中 `run_cycle` | 删除 |
| `_make_args()` 默认值中的 cycle/rounds/games/rate/温度参数 | 删除 |
| `TestRunCycle` 整个类 | 删除（含 4 个用例） |
| `TestRunTrainFromDir` | 保留（6 个用例不变） |
| `TestCreateTempSampleDir` | 保留（2 个用例不变） |

### 2.4 tests/ai/test_fc_v1_trainer.py

**需要修改**。

| 用例 | 操作 |
|------|------|
| `TestTrainCycleRateParameter.test_train_cycle_has_rate_parameter` | 删除（train_cycle 不存在） |
| `TestTrainCycleRateParameter.test_train_main_script_rate_default` | 删除（`--rate` 不存在） |

### 2.5 tests/ai/test_self_play_generator.py

**需要修改**。

| 用例 | 操作 |
|------|------|
| `TestSelfPlayGeneratorRate.test_rate_passed_to_evaluator` | 删除（rate 参数不存在） |
| `TestSelfPlayGeneratorEvaluator.test_generate_passes_rate_to_evaluator`（第 154-166 行） | 删除（整方法，因 rate 参数不存在） |
| `TestSelfPlayGeneratorEvaluator.test_generate_calls_evaluator_evaluate_round`（第 134-148 行） | 保留，断言不变 |
| `TestSelfPlayGeneratorEvaluator.test_generate_passes_max_workers_to_evaluator`（第 172 行） | 保留，Mock Evaluator 创建时不传 decay_rate |
| 其他用例 | 适配：Mock Evaluator 创建时不传 decay_rate；断言 `eval_call_kwargs['decay_rate']` 不再存在 |

**具体断言修改**：
- 第 166 行 `self.assertEqual(eval_call_kwargs['decay_rate'], 0.85)` → 随 `test_generate_passes_rate_to_evaluator` 方法一并删除
- 其他用例中若断言 Evaluator call_args 包含 decay_rate → 改为验证 Evaluator 创建时不含 decay_rate 关键字

### 2.6 tests/ai/test_bugfix_dense_log_and_val_ratio.py

**需要修改注释**。

| 项 | 操作 |
|----|------|
| 第 47 行注释 | "self_play_cycle.py" → "run_train_from_dir" 或删除对 self_play_cycle.py 的引用 |

### 2.7 tests/ai/primary/test_evaluator_shape_detection.py

**无修改**。已确认所有用例创建 Evaluator 时仅传 `ai_data_dir` 和 `train_data_dir`（第 36 行 `Evaluator(self.ai_data_dir, self.train_data_dir)`），不含 `decay_rate` 参数，签名变更后仍合法。

### 2.8 tests/ai/test_performance_level2_parallel.py

**需要修改**。其中 `_evaluate_record_worker` 调用传了 `decay_rate=0.9`，需去掉。具体 3 处：

| 用例 | 位置 | 操作 |
|------|------|------|
| `TestEvaluateRecordWorker.test_worker_returns_sample_path` | 第 109-110 行 | `_evaluate_record_worker(record_path, self.ai_data_dir, self.train_data_dir, 0.9)` → 去掉第 4 个参数 `0.9` |
| `TestEvaluateRecordWorker.test_worker_creates_valid_sample` | 第 121-122 行 | `_evaluate_record_worker(record_path, self.ai_data_dir, self.train_data_dir, 0.9)` → 去掉第 4 个参数 `0.9` |
| `TestEvaluateRecordWorker.test_worker_can_be_pickled_by_processpool` | 第 142-143 行 | `executor.submit(_evaluate_record_worker, record_path, self.ai_data_dir, self.train_data_dir, 0.9)` → 去掉第 4 个参数 `0.9` |

### 2.9 tests/ai/edge/test_evaluator_shape_detection_edge.py

**无修改**。该文件创建 Evaluator 时仅传 `ai_data_dir` 和 `train_data_dir`，不传 `decay_rate`，签名变更后仍合法。列为完整性确认。

---

## 3. 验收用例集

> 验收用例编号与 PRD 对齐：AC-1 至 AC-5。

### AC-1 验收：五连追溯奖励已完全移除（对应 PRD AC-1）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-1.1 | ai/line_tracer.py 文件不存在 | 文件系统检查 |
| AC-1.2 | Evaluator 类无 LineTracer 引用 | grep 搜索 ai/evaluator.py 无 "LineTracer" |
| AC-1.3 | Evaluator 类无 line_tracer import | grep 搜索 ai/evaluator.py 无 "line_tracer" |
| AC-1.4 | Evaluator 无 BLEND_ALPHA 常量 | grep 搜索 ai/evaluator.py 无 "BLEND_ALPHA" |
| AC-1.5 | _compute_all_rewards 无 line_bonuses 逻辑 | grep 搜索 ai/evaluator.py 无 "line_bonuses" |
| AC-1.6 | Evaluator 构造器无 decay_rate 参数 | grep 搜索 ai/evaluator.py 无 "decay_rate" |
| AC-1.7 | Evaluator 无 self.decay_rate 属性 | grep 搜索 ai/evaluator.py 无 "self.decay_rate" |
| AC-1.8 | _evaluate_files submit 调用无 decay_rate | grep 搜索 ai/evaluator.py 无 "self.decay_rate"（包括 submit 调用） |
| AC-1.9 | _compute_all_rewards 无 winner 和 end_board 参数 | inspect.signature 检查 _compute_all_rewards 签名仅含 history |
| AC-1.10 | evaluate_record 无 winner/end_board 传参给 _compute_all_rewards | 代码审查 evaluate_record 调用 |
| AC-1.11 | evaluate_record 保留 winner 用于自身有效性判断 | 代码审查 winner == 0 校验仍存在 |
| AC-1.12 | _evaluate_record_worker 签名不含 decay_rate | inspect.signature 检查仅含 3 个参数（record_file, ai_data_dir, train_data_dir） |
| AC-1.13 | Evaluator docstring 无五连追溯描述 | grep 搜索 ai/evaluator.py 无 "五连追溯" |
| AC-1.14 | Evaluator 正确生成 sample.json | 构造 record.json → evaluate_record → 验证 sample 结构含 input/output/mask |
| AC-1.15 | sample.json output 仅来自棋型评估 + masked_softmax | 构造已知棋盘（含活四） → 验证 output 中对应位置概率最高 |
| AC-1.16 | --rate CLI 参数不存在 | python ai/train_main.py --rate 0.9 → SystemExit |
| AC-1.17 | test_line_tracer.py 不存在 | 文件系统检查 |
| AC-1.18 | test_evaluator_winning_line.py 不存在 | 文件系统检查 |

### AC-2 验收：Cycle 模式已完全移除（对应 PRD AC-2）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-2.1 | ai/self_play_cycle.py 不存在 | 文件系统检查 |
| AC-2.2 | BaseTrainer 无 train_cycle 方法 | grep 搜索 ai/base_trainer.py 无 "train_cycle" |
| AC-2.3 | --cycle 不存在 | python ai/train_main.py --cycle → SystemExit |
| AC-2.4 | --rounds 不存在 | python ai/train_main.py --rounds 5 → SystemExit |
| AC-2.5 | --games 不存在 | python ai/train_main.py --games 10 → SystemExit |
| AC-2.6a | --no-temp-sampling 不存在 | python ai/train_main.py --no-temp-sampling → SystemExit |
| AC-2.6b | --temperature 不存在 | python ai/train_main.py --temperature 0.8 → SystemExit |
| AC-2.6c | --temp-moves 不存在 | python ai/train_main.py --temp-moves 10 → SystemExit |
| AC-2.6d | --temp-side 不存在 | python ai/train_main.py --temp-side both → SystemExit |
| AC-2.6e | --temp-seed 不存在 | python ai/train_main.py --temp-seed 42 → SystemExit |
| AC-2.7 | runners.py 无 run_cycle | grep 搜索 ai/runners.py 无 "run_cycle" |
| AC-2.8 | train_main.py 无 SelfPlayCycle 引用 | grep 搜索 ai/train_main.py 无 "SelfPlayCycle" |
| AC-2.9 | --train-from-dir 正常工作 | 构造样本目录 → run_train_from_dir → trainer.train_auto 被调用 |
| AC-2.10 | 不指定 --train-from-dir 报错 | python ai/train_main.py --model fc_v1 → SystemExit |
| AC-2.11 | test_self_play_cycle.py 不存在 | 文件系统检查 |

### AC-3 验收：SelfPlayGenerator 不受影响（对应 PRD AC-3）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-3.1 | SelfPlayGenerator.generate 无 rate 参数 | inspect.signature 检查 generate 方法 |
| AC-3.2 | run_self_play_and_evaluate 无 rate 参数 | inspect.signature 检查函数签名 |
| AC-3.3 | SelfPlayGenerator 正常生成数据 | Mock AIGameManager → generate → 验证返回字典结构 |
| AC-3.4 | create_samples.py 不受影响 | 现有测试全部通过 |

### AC-4 验收：相关测试已清理（对应 PRD AC-4）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-4.1 | tests/ai/test_line_tracer.py 不存在 | 文件系统检查 |
| AC-4.2 | tests/ai/test_self_play_cycle.py 不存在 | 文件系统检查 |
| AC-4.3 | tests/ai/primary/test_evaluator_winning_line.py 不存在 | 文件系统检查 |
| AC-4.4 | 所有剩余测试通过 | pytest tests/ — exit code 0 |
| AC-4.5 | Evaluator 相关测试已更新（不含 line_bonuses 逻辑） | 代码审查 |
| AC-4.6 | train_main 相关测试已更新（不含 cycle 模式） | 代码审查 |

### AC-5 验收：全局引用已清理（对应 PRD AC-5）

| 编号 | 用例描述 | 验证方式 |
|------|---------|---------|
| AC-5.1 | ai/ 目录无 LineTracer 引用 | grep -r "LineTracer" ai/ 无匹配 |
| AC-5.2 | ai/ 目录无 SelfPlayCycle 引用 | grep -r "SelfPlayCycle" ai/ 无匹配 |
| AC-5.3 | tests/ 目录无 LineTracer 引用 | grep -r "LineTracer" tests/ 无匹配 |
| AC-5.4 | tests/ 目录无 SelfPlayCycle 引用 | grep -r "SelfPlayCycle" tests/ 无匹配 |
| AC-5.5 | 项目中无 train_cycle 引用（ai/ 和 tests/ 目录） | grep -r "train_cycle" ai/ tests/ 无匹配 |
| AC-5.6 | ai/ 目录无 line_tracer import | grep -r "from ai.line_tracer" ai/ 无匹配 |
| AC-5.7 | ai/ 目录无 self_play_cycle import | grep -r "from ai.self_play_cycle" ai/ 无匹配 |
| AC-5.8 | Evaluator/self_play_generator/runners/train_main 中无 decay_rate | grep -r "decay_rate" ai/ 无匹配 |
| AC-5.9 | ai/evaluator.py 无 end_board | grep 搜索 ai/evaluator.py 无 "end_board" |
| AC-5.10 | ai/ 目录无五连追溯描述 | grep -r "五连追溯" ai/ 无匹配 |
| AC-5.11 | BaseTrainer docstring 无 "训练周期" | grep 搜索 ai/base_trainer.py 无 "训练周期" |
| AC-5.12 | runners.py docstring 无 run_cycle | grep 搜索 ai/runners.py 无 "run_cycle" |
| AC-5.13 | train_main.py docstring 无 cycle 用法 | grep 搜索 ai/train_main.py 无 "cycle"（除 argparse 参数注释外） |