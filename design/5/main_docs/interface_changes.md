# 接口变更详细设计

> **行号说明**：本文档中所有行号基于变更前代码版本，实施时以实际代码为准。

---

## 1. ai/evaluator.py

### 1.1 删除项

| 项 | 位置 | 说明 |
|----|------|------|
| `from ai.line_tracer import LineTracer` | 第 21 行 | import 语句 |
| `BLEND_ALPHA = 1.0` | 第 55 行 | 类常量 |
| `decay_rate` 参数 | 第 58 行构造器 | 构造器参数 |
| `self.decay_rate = decay_rate` | 第 78 行 | 属性赋值 |
| `self._line_tracer = LineTracer(...)` | 第 89-94 行 | 对象创建 |
| `line_bonuses` 计算块 | 第 153-160 行 | `_compute_all_rewards` 中的五连追溯逻辑 |
| `if step_idx in line_bonuses:` 块 | 第 188-199 行 | 奖励叠加逻辑 |
| 构造器注释中关于 LineTracer 的描述 | 第 63 行 | 文档注释 |
| `_evaluate_files` 中 `executor.submit` 的 `self.decay_rate` 参数 | 第 305-315 行 | 并行调用时传递给 `_evaluate_record_worker` 的第 4 个参数 |
| `evaluate_record` 注释 | 第 117 行 | "融合棋型评估和五连追溯" → "棋型评估" |
| `_compute_all_rewards` docstring | 第 135-140 行 | "双向棋型检测法 + 五连追溯" 及 "叠加五连追溯奖励" → 仅棋型评估描述 |
| `_compute_all_rewards` 的 `end_board` 参数 | 第 133 行 | 简化后不再使用，移除 |
| `_compute_all_rewards` 的 `winner` 参数 | 第 132 行 | 简化后不再使用（仅 line_bonuses 计算块使用它，已一并删除；winner 的有效性判断在 `evaluate_record` 中完成） |
| `evaluate_record` 中 `winner` 传参 | 第 118 行 | 不再传给 `_compute_all_rewards`，仅在 `evaluate_record` 内部用于有效性判断 |
| `evaluate_record` 中 `end_board` 提取 | 第 112 行 | 不再读取 end_board，不再传给 `_compute_all_rewards` |

### 1.2 Evaluator.__init__ 简化后签名

```
Evaluator(ai_data_dir: str = None, train_data_dir: str = None, max_workers: int = None)
```

内部仅创建 `_reward_config` 和 `_reward_calculator`，不再创建 `_line_tracer`。

### 1.3 _compute_all_rewards 简化后签名

```
_compute_all_rewards(history: List[Dict]) -> List[Dict]
```

移除 `winner` 和 `end_board` 参数（简化后均不再使用：`winner` 仅在 `line_bonuses` 计算块中使用，已一并删除；`winner` 的有效性判断在 `evaluate_record` 中完成，不在此方法内）。`evaluate_record` 保留 `winner` 用于自身的有效性判断（`winner == 0` 时抛 ValueError），但不再传递给 `_compute_all_rewards`。

### 1.4 _evaluate_record_worker 简化后签名

```
_evaluate_record_worker(record_file: str, ai_data_dir: str, train_data_dir: str) -> str
```

删除 `decay_rate` 参数。内部创建 Evaluator 时不再传 decay_rate。

**注意**：`_evaluate_files` 中的 `executor.submit(_evaluate_record_worker, ..., self.decay_rate)` 调用（约第 305-315 行）必须去掉 `self.decay_rate` 参数，否则运行时 TypeError。

---

## 2. ai/base_trainer.py

### 2.1 删除项

| 项 | 位置 | 说明 |
|----|------|------|
| `from ai.self_play_cycle import SelfPlayCycle` | 第 18 行 | import 语句 |
| `train_cycle()` 方法 | 第 622-661 行 | 整个方法 |

### 2.2 无新增接口

删除 train_cycle 后，BaseTrainer 对外暴露的训练入口仅保留：
- `train_auto()` — 自动选择训练方法
- `train_batch()` — 全量更新
- `train_mini_batch()` — mini-batch 更新

---

## 3. ai/train_main.py

### 3.1 删除项

| 项 | 位置 | 说明 |
|----|------|------|
| `from ai.runners import run_cycle` | 第 24 行 | import |
| `--cycle` 参数 | 第 58-62 行 | argparse 参数 |
| `--rounds` 参数 | 第 64-68 行 | argparse 参数 |
| `--games` 参数 | 第 70-76 行 | argparse 参数 |
| `--rate` 参数 | 第 85-90 行 | argparse 参数 |
| 温度采样参数组 | 第 161-192 行 | `--no-temp-sampling`、`--temperature`、`--temp-moves`、`--temp-side`、`--temp-seed` |
| 模式互斥逻辑中的 cycle 判断 | 第 197-206 行 | `if args.cycle` 分支 |
| `elif args.cycle:` 路由 | 第 228-229 行 | `run_cycle(args, trainer)` 调用 |
| CLI 示例中 cycle 相关用法 | 第 32-47 行 | epilog 文本 |

### 3.2 简化后模式路由

```
args = parser.parse_args()

# 验证：必须指定 --train-from-dir
if not args.train_from_dir:
    parser.error("请指定 --train-from-dir 参数")

# 创建训练器
trainer = get_trainer(args.model, fresh=args.fresh, **scheduler_kwargs)

# 执行训练
run_train_from_dir(args, trainer)
```

### 3.3 保留的参数

`--model`、`--fresh`、`--train-from-dir`、`--epochs`、`--batch-size`、`--shuffle`、`--lr-scheduler`（及子参数）、`--val-ratio`

---

## 4. ai/runners.py

### 4.1 删除项

| 项 | 位置 | 说明 |
|----|------|------|
| `run_cycle()` 函数 | 第 173-254 行 | 整个函数 |
| 模块注释中 `run_cycle` 引用 | 第 8 行 | docstring |
| import 中 `run_cycle` 引用 | 不存在 | run_cycle 定义在 runners.py 本身 |

### 4.2 保留项

`run_train_from_dir()`、`discover_sample_dirs()`、`_create_temp_sample_dir()`

---

## 5. ai/self_play_generator.py

### 5.1 删除项

| 项 | 位置 | 说明 |
|----|------|------|
| `rate` 参数 | 第 24 行 | `run_self_play_and_evaluate` 函数参数 |
| `rate` 参数 | 第 198 行 | `SelfPlayGenerator.generate` 方法参数 |
| `decay_rate=rate` | 第 105 行 | Evaluator 创建时的 decay_rate 传递 |
| `rate=rate` | 第 229 行 | 委托调用中的 rate 传递 |
| 注释中 SelfPlayCycle 的引用 | 第 4-5 行、第 34 行、第 180 行 | 文本描述 |

### 5.2 run_self_play_and_evaluate 简化后签名

```
run_self_play_and_evaluate(model, games, ai_data_dir, train_data_dir, max_workers, name_prefix, project_root, temp_config) -> dict
```

内部创建 Evaluator 时不传 decay_rate：
```
evaluator = Evaluator(ai_data_dir=ai_data_dir, train_data_dir=train_data_dir, max_workers=max_workers)
```

### 5.3 SelfPlayGenerator.generate 简化后签名

```
generate(games: int = 100, project_root: str = None, temp_config: Optional["TempConfig"] = None) -> dict
```

委托调用时不传 rate：
```
result = run_self_play_and_evaluate(model=self.model, games=games, max_workers=self.max_workers, name_prefix="self_play", project_root=project_root, temp_config=temp_config)
```

---

## 6. 测试文件修改

详见 test_design.md。

---

## 7. 文档/注释清理

以下位置的注释或文档引用了已删除的概念，需更新：

| 文件 | 位置 | 说明 |
|------|------|------|
| `ai/self_play_generator.py` | 第 4-5 行 | "从 SelfPlayCycle 中提取" → 删除此描述 |
| `ai/self_play_generator.py` | 第 34 行 | "供 SelfPlayGenerator 和 SelfPlayCycle 复用" → "供 SelfPlayGenerator 使用" |
| `ai/self_play_generator.py` | 第 180 行 | "与 SelfPlayCycle 的区别" → 删除此段 |
| `ai/evaluator.py` | 第 60-63 行 | 构造器注释中 LineTracer 引用 → 删除 |
| `ai/evaluator.py` | 第 80 行 | 注释 "LineTracer / RewardCalculator 均从此获取" → "RewardCalculator 从此获取" |
| `ai/evaluator.py` | 第 117 行 | 注释 "融合棋型评估和五连追溯" → "棋型评估" |
| `ai/evaluator.py` | 第 135-140 行 | `_compute_all_rewards` docstring 中 "双向棋型检测法 + 五连追溯" 及 "叠加五连追溯奖励" → 仅棋型评估描述 |
| `ai/base_trainer.py` | 第 36 行 | BaseTrainer 类 docstring "提供通用的训练流程：单轮次训练、批量训练、训练周期。" → "提供通用的训练流程：单轮次训练、批量训练。"（去掉"训练周期"及末尾顿号） |
| `ai/runners.py` | 第 8 行 | 模块 docstring `run_cycle` 引用 → 删除 |
| `ai/train_main.py` | 第 6-10 行 | 模块级 docstring cycle 用法示例 → 仅保留 --train-from-dir 示例 |
| `tests/ai/test_bugfix_dense_log_and_val_ratio.py` | 第 47 行 | "self_play_cycle.py" 引用 → 更新描述（self_play_cycle.py 已删除，该 bug 的根因描述需修正）