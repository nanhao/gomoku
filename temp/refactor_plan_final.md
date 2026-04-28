# ai 包重构计划书（最终版）

> 评审方式：两位资深架构师独立拟定计划并交叉辩论，裁判综合双方观点、纠正事实偏差后输出本计划。

## 一、评审结论

两份计划对核心问题的识别高度一致：模型注册表分散、Evaluator 委托层冗余、CNN 模型重复代码、BaseTrainer 训练循环重复、train_main.py 过于臃肿。差异主要体现在抽象粒度和执行策略上。

**关键裁定**：
- 训练循环**不强行统一为单一模板**，改为提取细粒度组合子，保持三个训练方法独立，避免 `if mini_batch:` 分支爆炸。
- CNN 模型抽取基类，但**不做 CNNTrainerBase**（仅 9 行重复，YAGNI）。
- 模型注册表采用**集中式懒加载**（字符串元数据 + `importlib`），而非自注册——显式优于隐式，更利于新成员理解。
- 死代码（`_prepare_tensors` 等）**被 5 个测试文件直接调用**，必须先迁移测试再删除，不能一步清理。
- AIGameManager 内战方法合并（选手 B 独有提议）纳入计划，低风险高收益。

## 二、重构目标

1. **消除高重复代码**：注册表、CNN 模型模板代码、AIGameManager 内战循环统一收口。
2. **精简 API 边界**：Evaluator 只保留评价流程入口，移除无意义委托。
3. **提升可测试性**：train_main.py 业务逻辑与 CLI 解耦；死代码迁移到测试可直接调用的公共 API。

## 三、重构任务清单

### 任务 1：AIGameManager 内战方法合并
- **涉及文件**：`ai/game_manager.py`
- **具体做法**：提取 `_play_single_game(engine_black, engine_white, recorder, game_num)` 封装单局游戏循环（创建 controller → 落子循环 → 记录 → 保存结果）。`start_ai_vs_ai` 和 `start_ai_vs_ai_random` 仅保留外层执方分配和结果聚合。
- **预期收益**：消除约 80 行重复；游戏循环修改只需改一处。
- **风险与回退**：低风险。回退方式为保留原方法，新增内部方法逐步替换。

### 任务 2：CNN 模型提取共享基类
- **涉及文件**：新建 `ai/models/cnn_base.py`；修改 `ai/models/cnn_v1/model.py`、`ai/models/cnn_v2/model.py`
- **具体做法**：创建 `CNNBaseModel(BaseModel)`，将 `_vector_to_tensor`、`predict`、`save`、`load`、`__init__` 上移至基类。子类仅定义 `network_class`、`DEFAULT_MODEL_PATH`、`name`。`_convert_inputs` 仍在各自 Trainer 中保持独立（仅 9 行，不值得新增 TrainerBase）。
- **预期收益**：每个 CNN 模型文件减少约 80 行；新增 CNN 变体时只需定义网络结构。
- **风险与回退**：低风险。纯结构重组，不改变运行时行为，用现有测试回归验证。

### 任务 3：Evaluator 两阶段瘦身
- **涉及文件**：`ai/evaluator.py`、`ai/shape_detector.py`、相关测试
- **具体做法**：
  - **阶段 1（当前版本）**：删除所有下划线私有委托方法（零外部影响）；保留公开委托方法（`detect_shapes`、`shape_to_reward`、`find_five_in_a_row`、`trace_winning_line`、`blend_rewards`），内部加 `warnings.warn(DeprecationWarning, stacklevel=2)`；将 `_compute_all_rewards` 内对 `blend_rewards` 的调用改为直接 `self._line_tracer.blend_rewards()`；删除 `DIRECTIONS`/`DIRECTION_NAMES`/`SOFTMAX_SCALE` 重复常量，统一引用 `shape_detector`/`probability_utils`。
  - **阶段 2（下一版本）**：迁移所有调用方后彻底删除公开委托方法。
- **预期收益**：Evaluator 从 414 行降至约 180 行；职责边界清晰。
- **风险与回退**：中低风险。阶段 1 不破坏公开 API；阶段 2 需先全局 `grep` 确认无外部调用。

### 任务 4：统一模型注册表（懒加载）
- **涉及文件**：新建 `ai/registry.py`；修改 `ai/engine.py`、`ai/battle.py`、`ai/train_main.py`
- **具体做法**：新建 `ai/registry.py`，定义 `MODEL_REGISTRY: dict[str, tuple[str, str | None]]`（name → `(model_module_path, trainer_module_path)`），仅存字符串元数据。提供 `get_model(name)`、`get_trainer(name, **kwargs)`、`list_models()`，内部使用 `importlib.import_module` 按需加载。`create_engine`、`battle.py:load_model`、`train_main.py:get_trainer` 统一调用 registry。容错策略：`get_model` 未知抛 `ValueError`；`create_engine` 在调用前捕获并回退 `RandomModel`。
- **预期收益**：新增模型只需在 `registry.py` 加一行；消除三处映射不一致风险；保留原懒加载启动性能。
- **风险与回退**：低风险。若 registry 导入失败，可回退到各文件内联映射。

### 任务 5：BaseTrainer 提取训练组合子 & 死代码迁移
- **涉及文件**：`ai/base_trainer.py`、相关测试
- **具体做法**：
  1. **提取组合子**：在 `BaseTrainer` 中提取 `_forward_backward_step(inputs, targets, masks)` 封装（zero_grad → forward → `_masked_loss` → backward → grad_norm → step），供 `train_single`、`train_batch`、`train_mini_batch` 复用。`train_mini_batch` 保留其特有的 GPU 张量累加器、密集日志批量同步、global_step 逻辑，不强行注入通用循环。
  2. **死代码迁移**：`_prepare_tensors`、`_load_sample_data`、`_compute_loss` 被 5 个测试文件直接调用，不能删除。先将它们迁移为 `SampleLoader` 的类方法或独立测试工具函数，更新测试引用后，再从 `BaseTrainer` 中删除。
- **预期收益**：消除约 120 行重复；训练核心逻辑修改只需改一处；测试与死代码解耦。
- **风险与回退**：中风险。训练循环是核心路径，每步均需全量回归测试（`tests/ai/` 37+ 文件）。回退策略为逐方法替换，保留旧方法加 `DeprecationWarning` 过渡。

### 任务 6：train_main.py 职责拆分
- **涉及文件**：`ai/train_main.py`，新建 `ai/runners.py`
- **具体做法**：将四个模式的业务逻辑提取为 `ai/runners.py` 中的独立顶层函数：`run_sample_dir`、`run_self_play`、`run_train_from_dir`、`run_cycle`。每个函数接收已解析的参数对象或具体数据结构，负责流程编排和结果输出。`train_main.py` 的 `main()` 仅负责 `argparse` 解析、参数校验和函数路由。`get_trainer` 迁移到 `registry.py`。
- **预期收益**：`main()` 从 386 行降至约 50 行；业务逻辑可独立单元测试。
- **风险与回退**：低风险。CLI 入口变更不影响库级调用。

## 四、执行顺序

```
阶段 1（可并行，低风险快速落地）:
  ├── 任务 1：AIGameManager 内战合并
  └── 任务 2：CNN 模型基类抽取

阶段 2（可并行，中等风险）:
  ├── 任务 3 阶段 1：Evaluator 私有委托删除 + 公开委托加 Warning
  └── 任务 4：统一模型注册表（懒加载）

阶段 3（串行，核心路径）:
  └── 任务 5：BaseTrainer 组合子提取 + 死代码迁移
        （先迁移测试引用，再提取组合子，最后删除旧方法）

阶段 4（串行，依赖阶段 2、3）:
  ├── 任务 3 阶段 2：Evaluator 公开委托彻底删除
  └── 任务 6：train_main.py 职责拆分
```

每阶段完成后必须运行 `tests/ai/` 全量测试，确保零回归。

## 五、风险管控

| 风险项 | 等级 | 应对措施 |
|---|---|---|
| 训练循环改动引入 loss 曲线漂移 | 高 | 每步替换后对比相同 seed 下的 loss 曲线；保留旧方法过渡 |
| Evaluator 公开委托删除破坏外部脚本 | 中 | 阶段 1 加 `DeprecationWarning`；阶段 2 前全局 grep 确认无调用方 |
| 死代码直接删除导致测试失败 | 中 | 先迁移测试引用到 `SampleLoader`，确认全绿后再删除 |
| 注册表集中导入导致启动变慢 | 低 | 采用字符串元数据 + `importlib` 懒加载，首屏不触发模型导入 |
| CNN 基类引入后子类覆盖行为变化 | 低 | 纯结构移动，无逻辑变更；用现有模型加载测试回归验证 |
