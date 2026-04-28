# Code Review 报告 — 重复代码与冗余代码

审查日期：2026-04-28

---

## P0: 严重重复（需立即修复）

### 1. CNNV1Trainer._convert_inputs 与 CNNV2Trainer._convert_inputs 完全相同

- **位置**: `ai/models/cnn_v1/trainer.py#L45-L71` 与 `ai/models/cnn_v2/trainer.py#L45-L71`
- **问题**: 两个方法逐字一致，都是循环调用 `board_vector_to_cnn_tensor` 再 `torch.cat`。28行完全重复。
- **建议**: 将 `_convert_inputs` 的 CNN 实现提取到 `BaseTrainer` 的公共方法 `convert_cnn_inputs(inputs)` 中，两个 Trainer 直接复用。

### 2. BaseTrainer.train_single 与 train_batch 训练循环高度重复

- **位置**: `ai/base_trainer.py#L212-L272` 与 `ai/base_trainer.py#L274-L337`
- **问题**: 两个方法的核心循环体几乎完全相同：
  ```python
  # 相同结构：setup -> for epoch -> _forward_backward_step -> logging -> _step_lr_scheduler -> KeyboardInterrupt -> _finalize_training
  ```
  唯一差异是数据加载方式（单文件 vs 合并多文件）和参数校验。约 40 行重复代码。
- **建议**: 提取 `_run_training_loop(inputs, targets, masks, epochs, loss_logger, verbose)` 私有方法，`train_single` 和 `train_batch` 只负责数据加载后调用它。

### 3. SelfPlayCycle.run 与 SelfPlayGenerator.generate 大量逻辑重复

- **位置**: `ai/self_play_cycle.py#L35-L219` 与 `ai/self_play_generator.py#L45-L170`
- **问题**: 两者共享以下步骤（约 60 行重复）：
  - 创建 AIEngine → 创建 DataRecorder → 调用 `AIGameManager.start_ai_vs_ai` → 统计黑白胜平 → 创建 Evaluator → `evaluate_round` → 收集 sample_dirs → 打印错误信息
  - KeyboardInterrupt 处理逻辑也高度相似
- **建议**: 提取公共的 `_run_self_play(model, games, rate, ai_data_dir, train_data_dir, max_workers)` 函数，两者都委托给它。`SelfPlayCycle` 在此基础上加训练步骤。

---

## P1: 中度重复（建议修复）

### 4. SampleLoader.load 与 SampleLoader.load_merge_file 完全相同

- **位置**: `ai/sample_loader.py#L95-L109` 与 `ai/sample_loader.py#L131-L147`
- **问题**: 两个方法的实现完全一致（`open(path, 'r') + json.load(f)`），注释也承认"与 load 方法行为一致"。
- **建议**: `load_merge_file` 直接调用 `self.load(merge_path)`，或删除 `load_merge_file` 统一用 `load`。

### 5. Evaluator._evaluate_files 串行回退逻辑重复

- **位置**: `ai/evaluator.py#L257-L341`
- **问题**: 串行评价逻辑出现两次：
  - 第 281-288 行：max_workers=1 时的串行循环
  - 第 333-339 行：broken_pool 回退时的串行循环
  两段代码几乎完全一致（for record_file → try → evaluate_record → except → errors.append）。
- **建议**: 提取 `_evaluate_serial(record_files, sample_files, errors)` 私有方法。

### 6. runners.py 中训练方法选择逻辑重复

- **位置**: `ai/runners.py#L241-L253` 与 `ai/self_play_cycle.py#L149-L171`
- **问题**: `if batch_size: trainer.train_mini_batch(...) else: trainer.train_batch(...)` 这段分发逻辑在多处重复。
- **建议**: 在 `BaseTrainer` 上新增 `train_auto(sample_dirs, batch_size=None, **kwargs)` 方法，内部做分发。

### 7. CNNV1Network.forward 与 CNNV2Network.forward 结构完全一致

- **位置**: `ai/models/cnn_v1/model.py#L68-L103` 与 `ai/models/cnn_v2/model.py#L79-L114`
- **问题**: 前向传播逻辑完全相同（3层 Conv+BN+ReLU → flatten → FC1+ReLU → FC_output+Softmax），仅参数维度不同。
- **建议**: 提取 `BaseCNNNetwork` 基类，参数化通道数列表 `[conv_channels]` 和 FC1 输出维度，`forward` 在基类实现一次。

### 8. Evaluator 常量与 RewardConfig 常量重复

- **位置**: `ai/evaluator.py#L55-L56` 与 `ai/reward_calculator.py#L19-L20`
- **问题**: `Evaluator.MAX_REWARD = 1000000` 与 `RewardConfig.max_reward = 1_000_000` 存储了相同的值，且 Evaluator 又把它传给了 LineTracer 和 RewardCalculator。如果有人改了 RewardConfig 忘记改 Evaluator，就会不一致。
- **建议**: Evaluator 不再自定义 `MAX_REWARD`/`MIN_REWARD`，直接使用 `RewardConfig` 的默认值或从 `RewardCalculator` 获取。

---

## P2: 轻度重复/冗余（可选修复）

### 9. 冗余函数: battle.py 中 load_model 只是 get_model 的薄包装

- **位置**: `ai/battle.py#L35-L50`
- **问题**: `load_model(ai_name)` 只是 `return get_model(ai_name)`，无任何额外逻辑。函数体只有一行，且已有注释说明委托给 registry。
- **建议**: 在 `battle.py` 中直接调用 `get_model`，删除 `load_model` 函数。

### 10. 冗余函数: SampleLoader.load_merge_file

- **位置**: 同问题 4，虽标记为语义区分但实现完全相同。
- **建议**: 至少 `load_merge_file` 应该调用 `self.load()` 而非重新实现。

### 11. winner 文本转换逻辑散落多处

- **位置**:
  - `app/gui_helper.py#L153-L168` → "黑方获胜/白方获胜/平局"
  - `cli/main.py#L66-L83` → "黑方胜/白方胜/平局"
  - `game/controller.py#L72-L75,L98` → 硬编码"黑方/白方"
  - `gui/display_updater.py#L67-L79` → 也有隐含的 winner→text 逻辑
- **问题**: winner 整数到文本的映射在 4 个地方以不同形式存在，用词也不统一（"获胜" vs "胜"）。
- **建议**: 在 `game` 包中定义 `get_winner_text(winner: int) -> str`，所有上层统一调用。

### 12. 动物表情常量多处重复定义

- **位置**:
  - `gui/main.py#L41-L42` (`GomokuGUI.ANIMAL_BLACK/ANIMAL_WHITE`)
  - `gui/display_updater.py#L14-L15` (`GameDisplayUpdater.ANIMAL_BLACK/ANIMAL_WHITE`)
  - `app/gui_helper.py#L15-L16` (`GUIHelper.ANIMAL_BLACK/ANIMAL_WHITE`)
- **问题**: `"🐻"` 和 `"🐰"` 在 3 个类中各定义了一次。
- **建议**: 集中定义在 `gui/components/constants.py` 中，其他类引用即可。

### 13. DataRecorder 中 game_id 校验逻辑重复

- **位置**: `ai/data_recorder.py#L94-L98` 与 `ai/data_recorder.py#L115-L119`
- **问题**: 两处都有完全相同的 `game_id != self.current_game_id` 检查 + warning + 修正逻辑。
- **建议**: 提取 `_ensure_game_id(game_id)` 私有方法。

### 14. discover_sample_dirs (runners.py) 与 discover_all_data (sample_loader.py) 功能重叠

- **位置**: `ai/runners.py#L25-L64` 与 `ai/sample_loader.py#L35-L79`
- **问题**: 两者都扫描目录下含 `sample.json` 的子目录。`discover_sample_dirs` 只找直接子目录，`discover_all_data` 还找 merge 文件，但核心逻辑重叠。
- **建议**: `discover_sample_dirs` 应复用 `discover_all_data` 的结果，或直接使用 `SampleLoader.discover_and_load`。

### 15. 多处 `sys.path.insert` 样板代码

- **位置**:
  - `cli/main.py#L7`
  - `ai/battle.py#L19-L26`
  - `gui/main.py#L13`
  - `gui/components/ai_battle.py#L11`
- **问题**: 4 处都有 `sys.path.insert(0, ...)` 的项目根目录路径设置逻辑，且 `battle.py` 还做了额外的 `sys.path.remove` 处理。
- **建议**: 创建 `bootstrap.py` 或在 `__init__.py` 中统一处理路径设置。

---

## 其他发现（非重复但值得注意）

### 16. Evaluator 的 `DECAY_RATE` 类属性与构造函数参数 `decay_rate` 冲突

- **位置**: `ai/evaluator.py#L57,L77`
- **问题**: 类属性 `DECAY_RATE = 0.9` 和构造函数 `decay_rate: float = None` 功能重叠。`__init__` 中 `self.decay_rate = decay_rate if decay_rate is not None else self.DECAY_RATE`，这个类属性只在默认值场景使用。
- **建议**: 去掉 `DECAY_RATE` 类属性，直接在构造函数默认参数中设置 `decay_rate: float = 0.9`。

### 17. SelfPlayCycle 中 `project_root` 参数的冗余路径构建

- **位置**: `ai/self_play_cycle.py#L60-L68`
- **问题**: 当 `project_root=None` 时，方法自己拼了 `ai_data_dir` 和 `train_data_dir`，但这两个常量已在 `ai/constants.py` 中定义。`SelfPlayGenerator` 则直接用了常量。
- **建议**: `SelfPlayCycle` 也应优先使用常量，`project_root` 仅在测试隔离场景使用。

---

## 统计汇总

| 优先级 | 问题数 | 预估可消除重复行数 |
|--------|--------|-------------------|
| P0 严重 | 3 | ~130 行 |
| P1 中度 | 5 | ~100 行 |
| P2 轻度 | 7 | ~50 行 |
| **合计** | **15** | **~280 行** |
