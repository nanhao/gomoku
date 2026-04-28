# 训练数据 Train/Val 分割与 Val Loss 监控设计

## 1. 背景与动机

当前训练流程中，所有 self-play 生成的样本数据全部用于训练，没有留出验证集。
无法判断模型是否过拟合，也无法通过 val loss 监控训练质量。

本次新增 train/val 分割功能，在训练过程中持续评估 val loss，
与 train loss、梯度范数叠加在同一图表中，直观呈现过拟合趋势。

**决策记录**：
- 分割粒度：**样本级别**（非对局级别），因为模型输入是单个棋盘状态，非序列
- 分割位置：**SampleLoader 层**（职责清晰，不影响 Trainer 接口语义）
- val loss 评估频率：**epoch 级**（非 step 级），避免 mini-batch 模式下计算开销翻倍
  - `train_batch` 模式：每 LOSS_LOG_INTERVAL 个 epoch 评估一次（与 train loss 同步）
  - `train_mini_batch` 模式：每个 epoch 结束后评估一次

## 2. API 设计

### 2.1 SampleLoader.load_and_split()

**文件**: `ai/sample_loader.py`

```python
def load_and_split(
    self,
    sample_dirs: List[str],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[list, list]:
    """加载多个 sample.json 并按比例分割为训练集和验证集

    Args:
        sample_dirs: 包含 sample.json 的目录路径列表
        val_ratio: 验证集比例，默认 0.2（20%）
        seed: 随机种子，确保分割可复现，默认 42

    Returns:
        (train_data, val_data): 训练集和验证集的 sample 数据列表

    Raises:
        ValueError: val_ratio 不在 (0, 1) 范围内
        FileNotFoundError: 某个目录下不存在 sample.json
    """
```

### 2.2 LossLogger.log_val()

**文件**: `ai/loss_logger.py`

```python
def log_val(self, x_value: int, val_loss: float) -> None:
    """记录一条验证集 loss 日志

    与 train loss 日志分开记录，写入同一文件的不同条目。
    val_loss 条目使用 val_loss 字段标识，绘图时独立绘制。

    Args:
        x_value: x 轴值（epoch 编号或 step 编号，与当前日志格式一致）
        val_loss: 验证集 loss 值
    """
```

条目格式：
- 稀疏模式（train_batch）：`{"epoch": N, "val_loss": X}`
- 密集模式（train_mini_batch）：`{"step": N, "val_loss": X}`

### 2.3 LossLogger.plot() 修改

在现有双 Y 轴图基础上，叠加 val_loss 折线：
- val_loss 与 train loss 共享左 Y 轴（Loss）
- val_loss 使用不同颜色（绿色），标记为 "Val Loss"
- val_loss 数据点通常比 train loss 稀疏，自然呈现为较少点的折线

### 2.4 BaseTrainer 新增方法

**文件**: `ai/base_trainer.py`

```python
def _evaluate_val_loss(
    self,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    val_masks: torch.Tensor
) -> float:
    """在验证集上计算 loss（仅前向传播，不更新参数）

    Args:
        val_inputs: 验证集输入张量
        val_targets: 验证集目标张量
        val_masks: 验证集 mask 张量

    Returns:
        验证集 loss 值（标量）
    """
```

### 2.5 BaseTrainer 参数变更

以下方法新增 `val_ratio` 参数（默认 0.0，即不分割）：

| 方法 | 新增参数 | 说明 |
|------|----------|------|
| `train_batch()` | `val_ratio: float = 0.0` | 全量训练模式下的验证分割 |
| `train_mini_batch()` | `val_ratio: float = 0.0` | mini-batch 模式下的验证分割 |
| `train_auto()` | `val_ratio: float = 0.0` | 透传给实际调用的方法 |
| `_run_training_loop()` | `val_inputs=None, val_targets=None, val_masks=None` | 验证集张量（可选） |

### 2.6 CLI 参数

**文件**: `ai/train_main.py`

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--val-ratio` | float | 0.0 | 验证集比例，0 表示不分割，推荐 0.2 |

`--train-from-dir` 和 `--cycle` 模式均可使用。

## 3. 数据流

### 3.1 现有流程（无验证集）

```
sample_dirs → SampleLoader.load_multiple() → all_data
           → SampleLoader.prepare_tensors(all_data) → (inputs, targets, masks)
           → _run_training_loop(inputs, targets, masks)
```

### 3.2 新增流程（有验证集）

```
sample_dirs → SampleLoader.load_and_split(sample_dirs, val_ratio) → (train_data, val_data)
           → SampleLoader.prepare_tensors(train_data) → (train_inputs, train_targets, train_masks)
           → SampleLoader.prepare_tensors(val_data)   → (val_inputs, val_targets, val_masks)
           → _run_training_loop(train_inputs, train_targets, train_masks,
                                val_inputs, val_targets, val_masks)
```

## 4. 主流程伪代码

### 4.1 train_batch（含 val_ratio）

```
1. if val_ratio > 0:
       train_data, val_data = sample_loader.load_and_split(sample_dirs, val_ratio)
       train_tensors = prepare_tensors(train_data)
       val_tensors = prepare_tensors(val_data)
   else:
       train_tensors = prepare_tensors(load_multiple(sample_dirs))
       val_tensors = (None, None, None)

2. loss_logger = _setup_training(epochs=epochs)

3. _run_training_loop(*train_tensors, *val_tensors, epochs, loss_logger, verbose)
```

### 4.2 _run_training_loop（含 val 评估）

```
for epoch in range(epochs):
    # 正常训练（与现有逻辑完全一致）
    loss, grad_norm = _forward_backward_step(inputs, targets, masks)

    # val loss 评估：每 LOSS_LOG_INTERVAL 个 epoch
    if val_inputs is not None and (epoch + 1) % LOSS_LOG_INTERVAL == 0:
        val_loss = _evaluate_val_loss(val_inputs, val_targets, val_masks)
        # 确定 x 轴键名
        x_key = "step" if dense_log else "epoch"
        x_value = global_step if dense_log else (epoch + 1)
        loss_logger.log_val(x_value, val_loss)

    # 现有日志记录逻辑不变
    if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
        loss_logger.log(epoch + 1, loss.item(), grad_norm)
```

### 4.3 train_mini_batch（含 val_ratio）

```
for epoch in range(epochs):
    # 正常 mini-batch 训练（与现有逻辑完全一致）
    for batch in mini_batches:
        forward + backward + step
        log_dense(step, batch_loss, batch_grad)

    # epoch 结束后评估 val loss
    if val_inputs is not None:
        val_loss = _evaluate_val_loss(val_inputs, val_targets, val_masks)
        loss_logger.log_val(global_step, val_loss)

    # 学习率衰减
    _step_lr_scheduler(epoch_avg_loss)
```

### 4.4 _evaluate_val_loss

```
1. model.network.eval()
2. with torch.no_grad():
       outputs = model.network(val_inputs)
       val_loss = _masked_loss(outputs, val_targets, val_masks)
3. model.network.train()
4. return val_loss.item()
```

### 4.5 LossLogger.plot() 修改

```
1. 读取所有 entries
2. 分离 train entries（含 "loss" 字段）和 val entries（含 "val_loss" 字段）
3. train entries 按现有逻辑绘制（降采样 + 双 Y 轴）
4. val entries 叠加绘制到左 Y 轴：
   - 绿色折线，标记 "Val Loss"
   - 无需降采样（数据点已经稀疏）
   - x 轴值使用与 train entries 相同的 key（step 或 epoch）
```

### 4.6 CLI 集成

```
# --train-from-dir 模式
python ai/train_main.py --model cnn_v2 --train-from-dir ai/train_data/xxx \
    --epochs 1000 --batch-size 32 --val-ratio 0.2

# --cycle 模式（SelfPlayCycle 透传 val_ratio）
python ai/train_main.py --model cnn_v2 --cycle --rounds 5 --games 100 --val-ratio 0.2
```

## 5. 向后兼容性

- `val_ratio` 默认 0.0，所有现有调用无需修改
- `log_val()` 是新增方法，不影响现有 `log()` / `log_dense()`
- `plot()` 兼容无 val_loss 条目的旧日志文件（检测到无 val_loss 条目时行为不变）
- `_run_training_loop` 的 val 参数默认 None，现有调用无需修改

## 6. 测试用例

### 6.1 SampleLoader.load_and_split() 测试

| 用例 | 说明 |
|------|------|
| `test_split_ratio_correct` | 验证分割比例大致正确（允许 ±1 误差） |
| `test_split_no_overlap` | 训练集和验证集没有重叠样本 |
| `test_split_all_data_preserved` | 训练集 + 验证集 = 全部数据 |
| `test_split_reproducible_with_same_seed` | 相同 seed 产生相同分割 |
| `test_split_different_with_different_seed` | 不同 seed 产生不同分割 |
| `test_split_val_ratio_zero_returns_all_train` | val_ratio=0 时所有数据归训练集 |
| `test_split_val_ratio_invalid_raises` | val_ratio ≤ 0 或 ≥ 1 时抛出 ValueError |
| `test_split_single_sample` | 只有 1 个样本时，归训练集，验证集为空 |

### 6.2 LossLogger.log_val() 测试

| 用例 | 说明 |
|------|------|
| `test_log_val_writes_entry` | log_val 写入正确格式的条目 |
| `test_log_val_sparse_mode` | 稀疏模式使用 epoch 作为 x 轴 |
| `test_log_val_dense_mode` | 密集模式使用 step 作为 x 轴 |

### 6.3 LossLogger.plot() val_loss 测试

| 用例 | 说明 |
|------|------|
| `test_plot_with_val_loss` | 含 val_loss 条目时图表包含 Val Loss 折线 |
| `test_plot_without_val_loss` | 无 val_loss 条目时图表与原逻辑一致 |
| `test_plot_val_loss_same_x_axis` | val loss 与 train loss 共享 x 轴 |

### 6.4 BaseTrainer._evaluate_val_loss() 测试

| 用例 | 说明 |
|------|------|
| `test_evaluate_val_loss_returns_float` | 返回值为 float |
| `test_evaluate_val_loss_no_grad` | 评估时不计算梯度（参数不变） |
| `test_evaluate_val_loss_model_state` | 评估后模型恢复 train 模式 |

### 6.5 BaseTrainer train_batch/mini_batch 集成测试

| 用例 | 说明 |
|------|------|
| `test_train_batch_with_val_ratio` | val_ratio > 0 时 val_loss 被记录 |
| `test_train_batch_val_ratio_zero` | val_ratio=0 时行为与原逻辑一致 |
| `test_train_mini_batch_with_val_ratio` | mini-batch 模式下 val_loss 每 epoch 记录 |
| `test_train_auto_passes_val_ratio` | train_auto 透传 val_ratio |

### 6.6 CLI 参数测试

| 用例 | 说明 |
|------|------|
| `test_val_ratio_default_zero` | 默认值为 0.0 |
| `test_val_ratio_passed_to_trainer` | --val-ratio 参数正确传递到 trainer |
