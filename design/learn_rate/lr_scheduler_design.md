# 学习率衰减（LR Scheduler）设计文档

## 问题背景

当前训练使用固定学习率 0.001 + Adam 优化器。训练后期 loss landscape 变窄，固定学习率导致参数在谷底附近过冲（overshoot），表现为梯度范数增大且剧烈震荡。

## 设计目标

在训练过程中动态降低学习率，使参数能在后期"精细着陆"，减少震荡，提升收敛稳定性。

## 方案选择

实现两种 PyTorch 标准 scheduler，通过参数切换：

### 1. Step Decay（默认）

每 `step_size` 个 epoch，学习率乘以 `gamma`。

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
```

### 2. ReduceLROnPlateau

当 loss 连续 `patience` 个 epoch 不下降时，学习率乘以 `factor`。

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)
```

## API 变更

### BaseTrainer.__init__

新增可选参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr_scheduler_type` | `str` | `None` | 学习率调度类型，`"step"` / `"plateau"` / `None` |
| `lr_step_size` | `int` | `500` | StepLR 的衰减间隔（epoch） |
| `lr_gamma` | `float` | `0.9` | StepLR 的衰减系数 |
| `lr_patience` | `int` | `100` | ReduceLROnPlateau 的耐心值 |
| `lr_factor` | `float` | `0.5` | ReduceLROnPlateau 的衰减系数 |

### 新增属性

```python
self.lr_scheduler = None  # 初始化时根据 lr_scheduler_type 创建
```

### 训练方法中的调用

在每个 epoch 结束时调用 `scheduler.step()`：

```python
# StepLR：无参数
if self.lr_scheduler is not None:
    self.lr_scheduler.step()

# ReduceLROnPlateau：需要传入当前 loss
if self.lr_scheduler is not None:
    self.lr_scheduler.step(epoch_avg_loss)
```

### 新增方法：BaseTrainer._setup_lr_scheduler

```python
def _setup_lr_scheduler(self):
    """根据 lr_scheduler_type 初始化学习率调度器"""
    if self.lr_scheduler_type == "step":
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )
    elif self.lr_scheduler_type == "plateau":
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience
        )
```

### CLI 参数（train_main.py）

```python
parser.add_argument('--lr-scheduler', type=str, default=None,
                    choices=['step', 'plateau'],
                    help='学习率调度类型：step（固定间隔衰减）或 plateau（按平台衰减）')
parser.add_argument('--lr-step-size', type=int, default=500,
                    help='StepLR 衰减间隔（epoch），默认 500')
parser.add_argument('--lr-gamma', type=float, default=0.9,
                    help='StepLR 衰减系数，默认 0.9')
parser.add_argument('--lr-patience', type=int, default=100,
                    help='ReduceLROnPlateau 耐心值，默认 100')
parser.add_argument('--lr-factor', type=float, default=0.5,
                    help='ReduceLROnPlateau 衰减系数，默认 0.5')
```

### SelfPlayCycle 参数透传

在 `run()` 方法中，将 lr_scheduler 相关参数传递给 trainer 的训练方法。

## 训练循环修改点

### train_batch

```python
for epoch in range(epochs):
    # ... 训练逻辑 ...
    
    if (epoch + 1) % LOSS_LOG_INTERVAL == 0:
        # ... 记录 loss ...
        
    # 学习率衰减
    if self.lr_scheduler is not None:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(epoch_avg_loss)
        else:
            self.lr_scheduler.step()
```

### train_mini_batch

同上，在每个 epoch 结束时调用。

### train_single

同上。

### train_cycle

在 `run()` 方法中创建 scheduler（如果 trainer 还没有）。由于 train_cycle 是多轮的，scheduler 应在 train_cycle 开始时创建，并在每轮的每个 epoch 中 step。

**关键设计**：scheduler 的生命周期与 trainer 绑定，在 `__init__` 中创建。每个训练方法（train_batch/train_mini_batch）在每个 epoch 结束时 step。

## 日志输出

启用 scheduler 后，训练日志增加学习率信息：

```python
if verbose and (epoch + 1) % LOSS_LOG_INTERVAL == 0:
    current_lr = self.optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {epoch_avg_loss:.4f}, lr: {current_lr:.6f}")
```

## 向后兼容

- `lr_scheduler_type=None`（默认）时，不创建 scheduler，行为与之前完全一致
- 所有新增参数均为可选，不影响现有调用

## 测试策略

### 新增测试用例

1. **`test_step_lr_decay`**：
   - 创建 trainer，lr_scheduler_type="step"
   - 运行 3 个 epoch
   - 断言学习率按 gamma 衰减

2. **`test_plateau_lr_decay`**：
   - 创建 trainer，lr_scheduler_type="plateau"
   - 模拟 loss 不下降的情况
   - 断言学习率按 factor 衰减

3. **`test_no_scheduler_by_default`**：
   - 默认创建 trainer
   - 断言 lr_scheduler 为 None

4. **`test_scheduler_logs_lr`**：
   - 启用 scheduler
   - 训练时断言日志包含 lr 信息

5. **`test_train_cycle_passes_scheduler_kwargs`**：
   - mock trainer，调用 train_cycle 传入 lr_scheduler_type
   - 断言 trainer 被正确初始化

### 回归测试

- 运行 `tests/run_tests.py`，确保 494 个测试全部通过。
