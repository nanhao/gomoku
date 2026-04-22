# Mini-Batch 密集 Loss 日志与梯度可视化设计文档

## 问题背景

当前 `train_mini_batch` 只在每 `LOSS_LOG_INTERVAL`（200）个 epoch 记录一次平均 loss，无法观察：
1. 单个 batch 级别的 loss 波动
2. 梯度大小与 loss 变化的关系
3. 训练过程中的瞬态行为（如梯度爆炸/消失）

用户希望记录每次参数更新时的 loss 和梯度大小，同时避免：
- 写文件太频繁拖累训练速度
- 画图时点太多把图撑爆

## 设计目标

1. **密集日志**：mini-batch 模式下，每个 batch 记录 `(step, loss, grad_norm)`
2. **性能友好**：内存缓冲 + 批量写入，最小化磁盘 I/O
3. **可视化增强**：双 Y 轴图（Loss + Gradient Norm），自动降采样防图撑爆
4. **向后兼容**：全量更新模式（train_batch）保持现有稀疏日志行为

## API 变更

### LossLogger.__init__

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dense_log` | `bool` | `False` | 是否启用密集日志模式（仅 mini-batch 有效） |

### 新增方法：LossLogger.log_dense

```python
def log_dense(self, step: int, loss: float, grad_norm: float) -> None:
    """记录一条密集日志（存入内存缓冲区，不直接写文件）

    Args:
        step: 全局步数（batch 序号）
        loss: 当前 batch 的 loss 值
        grad_norm: 当前 batch 的梯度 L2 范数
    """
```

### 新增方法：LossLogger.flush

```python
def flush(self) -> None:
    """将内存缓冲区中的所有日志批量写入文件

    应在每个 epoch 结束后调用，或训练结束时调用。
    """
```

### LossLogger.plot 升级

- **数据格式识别**：自动检测日志中是否包含 `grad_norm` 字段
- **双 Y 轴**：如果包含 `grad_norm`，左轴显示 Loss（蓝色），右轴显示 Grad Norm（红色）
- **自动降采样**：总点数 > 5000 时，每 `ceil(total / 5000)` 个点取平均，确保图上最多 5000 个点

### 数据格式

密集模式下的 jsonl 格式：

```jsonl
{"step": 1, "loss": 0.0025, "grad_norm": 0.15}
{"step": 2, "loss": 0.0023, "grad_norm": 0.14}
```

稀疏模式（现有）保持不变：

```jsonl
{"epoch": 200, "loss": 0.0020}
```

## 梯度采集逻辑

在 `train_mini_batch` 中，每个 batch 的 `backward()` 之后、`step()` 之前计算梯度范数：

```python
loss.backward()

# 计算梯度 L2 范数
grad_norm = sum(
    p.grad.norm().item()
    for p in self.model.network.parameters()
    if p.grad is not None
)

self.optimizer.step()

# 记录密集日志
if loss_logger and dense_log:
    loss_logger.log_dense(global_step, loss.item(), grad_norm)
    global_step += 1
```

## 内存缓冲与批量写入策略

```python
def __init__(self, ..., dense_log: bool = False):
    self.dense_log = dense_log
    self._buffer = []  # 内存缓冲区

def log_dense(self, step, loss, grad_norm):
    self._buffer.append({"step": step, "loss": loss, "grad_norm": grad_norm})

def flush(self):
    if self._buffer:
        with open(self._log_path, 'a', encoding='utf-8') as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()
```

写入时机：
- 每个 epoch 结束后调用 `flush()`
- `_finalize_training()` 中调用 `flush()` 确保最后一批数据落盘

## 双 Y 轴图与降采样

```python
def _downsample(self, entries: list, max_points: int = 5000) -> list:
    """降采样：超过 max_points 时合并相邻点"""
    if len(entries) <= max_points:
        return entries
    
    window = max(1, len(entries) // max_points)
    result = []
    for i in range(0, len(entries), window):
        chunk = entries[i:i + window]
        avg_step = sum(e["step"] for e in chunk) / len(chunk)
        avg_loss = sum(e["loss"] for e in chunk) / len(chunk)
        avg_grad = sum(e["grad_norm"] for e in chunk) / len(chunk)
        result.append({"step": avg_step, "loss": avg_loss, "grad_norm": avg_grad})
    return result
```

绘图逻辑：

```python
if has_grad_norm:
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(steps, losses, 'b-', label='Loss', linewidth=1)
    ax2.plot(steps, grad_norms, 'r-', label='Grad Norm', linewidth=1, alpha=0.7)
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color='b')
    ax2.set_ylabel("Gradient Norm", color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
else:
    # 旧行为
    plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=1.5, markersize=3)
```

## 调用链修改

### SelfPlayCycle.run

当使用 mini-batch（batch_size 提供）时，创建 LossLogger 传入 `dense_log=True`：

```python
loss_logger = LossLogger(
    model_name=self.trainer.model_name,
    session_dir=session_dir,
    log_type="epoch_level",
    round_num=round_num,
    total_rounds=rounds,
    epochs=epochs,
    dense_log=True  # mini-batch 模式启用密集日志
)
```

### BaseTrainer._finalize_training

训练结束时调用 `flush()`：

```python
def _finalize_training(self, loss_logger, total_loss, count):
    if loss_logger:
        loss_logger.flush()  # 确保最后一批数据落盘
        loss_logger.plot()
    return total_loss / count if count > 0 else 0.0
```

## 向后兼容

- `dense_log=False`（默认）时，`log()` 和 `plot()` 行为与之前完全一致
- `train_batch`（全量更新模式）不使用 `log_dense`，保持稀疏日志
- 旧日志文件（只有 `epoch`/`loss`）的 `plot()` 仍然正常工作

## 测试策略

### 新增测试用例

1. **`test_log_dense_buffers_in_memory`**：
   - 调用 `log_dense()` 多次
   - 断言文件未写入（或为空）
   - 调用 `flush()` 后断言文件有数据

2. **`test_log_dense_format`**：
   - flush 后读取 jsonl，断言每条包含 `step`/`loss`/`grad_norm`

3. **`test_plot_with_grad_norm_dual_axis`**：
   - 写入含 `grad_norm` 的数据
   - mock `plt.subplots` 断言创建了 `twinx()`

4. **`test_downsample_reduces_points`**：
   - 构造 10000 条密集数据
   - 断言降采样后 <= 5000 点

5. **`test_backward_compatibility_sparse_plot`**：
   - 只有 `epoch`/`loss` 的旧格式数据
   - plot 正常生成，不尝试创建双 Y 轴

### 回归测试

- 运行 `tests/run_tests.py`，确保 487 个测试全部通过。
