# train_cycle 支持 Mini-Batch 训练设计文档

## 问题背景

当前 `train_cycle` 内部每轮调用的是 `train_batch`，其语义是"全量数据一次性前向传播，每个 epoch 只更新一次参数"。

项目中已实现了 `train_mini_batch` 方法（支持 `batch_size` 和 `shuffle`），但 `train_cycle` / `SelfPlayCycle` 并未使用它。用户希望 `train_cycle` 也能使用 mini-batch 训练。

## 设计目标

1. **命令行支持**：`train_main.py` 的 `--cycle` 模式新增 `--batch-size` 和 `--shuffle` 参数。
2. **训练器支持**：`BaseTrainer.train_cycle` 和 `SelfPlayCycle.run` 接收 `batch_size` / `shuffle` 参数。
3. **自动切换**：提供 `batch_size` 时调用 `train_mini_batch`，否则保持 `train_batch`（向后兼容）。
4. **参数打印**：`train_main.py` 在 cycle 模式下打印 batch_size 和 shuffle 信息。

## API 变更

### train_main.py 参数扩展

```
python ai/train_main.py --model cnn_v1 --cycle --rounds 3 --games 50 --batch-size 32 --shuffle
```

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | `int` | `None` | 每批次样本数。提供时启用 mini-batch 训练 |
| `--shuffle` | `flag` | `False` | 每个 epoch 是否打乱数据顺序（仅 mini-batch 模式有效） |

### BaseTrainer.train_cycle

```python
def train_cycle(
    self,
    rounds: int = 10,
    games: int = 100,
    epochs: int = 1000,
    rate: float = 0.9,
    batch_size: int = None,   # 新增
    shuffle: bool = False,    # 新增
    project_root: str = None
) -> None:
```

### SelfPlayCycle.run

```python
def run(
    self,
    rounds: int = 10,
    games: int = 100,
    epochs: int = 1000,
    rate: float = 0.9,
    batch_size: int = None,   # 新增
    shuffle: bool = False,    # 新增
    project_root: str = None
) -> None:
```

### 训练方法切换逻辑

在 `SelfPlayCycle.run` 内部，根据 `batch_size` 决定调用哪个方法：

```python
if batch_size is not None and batch_size > 0:
    # Mini-Batch 训练
    avg_loss = self.trainer.train_mini_batch(
        sample_dirs,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=True,
        session_dir=session_dir,
        log_type="epoch_level",
        round_num=round_num,
        total_rounds=rounds,
        epochs=epochs
    )
else:
    # 全量训练（向后兼容）
    avg_loss = self.trainer.train_batch(
        sample_dirs,
        epochs=epochs,
        verbose=True,
        session_dir=session_dir,
        log_type="epoch_level",
        round_num=round_num,
        total_rounds=rounds,
        epochs=epochs
    )
```

### 终端打印

Cycle 模式下的参数打印增加两行：

```
训练参数:
  轮次数 (rounds):     3
  每轮对局数 (games):  50
  每轮迭代数 (epochs): 1000
  奖励衰减率 (rate):   0.9
  批次大小 (batch):    32      ← 新增
  数据打乱 (shuffle):  True    ← 新增
  训练方法:            mini-batch
```

当 `batch_size` 为 None 时：

```
  批次大小 (batch):    N/A (全量更新)
  数据打乱 (shuffle):  N/A
  训练方法:            train_batch (全量更新)
```

## 向后兼容

- `--batch-size` 不提供时，`train_cycle` 行为与之前完全一致（调用 `train_batch`）。
- `BaseTrainer.train_cycle` 和 `SelfPlayCycle.run` 的新增参数均有默认值，不影响现有调用。

## 测试策略

### 新增测试用例

1. **`test_train_cycle_with_batch_size_calls_train_mini_batch`**：
   - Mock `trainer.train_mini_batch` 和 `train_batch`
   - 传入 `batch_size=32` 调用 `SelfPlayCycle.run`
   - 断言 `train_mini_batch` 被调用，`train_batch` 未被调用

2. **`test_train_cycle_without_batch_size_calls_train_batch`**：
   - Mock `trainer.train_mini_batch` 和 `train_batch`
   - 不传 `batch_size` 调用 `SelfPlayCycle.run`
   - 断言 `train_batch` 被调用，`train_mini_batch` 未被调用

3. **`test_train_cycle_passes_shuffle_to_mini_batch`**：
   - Mock `trainer.train_mini_batch`
   - 传入 `batch_size=16, shuffle=True`
   - 断言 `train_mini_batch` 的 `shuffle=True`

4. **`test_train_main_parser_batch_size_and_shuffle`**：
   - 测试 `argparse` 能正确解析 `--batch-size` 和 `--shuffle`

### 回归测试

- 运行 `tests/run_tests.py`，确保 482 个测试全部通过。
