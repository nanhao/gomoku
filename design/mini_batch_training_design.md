# Mini-Batch 训练设计文档

## 需求

将训练代码从全批量梯度下降改为 mini-batch 训练，提高 GPU 利用效率和训练效果。

## 解决思路

在 `BaseTrainer` 中新增 `train_mini_batch` 方法：
- 将合并后的数据集分成小批次
- 每个 epoch 内按批次进行前向、反向传播
- 支持每个 epoch 打乱数据顺序

## 设计

### 新增方法签名

```python
def train_mini_batch(
    self,
    sample_dirs: List[str],
    epochs: int = 1000,
    batch_size: int = 32,
    shuffle: bool = True,
    verbose: bool = True
) -> float:
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sample_dirs | List[str] | 必填 | sample.json 目录列表 |
| epochs | int | 1000 | 训练轮数 |
| batch_size | int | 32 | 每批次样本数 |
| shuffle | bool | True | 每个 epoch 是否打乱数据 |
| verbose | bool | True | 是否输出训练信息 |

### 返回值

- `float`: 整个训练过程的平均损失值

### 伪代码主流程

```
def train_mini_batch(sample_dirs, epochs, batch_size, shuffle, verbose):
    # 1. 合并所有 sample 数据
    all_sample_data = []
    for dir in sample_dirs:
        all_sample_data.extend(load_sample(dir))
    
    # 2. 转换为张量
    inputs, targets, masks = prepare_tensors(all_sample_data)
    N = inputs.shape[0]  # 样本总数
    
    # 3. 训练循环
    total_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        # 3.1 生成批次索引
        if shuffle:
            indices = random_permutation(N)
        else:
            indices = range(N)
        
        # 3.2 按批次训练
        for batch_indices in split_into_batches(indices, batch_size):
            # 获取批次数据
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            batch_masks = masks[batch_indices]
            
            # 前向传播
            outputs = model(batch_inputs)
            loss = masked_loss(outputs, batch_targets, batch_masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        # 3.3 输出训练信息
        if verbose and (epoch + 1) % 200 == 0:
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch + 1}/{epochs}, avg_loss: {avg_loss:.4f}")
    
    return total_loss / total_batches
```

### 与 `train_batch` 的区别

| 特性 | train_batch | train_mini_batch |
|------|-------------|------------------|
| 更新频率 | 每 epoch 1 次 | 每 epoch N/batch_size 次 |
| GPU 内存占用 | 高（全部数据） | 低（batch_size） |
| 梯度稳定性 | 高 | 中等 |
| 随机性 | 无 | 有（shuffle） |
| 适用场景 | 小数据集 | 大数据集 |

## 主流程测试用例

### 测试 1：基本功能测试

```python
def test_train_mini_batch_basic(trainer, sample_dirs):
    """测试 mini-batch 训练基本功能"""
    avg_loss = trainer.train_mini_batch(
        sample_dirs,
        epochs=10,
        batch_size=4,
        shuffle=False,
        verbose=False
    )
    
    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
```

### 测试 2：shuffle 功能测试

```python
def test_train_mini_batch_shuffle(trainer, sample_dirs):
    """测试 shuffle 功能不影响训练结果"""
    # 固定随机种子，两次训练结果应相近
    set_seed(42)
    loss1 = trainer.train_mini_batch(sample_dirs, epochs=5, shuffle=True)
    
    set_seed(42)
    loss2 = trainer.train_mini_batch(sample_dirs, epochs=5, shuffle=True)
    
    assert abs(loss1 - loss2) < 1e-6
```

### 测试 3：batch_size 边界测试

```python
def test_train_mini_batch_batch_size_edge(trainer, sample_dirs):
    """测试 batch_size 大于样本数的情况"""
    # batch_size 设为很大的值，应退化为单批次
    avg_loss = trainer.train_mini_batch(
        sample_dirs,
        epochs=5,
        batch_size=10000,  # 远大于样本数
        verbose=False
    )
    
    assert isinstance(avg_loss, float)
```

### 测试 4：空数据异常测试

```python
def test_train_mini_batch_empty_data(trainer):
    """测试空数据列表抛出异常"""
    with pytest.raises(ValueError):
        trainer.train_mini_batch([])
```

## 文件修改范围

- `ai/base_trainer.py`: 新增 `train_mini_batch` 方法
- `tests/ai/`: 新增测试用例
