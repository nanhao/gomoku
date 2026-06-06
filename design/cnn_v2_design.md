# CNN v2 模型设计文档

## 1. 概述

CNN v2 是 CNN v1 的轻量化改进版本，基于 `design/cnn_design.md` 进行优化。

### 1.1 改进背景

CNN v1 参数量约 445 万，与 FC v1 对齐。但在实际训练中发现：
- 训练 2000 epoch 耗时较长
- FC1 层（28800→150）独占 432 万参数（97%），是计算瓶颈
- 卷积层提取的特征维度（150）可能存在冗余

### 1.2 改进目标

在保持卷积层特征提取能力不变的前提下，**压缩 FC 层参数量 56%**，从而：
- 减少训练时间（forward/backward 计算量减少）
- 降低内存占用
- 保持与 CNN v1 可比的预测精度

## 2. 相对 CNN v1 的变更

### 2.1 网络结构变更

| 层 | CNN v1 | CNN v2 | 变更说明 |
|---|--------|--------|---------|
| Conv1 | 3→32 | **3→64** | **通道数翻倍** |
| Conv2 | 32→64 | **64→64** | **统一64通道** |
| Conv3 | 64→128 | **64→64** | **统一64通道** |
| Flatten | 128×15×15=28800 | **64×15×15=14400** | **降维50%** |
| **FC1** | **28800→150** | **14400→64** | **维度压缩** |
| FC_output | 150→225 | 64→225 | 适配 FC1 输出 |
| LogSoftmax | dim=1 | dim=1 | 不变 |

### 2.2 参数量对比

| 层 | CNN v1 参数量 | CNN v2 参数量 |
|---|--------------|--------------|
| Conv1 (3→64) | 896 | **1,792** |
| Conv2 (64→64) | 18,496 | **36,928** |
| Conv3 (64→64) | 73,856 | **36,928** |
| **FC1** | **4,320,150** | **921,664** |
| FC_output | 33,975 | 14,625 |
| BN 层 | ~1,000 | ~384 |
| **总计** | **约 445 万** | **约 101 万** |

**总参数量减少约 344 万（-77%）**。

### 2.3 接口变更

CNN v2 的 `CNNModel.name` 返回 `"cnn_v2"`，其余接口与 CNN v1 完全一致：
- `predict(board_vector)` → 225 维概率分布
- `save(model_path)` / `load(model_path)` → 状态字典存取
- `_vector_to_tensor(board_vector)` → (1, 3, 15, 15) 张量

## 3. 文件结构

```
ai/models/cnn_v2/
├── __init__.py
├── model.py          # CNNNetwork + CNNModel
└── trainer.py        # CNNTrainer
```

模型保存路径：`ai/models/cnn_v2/cnn_v2.pth`

## 4. 集成点

### 4.1 train_main.py

在 `get_trainer()` 中添加：
```python
elif model_name == "cnn_v2":
    from ai.models.cnn_v2.trainer import CNNTrainer
    return CNNTrainer(fresh=fresh, **kwargs)
```

### 4.2 CLI help 文本

`--model` 参数帮助文本更新为：
```
模型名称，如 fc_v1、cnn_v1、cnn_v2，默认 fc_v1
```

## 5. 测试用例

继承 CNN v1 的全部测试，并增加参数量验证：

```python
def test_cnn_v2_param_count():
    """CNN v2 参数量应约为 101 万"""
    model = CNNModel()
    total = sum(p.numel() for p in model.network.parameters())
    assert 1_000_000 <= total <= 1_050_000
```
