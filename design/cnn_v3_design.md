# CNN v3 极简模型设计文档

## 1. 概述

CNN v3 是极简卷积神经网络模型，追求结构最简化：
- 三层卷积统一 32 通道，无 BatchNorm，无 Dropout
- FC 层先压缩到 64 维瓶颈，再映射到 225 维输出
- Conv → ReLU 直接堆叠，FC → Softmax 输出

### 1.1 设计动机

- 探索极简结构在五子棋任务上的学习能力
- 去除 BN 层观察对训练收敛的影响
- 减少超参数（无 BN 的 momentum/eps）
- 64 维瓶颈压缩大幅降低 FC 层参数量

## 2. 相对 CNN v2 的变更

### 2.1 网络结构对比

| 层 | CNN v2 | CNN v3 | 变更说明 |
|---|--------|--------|---------|
| Conv1 | 3→64 + BN + ReLU | **3→32 + ReLU** | **通道减半，去 BN** |
| Conv2 | 64→64 + BN + ReLU | **32→32 + ReLU** | **通道减半，去 BN** |
| Conv3 | 64→64 + BN + ReLU | **32→32 + ReLU** | **通道减半，去 BN** |
| Flatten | 64×15×15=14400 | **32×15×15=7200** | **降维50%** |
| FC1 | 14400→64 + ReLU | **7200→64 + ReLU** | **输入减半，瓶颈维度不变** |
| FC_output | 64→225 | 64→225 | 不变 |
| Softmax | dim=1 | dim=1 | 不变 |

### 2.2 参数量对比

| 层 | CNN v2 参数量 | CNN v3 参数量 |
|---|--------------|--------------|
| Conv1 | 1,792 + BN128 | **896** |
| Conv2 | 36,928 + BN128 | **9,248** |
| Conv3 | 36,928 + BN128 | **9,248** |
| FC1 | 921,664 | **460,864** |
| FC_output | 14,625 | **14,625** |
| **总计** | **约 101 万** | **约 49.5 万** |

CNN v3 通过 32 通道卷积 + 64 维瓶颈 FC，参数量降至 CNN v2 的一半。

### 2.3 接口

与 CNN v2 完全一致：
- `predict(board_vector)` → 225 维概率分布
- `save(model_path)` / `load(model_path)` → 状态字典存取
- `_prepare_input(board_vector)` → (1, 3, 15, 15) 张量（复用 board_vector_to_cnn_tensor）

## 3. 文件结构

```
ai/models/cnn_v3/
├── __init__.py
├── model.py          # CNNV3Network + CNNV3Model
└── trainer.py        # CNNV3Trainer
```

模型保存路径：`ai/models/cnn_v3/cnn_v3.pth`

## 4. 集成点

### 4.1 registry.py

在 MODEL_REGISTRY 中添加：
```python
"cnn_v3": ("ai.models.cnn_v3.model.CNNV3Model", "ai.models.cnn_v3.trainer.CNNV3Trainer"),
```

### 4.2 models/__init__.py

添加导出：
```python
from .cnn_v3 import CNNV3Model, CNNV3Trainer
```

### 4.3 battle.py / README.md

更新 AI 名称列表，添加 `cnn_v3`。

## 5. 测试用例

### 5.1 网络结构测试

```python
def test_network_structure():
    """前向传播输出形状为 (batch, 225)"""

def test_network_softmax_output_range():
    """[primary] Softmax 输出值域 (0, 1)"""

def test_network_softmax_row_sum():
    """[primary] Softmax 每行概率和为 1"""

def test_network_forward_single():
    """单样本前向传播"""

def test_network_forward_batch():
    """批量前向传播"""

def test_network_no_bn():
    """网络中不包含 BatchNorm 层"""

def test_network_param_count():
    """CNN v3 参数量应约为 49.5 万"""
```

### 5.2 模型类测试

```python
def test_model_name():
    """模型名称为 'cnn_v3'"""

def test_model_predict_shape():
    """predict 返回 225 维"""

def test_model_predict_probability_range():
    """[primary] 概率范围 (0, 1)"""

def test_model_predict_probability_sum():
    """[primary] 概率总和为 1"""

def test_model_save_and_load():
    """模型保存和加载"""

def test_model_load_nonexistent_file():
    """加载不存在的文件"""

def test_vector_to_tensor_conversion():
    """向量到张量转换"""
```

### 5.3 训练器测试

```python
def test_trainer_initialization():
    """初始化训练器"""

def test_trainer_model_name():
    """model_name 为 'cnn_v3'"""

def test_trainer_train_single():
    """单轮次训练"""

def test_trainer_prepare_tensors():
    """准备张量，CNN 输入为 4D"""

def test_trainer_fresh_parameter():
    """fresh 参数创建新模型"""

def test_trainer_inherits_base_trainer():
    """继承自 BaseTrainer"""
```

### 5.4 注册表测试

```python
def test_get_model_cnn_v3():
    """registry 能获取 cnn_v3 模型"""

def test_get_trainer_cnn_v3():
    """registry 能获取 cnn_v3 训练器"""
```
