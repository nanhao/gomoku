# CNN v5 纯卷积极简模型设计文档

## 1. 概述

CNN v5 在 cnn_v4 基础上增加一个 32 通道隐藏层，保持纯卷积结构不变：
- 纯卷积网络，无 FC 层、无 BatchNorm、无 Dropout
- 用 1×1 卷积替代 FC 层，直接输出 15×15 空间概率图
- 全程保持空间结构，Conv → ReLU → 1×1 Conv → LogSoftmax

### 1.1 设计动机

- 在 cnn_v4（4 层卷积）基础上进一步增加网络深度
- 探索更深的纯卷积结构在五子棋任务上的表达能力
- 保持 cnn_v4 的所有其他设计不变，仅增加一层卷积

## 2. 相对 CNN v4 的变更

### 2.1 网络结构对比

| 层 | CNN v4 | CNN v5 | 变更说明 |
|---|--------|--------|---------|
| Conv1 | 3→32 + ReLU | 3→32 + ReLU | 不变 |
| Conv2 | 32→32 + ReLU | 32→32 + ReLU | 不变 |
| Conv3 | 32→32 + ReLU | 32→32 + ReLU | 不变 |
| Conv4 | 32→32 + ReLU | 32→32 + ReLU | 不变 |
| Conv5 | — | **32→32 + ReLU** | **新增** |
| 输出层 | Conv1×1(32→1) | Conv1×1(32→1) | 不变 |
| LogSoftmax | dim=1 | dim=1 | 不变 |

### 2.2 参数量对比

| 层 | CNN v4 参数量 | CNN v5 参数量 |
|---|--------------|--------------|
| Conv1 | 896 | 896 |
| Conv2 | 9,248 | 9,248 |
| Conv3 | 9,248 | 9,248 |
| Conv4 | 9,248 | 9,248 |
| Conv5 | — | **9,248** |
| Conv_out(1×1) | 33 | 33 |
| **总计** | **约 2.8 万** | **约 3.8 万** |

CNN v5 比 v4 多一层 32→32 卷积，参数量增加约 9248。

### 2.3 接口

与 CNN v4 完全一致：
- `predict(board_vector)` → 225 维概率分布
- `save(model_path)` / `load(model_path)` → 状态字典存取
- `_prepare_input(board_vector)` → (1, 3, 15, 15) 张量（复用 board_vector_to_cnn_tensor）
- 多代模型，支持 `gen` 参数

## 3. 文件结构

```
ai/models/cnn_v5/
├── __init__.py
├── model.py          # CNNV5Network + CNNV5Model
└── trainer.py        # CNNV5Trainer
```

模型保存路径（多代）：`ai/models/cnn_v5/gen{i}/cnn_v5.pth`

## 4. 集成点

### 4.1 registry.py

在 MODEL_REGISTRY 中添加：
```python
"cnn_v5": RegistryEntry(
    "ai.models.cnn_v5.model.CNNV5Model",
    "ai.models.cnn_v5.trainer.CNNV5Trainer", True),
```

### 4.2 models/__init__.py

添加导出：
```python
from .cnn_v5 import CNNV5Model, CNNV5Trainer
```

## 5. 测试用例

### 5.1 网络结构测试

在 `tests/ai/test_cnn_models.py` 中新增 `TestCNNV5Network` 类，继承 `CNNNetworkSharedMixin`：

```python
def test_network_no_bn():
    """网络中不包含 BatchNorm 层"""

def test_network_no_dropout():
    """网络中不包含 Dropout 层"""

def test_network_param_count():
    """CNN v5 参数量约 3.8 万（5 层 32 通道卷积 + 1×1 输出）"""

def test_network_conv_channels():
    """5 层卷积层，通道数均为 32"""

def test_network_fc_direct_output():
    """无 FC 层，输出层为 1×1 卷积(32→1)"""
```

### 5.2 模型类测试

新增 `TestCNNV5Model` 类，继承 `CNNModelSharedMixin`：
- `MODEL_CLS = CNNV5Model`
- `MODEL_NAME = "cnn_v5"`
- `DEFAULT_FILENAME = "cnn_v5.pth"`
