 # CNN 模型设计文档

## 1. 概述

实现 CNN 卷积神经网络模型，用于五子棋落子位置的 reward 预测。

## 2. 接口设计

### 2.1 CNNNetwork 类

```python
class CNNNetwork(nn.Module):
    """CNN 神经网络结构"""
    
    def __init__(self):
        """初始化网络结构"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, 15, 15)
            
        Returns:
            输出张量，形状为 (batch_size, 225)
        """
        pass
```

### 2.2 CNNModel 类

```python
class CNNModel(BaseModel):
    """CNN 模型"""
    
    DEFAULT_MODEL_PATH: str  # 默认模型保存路径
    
    def __init__(self, model_path: str = None, fresh: bool = False):
        """初始化模型
        
        Args:
            model_path: 模型文件路径
            fresh: 是否强制创建新模型
        """
        pass
    
    @property
    def name(self) -> str:
        """返回模型名称 "cnn" """
        pass
    
    def predict(self, board_vector: List[int]) -> List[int]:
        """预测每个落子位置的 reward
        
        Args:
            board_vector: 226 维向量
            
        Returns:
            225 维 reward 列表
        """
        pass
    
    def save(self, model_path: str = None) -> str:
        """保存模型参数"""
        pass
    
    def load(self, model_path: str):
        """加载模型参数"""
        pass
    
    def _vector_to_tensor(self, board_vector: List[int]) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量"""
        pass
```

## 3. 网络结构

### 3.1 设计原则

为了确保模型能在有限的训练数据下有效学习，采用轻量化设计：
- 参数量控制在 ~400 万，与 FC v1 模型相当
- 3 层卷积提取空间特征
- 紧凑的全连接层

### 3.2 网络架构

```
输入: (batch_size, 3, 15, 15)
  │
  ├── 通道1: 黑棋位置 (1=黑棋, 0=其他)
  ├── 通道2: 白棋位置 (1=白棋, 0=其他)
  └── 通道3: 当前落子方 (全1=黑方, 全0=白方)
  │
  ▼
Conv2d(3, 32, 3x3, padding=1) → BatchNorm2d → ReLU
  │
  ▼
Conv2d(32, 64, 3x3, padding=1) → BatchNorm2d → ReLU
  │
  ▼
Conv2d(64, 128, 3x3, padding=1) → BatchNorm2d → ReLU
  │
  ▼
全局平均池化 (AdaptiveAvgPool2d): (batch, 128, 15, 15) → (batch, 128)
  │
  ▼
Linear(128, 225)
  │
  ▼
Softmax
  │
  ▼
输出: (batch_size, 225) 概率分布，值域 (0, 1)，总和为 1
```

### 3.3 参数量估算

| 层 | 参数量 |
|---|--------|
| Conv1 (3→32) | 3×32×3×3 + 32 = 896 |
| Conv2 (32→64) | 32×64×3×3 + 64 = 18,496 |
| Conv3 (64→128) | 64×128×3×3 + 128 = 73,856 |
| FC (128→225) | 128×225 + 225 = 29,025 |
| BN 层 | ~1,000 |
| **总计** | **约 12 万** |

> 注：使用全局平均池化后，参数量从 1500 万降至约 12 万，比 FC v1 的 445 万小 37 倍，训练速度大幅提升。

## 4. 主流程伪代码

### 4.1 输入编码流程

```
函数 _vector_to_tensor(board_vector):
    # 提取棋盘和当前落子方
    board = board_vector[:225]
    current_player = board_vector[225]
    
    # 创建 3 个通道
    black_channel = [1.0 if x == 1 else 0.0 for x in board]
    white_channel = [1.0 if x == 2 else 0.0 for x in board]
    player_channel = [1.0 if current_player == 1 else 0.0] * 225
    
    # 重塑为 15x15 矩阵
    black_matrix = reshape(black_channel, (15, 15))
    white_matrix = reshape(white_channel, (15, 15))
    player_matrix = reshape(player_channel, (15, 15))
    
    # 堆叠为张量 (3, 15, 15)
    tensor = stack([black_matrix, white_matrix, player_matrix])
    
    # 添加 batch 维度 (1, 3, 15, 15)
    return tensor.unsqueeze(0)
```

### 4.2 预测流程

```
函数 predict(board_vector):
    # 转换输入
    input_tensor = _vector_to_tensor(board_vector)
    
    # 前向传播
    network.eval()
    with torch.no_grad():
        output_tensor = network(input_tensor)
    
    # 转换输出
    output_values = output_tensor.squeeze(0).tolist()
    
    # 构建 225 维结果，限制范围
    result = []
    for i in range(225):
        reward = int(output_values[i])
        reward = max(-1000000, min(1000000, reward))
        result.append(reward)
    
    return result
```

## 5. 主流程测试用例

### 5.1 网络结构测试

```python
def test_network_structure():
    """测试网络输入输出维度正确"""
    network = CNNNetwork()
    input_tensor = torch.randn(4, 3, 15, 15)
    output = network(input_tensor)
    assert output.shape == (4, 225)

def test_network_forward_single():
    """测试单样本前向传播"""
    network = CNNNetwork()
    input_tensor = torch.randn(1, 3, 15, 15)
    output = network(input_tensor)
    assert output.shape == (1, 225)
```

### 5.2 模型预测测试

```python
def test_model_predict_shape():
    """测试 predict 返回 225 维列表"""
    model = CNNModel()
    board_vector = [0] * 225 + [1]
    result = model.predict(board_vector)
    assert len(result) == 225

def test_model_predict_prob_range():
    """测试输出为 Softmax 概率，值域 (0, 1)，总和为 1"""
    model = CNNModel()
    board_vector = [0] * 225 + [1]
    result = model.predict(board_vector)
    for prob in result:
        assert 0 < prob < 1
    assert abs(sum(result) - 1.0) < 1e-5

def test_model_name():
    """测试模型名称为 "cnn_v1"""
    model = CNNModel()
    assert model.name == "cnn_v1"
```

### 5.3 输入编码测试

```python
def test_vector_to_tensor_conversion():
    """测试向量到张量的转换"""
    model = CNNModel()
    board_vector = [0] * 225 + [1]
    board_vector[0] = 1  # (0,0) 黑棋
    board_vector[16] = 2  # (1,1) 白棋
    
    tensor = model._vector_to_tensor(board_vector)
    
    assert tensor.shape == (1, 3, 15, 15)
    assert tensor[0, 0, 0, 0] == 1.0  # 黑棋通道
    assert tensor[0, 1, 1, 1] == 1.0  # 白棋通道
    assert tensor[0, 2, 0, 0] == 1.0  # 当前方通道（黑方）
```

### 5.4 保存加载测试

```python
def test_model_save_and_load():
    """测试模型保存和加载"""
    model1 = CNNModel()
    model1.save("test_model.pth")
    
    model2 = CNNModel("test_model.pth")
    
    board_vector = [0] * 225 + [1]
    assert model1.predict(board_vector) == model2.predict(board_vector)
```

## 6. 集成点

### 6.1 battle.py

在 `_AI_MODEL_MAP` 中添加：
```python
"cnn": "ai.models.cnn_v1.model.CNNModel"
```

### 6.2 engine.py

在 `create_engine` 的 `model_map` 中添加：
```python
"cnn": CNNModel
```

### 6.3 models/__init__.py

添加导出：
```python
from .cnn import CNNModel
```
