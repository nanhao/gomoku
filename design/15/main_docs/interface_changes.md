# 接口变更详细设计

---

## 1. 新增模块：ai/models/cnn_v6/

### 1.1 类：CNNV6Network

**继承**：`torch.nn.Module`

#### 公开接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `forward` | `(x: Tensor) -> Tensor` | 前向传播。输入 (N,3,15,15)，输出 (N,225) log-probability |

`forward` 不提供额外参数（如 temperature、mask 等），保持与所有已有 CNN 网络接口一致。

---

### 1.2 类：CNNV6Model

**继承**：`NeuralNetworkModel`

#### 公开接口

| 接口 | 类型 | 说明 |
|------|------|------|
| `name` | property → `str` | 返回 `"cnn_v6"` |
| `full_name` | property → `str` | 非多代返回 `"cnn_v6"`，多代返回 `"cnn_v6.N"`（继承自基类） |
| `DEFAULT_MODEL_PATH` | class attribute | `ai/models/cnn_v6/cnn_v6.pth` |

#### 继承自 NeuralNetworkModel 的公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `predict` | `(board_vector: List[int]) -> List[float]` | 返回 225 维概率分布，值域 (0,1)，总和 1 |
| `save` | `(model_path: str = None) -> str` | 保存 state_dict，返回实际路径 |
| `load` | `(model_path: str)` | 加载 state_dict |

#### 受保护方法（供基类/训练器调用）

| 方法 | 签名 | 说明 |
|------|------|------|
| `_create_network` | `() -> nn.Module` | 返回 `CNNV6Network()` 实例 |
| `_prepare_input` | `(board_vector: List[int], device: torch.device) -> Tensor` | 调用 `board_vector_to_cnn_tensor`，返回 (1,3,15,15) |

---

### 1.3 类：CNNV6Trainer

**继承**：`BaseTrainer`

#### 公开接口

| 接口 | 类型 | 说明 |
|------|------|------|
| `model_name` | property → `str` | 返回 `"cnn_v6"` |

#### 受保护方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `_create_model` | `(fresh: bool) -> CNNV6Model` | 创建模型实例，支持 gen 参数透传 |
| `_convert_inputs` | `(inputs: List[List[int]]) -> Tensor` | 逐条调用 `board_vector_to_cnn_tensor` 后 cat 为 (N,3,15,15) |

---

### 1.4 模块：ai/models/cnn_v6/__init__.py

#### 导出

```
CNNV6Model
CNNV6Trainer
```

---

## 2. 已有模块修改

### 2.1 ai/registry.py

**新增注册条目**（追加在 `MODEL_REGISTRY` 字典末尾）：

```
"cnn_v6": RegistryEntry(
    "ai.models.cnn_v6.model.CNNV6Model",
    "ai.models.cnn_v6.trainer.CNNV6Trainer",
    True),
```

### 2.2 ai/models/__init__.py

**新增导入**（追加在现有导入行之后）：

```
from .cnn_v6 import CNNV6Model, CNNV6Trainer
```

**修改 `__all__`**：追加 `"CNNV6Model"`, `"CNNV6Trainer"`

**修改模块文档**：在 docstring 中追加 `- cnn_v6: CNN v6 残差卷积神经网络 AI`

---

## 3. 未变更模块

| 模块 | 原因 |
|------|------|
| `ai/base_model.py` | `NeuralNetworkModel` 接口无需修改 |
| `ai/base_trainer.py` | `BaseTrainer` 接口无需修改 |
| `ai/models/cnn_v5/` | 完全保留 |
| `ai/engine.py` | 通过 `get_model("cnn_v6.N")` 间接使用，无需感知具体网络 |
| `ai/create_samples.py` | 同上 |
| `ai/battle.py` | 同上 |
| `ai/iterative_train.py` | 同上（若需支持 v6 迭代训练，在后续 PRD 中单独处理） |
