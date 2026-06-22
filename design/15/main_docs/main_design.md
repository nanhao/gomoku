# CNN v6 残差卷积网络设计文档

| 字段 | 值 |
|------|-----|
| intention_id | 15 |
| PRD | requirements/15/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

新增 **cnn_v6** 模型，在 cnn_v5 的五层纯卷积基础上引入 BatchNorm 和残差连接（Residual Connection）：

- 每层 Conv 后添加 BN，位于 ReLU 之前
- Conv2+Conv3 组成残差块 1，Conv4+Conv5 组成残差块 2
- 其他结构（输入编码、1×1 输出卷积、LogSoftmax）与 cnn_v5 一致

**修改范围**：3 个新增文件 + 3 个已有文件的注册/导出修改，无已有代码修改（cnn_v5 完全保留）。

前向传播流程见 [forward_flow.puml](forward_flow.puml)。

---

## 2. 网络结构设计

### 2.1 层定义

```
Conv1:  nn.Conv2d(3,  32, kernel_size=3, padding=1)
BN1:    nn.BatchNorm2d(32)
ReLU1:  nn.ReLU()

Conv2:  nn.Conv2d(32, 32, kernel_size=3, padding=1)
BN2:    nn.BatchNorm2d(32)
ReLU2:  nn.ReLU()

Conv3:  nn.Conv2d(32, 32, kernel_size=3, padding=1)
BN3:    nn.BatchNorm2d(32)
        残差连接: x = ReLU(BN3(Conv3(ReLU2(BN2(Conv2(x))))) + x)   ← 块输入为 Conv1+BN1+ReLU1 的输出
ReLU3:  nn.ReLU()

Conv4:  nn.Conv2d(32, 32, kernel_size=3, padding=1)
BN4:    nn.BatchNorm2d(32)
ReLU4:  nn.ReLU()

Conv5:  nn.Conv2d(32, 32, kernel_size=3, padding=1)
BN5:    nn.BatchNorm2d(32)
        残差连接: x = ReLU(BN5(Conv5(ReLU4(BN4(Conv4(x))))) + x)   ← 块输入为 ReLU3 的输出
ReLU5:  nn.ReLU()

Conv_out: nn.Conv2d(32, 1, kernel_size=1)
LogSoftmax: nn.LogSoftmax(dim=1)
```

### 2.2 残差块结构

两个残差块结构完全相同（仅来源不同），每个包含 2 层 3×3 卷积：

```
输入 x (batch, 32, 15, 15)
  │
  ├─ Conv_a (32→32, 3×3, padding=1)
  ├─ BN_a
  ├─ ReLU
  ├─ Conv_b (32→32, 3×3, padding=1)
  ├─ BN_b
  │
  └─[+] ← shortcut: 原始输入 x
     │
     └─ ReLU → 输出 (batch, 32, 15, 15)
```

> 由于卷积层使用 padding=1、stride=1、通道数保持 32，空间尺寸和通道数均不变，shortcut 可直接相加，无需 1×1 投影层。

### 2.3 参数量

| 组件 | 参数量 |
|------|--------|
| Conv1 + BN1 | 896 + 64 = 960 |
| Conv2 + BN2 | 9,248 + 64 = 9,312 |
| Conv3 + BN3 | 9,248 + 64 = 9,312 |
| Conv4 + BN4 | 9,248 + 64 = 9,312 |
| Conv5 + BN5 | 9,248 + 64 = 9,312 |
| Conv_out (1×1) | 33 |
| **总计** | **38,241** |

---

## 3. 组件设计

### 3.1 CNNV6Network（nn.Module 子类）

| 公开接口 | 说明 |
|----------|------|
| `forward(x: Tensor) -> Tensor` | 输入 (N,3,15,15)，输出 (N,225) 的 log-probability |

网络内部包含 Conv1～5、BN1～5、ReLU1～5、Conv_out、LogSoftmax 作为实例属性。残差连接逻辑在 forward 中内联实现（不单独封装为子模块，避免引入不必要的公开接口）。

### 3.2 CNNV6Model（NeuralNetworkModel 子类）

| 公开接口 | 说明 |
|----------|------|
| `predict(board_vector: List[int]) -> List[float]` | 继承自基类，返回 225 维概率分布 |
| `save(model_path: str = None) -> str` | 继承自基类，保存 state_dict |
| `load(model_path: str)` | 继承自基类，加载 state_dict |
| `name` (property) → `"cnn_v6"` | 模型名称 |
| `full_name` (property) → `"cnn_v6.N"` | 多代完整名称 |
| `DEFAULT_MODEL_PATH` (class attr) | `ai/models/cnn_v6/cnn_v6.pth` |
| `_create_network() -> nn.Module` | 受保护方法，返回 CNNV6Network 实例 |
| `_prepare_input(board_vector, device) -> Tensor` | 受保护方法，复用 `board_vector_to_cnn_tensor` |

### 3.3 CNNV6Trainer（BaseTrainer 子类）

| 公开接口 | 说明 |
|----------|------|
| `model_name` (property) → `"cnn_v6"` | 训练器关联的模型名 |
| `_create_model(fresh: bool) -> CNNV6Model` | 受保护方法，创建模型实例（支持 gen 参数） |
| `_convert_inputs(inputs: List[List[int]]) -> Tensor` | 受保护方法，复用 `board_vector_to_cnn_tensor` 做 batch 转换 |

---

## 4. 文件结构

```
ai/models/cnn_v6/
├── __init__.py         # 导出 CNNV6Model, CNNV6Trainer
├── model.py            # CNNV6Network + CNNV6Model
└── trainer.py          # CNNV6Trainer
```

模型保存路径（多代）：`ai/models/cnn_v6/gen{i}/cnn_v6.pth`

---

## 5. 集成点

### 5.1 registry.py（已有文件修改）

在 `MODEL_REGISTRY` 末尾新增一条：

```
"cnn_v6": RegistryEntry(
    "ai.models.cnn_v6.model.CNNV6Model",
    "ai.models.cnn_v6.trainer.CNNV6Trainer",
    True),
```

### 5.2 models/__init__.py（已有文件修改）

新增导入和导出：

```
from .cnn_v6 import CNNV6Model, CNNV6Trainer
```

在 `__all__` 中追加 `"CNNV6Model", "CNNV6Trainer"` 和模块文档说明。

### 5.3 tests/ai/test_cnn_models.py（已有文件修改）

新增两个测试类：

- `TestCNNV6Network(CNNNetworkSharedMixin, unittest.TestCase)`：`NETWORK_CLS = CNNV6Network`
- `TestCNNV6Model(CNNModelSharedMixin, unittest.TestCase)`：`MODEL_CLS = CNNV6Model` 等

详情见 [test_design.md](test_design.md)。

---

## 6. 设计决策记录

| 决策 | 理由 |
|------|------|
| 残差块不封装为独立 nn.Module 子类 | 避免引入不必要的公开接口；v6 仅此一处使用，内联实现更简洁 |
| BN 置于 Conv 之后、ReLU 之前（Conv→BN→ReLU） | 业界标准做法，BN 对卷积输出的线性激活做归一化后再非线性变换 |
| 残差 shortcut 加在第二个 BN 之后、ReLU 之前 | 标准 Pre-activation 残差块变体，保证加法操作在归一化后进行 |
| 第一层 Conv1 也有 BN | 统一各层行为，稳定初始输入分布；仅增加 64 参数 |
| 参数校验委托给 PyTorch 内置检查 | Conv2d/BN 构造时即校验参数合法性，无需额外代码 |
