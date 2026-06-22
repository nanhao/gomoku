# PRD：新增 CNN v6 —— 引入 BatchNorm 与残差连接

| 字段 | 值 |
|------|-----|
| intention_id | 15 |
| 版本 | v1.0 |
| 创建日期 | 2026-06-22 |
| 状态 | 待评审 |

---

## 1. 目标与背景

### 1.1 背景

项目当前已有一系列 CNN 模型（v1～v5），从 v3 开始采用纯卷积设计（无 FC、无 BN、无 Dropout）。cnn_v5 在 cnn_v4 基础上增加一层 3×3 卷积，形成 **5 层 Conv→ReLU 链式堆叠** 的结构，参数量约 3.8 万。

这种纯堆叠结构在网络加深时面临两个典型问题：
1. **内部协变量偏移**：各层输入的分布随训练变化，深层难以稳定训练
2. **梯度退化**：深层网络容易出现梯度消失或爆炸，导致实际有效深度不足

BatchNorm 和残差连接（Residual Connection）是解决上述问题的成熟技术，被广泛验证可以有效稳定深层网络的训练并提升表达能力。

### 1.2 目标

新增 **cnn_v6** 模型，在 cnn_v5 的网络结构基础上引入两个增强：
- **BatchNorm**：每层 Conv 后添加 BN，稳定各层输入分布
- **残差连接**：将第 2～3 层卷积组成第一个残差块，第 4～5 层卷积组成第二个残差块

cnn_v5 保持不变，作为对比基线。

### 1.3 不做的事

- 不修改 cnn_v5 及之前任何模型的代码和权重
- 不修改模型输入输出接口（`predict` 的输入格式和返回值语义不变）
- 不修改训练器基类和训练流程
- 不修改样本数据格式

---

## 2. 范围与边界

### 2.1 功能范围

| 编号 | 功能 | 说明 |
|------|------|------|
| FR-1 | 新增 CNNV6Network | 实现带 BN + 残差连接的网络结构（§3.1） |
| FR-2 | 新增 CNNV6Model | 继承 NeuralNetworkModel，封装网络为可训练的模型类 |
| FR-3 | 新增 CNNV6Trainer | 继承 BaseTrainer，训练器适配 |
| FR-4 | 注册到模型注册表 | 在 `MODEL_REGISTRY` 中添加 `cnn_v6` 条目，支持多代 |
| FR-5 | 集成到模块导出 | 在 `ai/models/__init__.py` 中导出新模型和训练器 |
| FR-6 | 新增测试用例 | 测试网络结构、模型行为、集成一致性（§5） |

### 2.2 边界

- **仅新增，不修改**：所有现有模型（v1～v5）的代码、权重、测试均保持不变
- **接口兼容**：`predict(board_vector)` 仍返回 225 维概率分布，值域 (0,1)，总和为 1
- **输入格式不变**：仍接受 (batch, 3, 15, 15) 的 4D 张量，3 通道分别为黑棋、白棋、当前方
- **多代模型**：cnn_v6 也是多代模型，权重存储路径为 `ai/models/cnn_v6/gen{i}/cnn_v6.pth`

---

## 3. 术语定义

| 术语 | 定义 |
|------|------|
| BatchNorm (BN) | 批归一化，对每个 mini-batch 的激活值做标准化（减均值除标准差），再通过可学习的 γ/β 做缩放和平移 |
| 残差连接 (Residual Connection) | 将网络块的输入直接加到块的输出上，形成 shortcut 路径。公式：`output = ReLU( F(x) + x )` |
| 残差块 (Residual Block) | 由若干卷积层 + BN 组成，最后将输入加到输出并过激活函数。本设计中每个残差块包含 2 层 3×3 卷积 |
| Conv1×1 | 核大小为 1×1 的卷积层，用于通道压缩（32→1），替代传统 FC 层 |
| LogSoftmax | Softmax 的对数形式，输出 log-probability 分布 |

---

## 4. 功能说明

### 4.1 网络结构

#### 4.1.1 整体架构

cnn_v6 网络包含 **5 层 3×3 卷积 + 1 层 1×1 输出卷积**，结构如下：

```
输入: (batch, 3, 15, 15)
  │
  ├─ Conv1 (3→32, 3×3) → BN → ReLU
  │
  ├─ [残差块 1] ─────────────────────┐
  │   Conv2 (32→32, 3×3) → BN → ReLU │
  │   Conv3 (32→32, 3×3) → BN ────[+]→ ReLU
  │                                  ↑
  │                           shortcut (块输入)
  │
  ├─ [残差块 2] ─────────────────────┐
  │   Conv4 (32→32, 3×3) → BN → ReLU │
  │   Conv5 (32→32, 3×3) → BN ────[+]→ ReLU
  │                                  ↑
  │                           shortcut (块输入)
  │
  ├─ Conv_out (32→1, 1×1)
  ├─ Flatten → 225
  └─ LogSoftmax(dim=1)

输出: (batch, 225) log-probability
```

#### 4.1.2 与 cnn_v5 的逐层对比

| 序号 | cnn_v5 | cnn_v6 | 变更说明 |
|------|--------|--------|---------|
| 1 | Conv(3→32) → ReLU | Conv(3→32) → **BN** → ReLU | 新增 BN |
| 2 | Conv(32→32) → ReLU | Conv(32→32) → **BN** → ReLU | 新增 BN |
| 3 | Conv(32→32) → ReLU | Conv(32→32) → **BN** → **[+残差] → ReLU** | 新增 BN + 残差连接（与第 2 层输入相加） |
| 4 | Conv(32→32) → ReLU | Conv(32→32) → **BN** → ReLU | 新增 BN |
| 5 | Conv(32→32) → ReLU | Conv(32→32) → **BN** → **[+残差] → ReLU** | 新增 BN + 残差连接（与第 4 层输入相加） |
| 6 | Conv(32→1, 1×1) | Conv(32→1, 1×1) | 不变 |
| 7 | Flatten → 225 | Flatten → 225 | 不变 |
| 8 | LogSoftmax | LogSoftmax | 不变 |

#### 4.1.3 残差块详述

**第一个残差块**（由 Conv2 + Conv3 组成）：
- 输入：Conv1+BN+ReLU 的输出（形状 (batch, 32, 15, 15)）
- 主路径：Conv2 → BN → ReLU → Conv3 → BN
- 快捷路径：将输入直接加到主路径的 BN 输出上
- 输出：ReLU(主路径输出 + 输入)

**第二个残差块**（由 Conv4 + Conv5 组成）：
- 输入：第一残差块 ReLU 的输出（形状 (batch, 32, 15, 15)）
- 主路径：Conv4 → BN → ReLU → Conv5 → BN
- 快捷路径：将输入直接加到主路径的 BN 输出上
- 输出：ReLU(主路径输出 + 输入)

> **关键约束**：残差连接要求输入和输出的形状一致。由于两个残差块内部的卷积层均使用 padding=1、stride=1、kernel=3×3，通道数保持 32 不变，空间尺寸保持 15×15 不变，因此可以直接相加，无需 1×1 投影层。

#### 4.1.4 参数量估算

| 层 | 参数组成 | 参数量 |
|---|---------|--------|
| Conv1 | 3×32×3×3 + 32 | 896 |
| BN1 | 32 + 32 (γ, β) | 64 |
| Conv2 | 32×32×3×3 + 32 | 9,248 |
| BN2 | 32 + 32 | 64 |
| Conv3 | 32×32×3×3 + 32 | 9,248 |
| BN3 | 32 + 32 | 64 |
| Conv4 | 32×32×3×3 + 32 | 9,248 |
| BN4 | 32 + 32 | 64 |
| Conv5 | 32×32×3×3 + 32 | 9,248 |
| BN5 | 32 + 32 | 64 |
| Conv_out | 32×1×1×1 + 1 | 33 |
| **总计** | | **约 3.8 万** |

相比 cnn_v5（约 37,921 参数），cnn_v6 增加约 320 参数（5 个 BN 层的 γ 和 β），参数量仍处于同一数量级。

### 4.2 模型接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `predict` | `(board_vector: List[int]) -> List[float]` | 不变。返回 225 维概率分布，值域 (0,1)，总和为 1 |
| `save` | `(model_path: str = None) -> str` | 不变。保存网络 state_dict |
| `load` | `(model_path: str)` | 不变。加载网络 state_dict |
| `name` | `-> str` | 返回 `"cnn_v6"` |
| `full_name` | `-> str` | 多代时返回 `"cnn_v6.N"`（如 `cnn_v6.1`） |

### 4.3 集成点

#### 4.3.1 模型注册表

在 `ai/registry.py` 的 `MODEL_REGISTRY` 中新增：

```
"cnn_v6": RegistryEntry(
    "ai.models.cnn_v6.model.CNNV6Model",
    "ai.models.cnn_v6.trainer.CNNV6Trainer",
    True  # 支持多代
)
```

#### 4.3.2 模块导出

在 `ai/models/__init__.py` 中添加导出：

```
from .cnn_v6 import CNNV6Model, CNNV6Trainer
```

#### 4.3.3 文件结构

```
ai/models/cnn_v6/
├── __init__.py        # 包声明，导出 CNNV6Model, CNNV6Trainer
├── model.py           # CNNV6Network + CNNV6Model
└── trainer.py         # CNNV6Trainer
```

---

## 5. 验收标准

### AC-1：文件结构完整性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-1.1 | 目录存在 | `ai/models/cnn_v6/` 目录存在，包含 `__init__.py`、`model.py`、`trainer.py` |
| AC-1.2 | 模块可导入 | `from ai.models.cnn_v6 import CNNV6Model, CNNV6Trainer` 无报错 |
| AC-1.3 | 注册表可解析 | `get_model("cnn_v6.1")` 可以正确加载模型 |

### AC-2：网络结构正确性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-2.1 | 包含 BN 层 | 网络中存在至少 1 个 `BatchNorm2d` 层 |
| AC-2.2 | 卷积通道数 | 5 层 3×3 卷积：3→32、32→32、32→32、32→32、32→32；1 层 1×1 卷积：32→1 |
| AC-2.3 | 无 FC 层 | 网络中不存在 `nn.Linear` |
| AC-2.4 | 无 Dropout 层 | 网络中不存在 `nn.Dropout` |
| AC-2.5 | 残差连接存在 | forward 中存在两次加法操作（shortcut + main path），分别位于两个残差块 |
| AC-2.6 | 参数量范围 | 约 3.8 万（范围 37,000～41,000） |
| AC-2.7 | 输出层为 LogSoftmax | 最后一层激活为 `nn.LogSoftmax(dim=1)` |

### AC-3：模型功能正确性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-3.1 | predict 输出格式 | 225 维列表，值域 (0,1)，总和 ≈ 1（误差 < 1e-5） |
| AC-3.2 | predict 可被已占位过滤 | 对已落子位置，predict 概率不受模型新结构影响（仍由调用方 mask 处理） |
| AC-3.3 | save/load 往返一致 | 保存 → 重新加载 → 对同一输入 predict 结果一致 |
| AC-3.4 | forward 输出 ≤ 0 | 直接调用 network.forward 返回 log-probability（值 ≤ 0） |

### AC-4：训练可用性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-4.1 | 小规模训练不报错 | 用少量样本训练 5 个 epoch，loss 逐步下降，无 NaN |
| AC-4.2 | 梯度正常流动 | 所有卷积层和 BN 层的参数梯度不为 None（训练后检查 `param.grad`） |
| AC-4.3 | 训练中 BN 处于训练模式 | 训练时 `self.training = True`，BN 使用 batch 统计量 |

### AC-5：测试覆盖

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-5.1 | 网络结构测试类 | `TestCNNV6Network(CNNNetworkSharedMixin)` 存在，包含 AC-2 对应测试方法 |
| AC-5.2 | 模型行为测试类 | `TestCNNV6Model(CNNModelSharedMixin)` 存在 |
| AC-5.3 | 全量测试通过 | `python -m pytest tests/ -v` 所有测试通过（含新增和已有） |

---

## 6. 约束与注意事项

1. **BN 训练/推理模式**：训练时 BN 使用 mini-batch 统计量，推理时使用全局移动平均。由 PyTorch 的 `model.train()` / `model.eval()` 自动切换，确保 predict 调用前网络处于 eval 模式
2. **旧权重不兼容**：cnn_v6 的网络结构与 cnn_v5 不同（多了 BN 参数、不同的层连接方式），无法加载 cnn_v5 的 `.pth` 权重文件
3. **残差块保持空间尺寸不变**：两个残差块内部的卷积均使用 `padding=1`、`kernel_size=3`、`stride=1`，确保 shortcut 直接相加无需维度匹配
4. **BN 动量默认值**：使用 PyTorch 默认的 momentum=0.1，无需特殊调整
5. **cnn_v5 完全保留**：本需求仅新增 cnn_v6，不对 cnn_v5 做任何修改，包括代码、权重、测试用例

---

## 7. 修订记录

| 日期 | 修订内容 | 涉及章节 |
|------|---------|---------|
| 2026-06-22 | 初版创建 | — |
