# PRD：CNN v7 新增 —— 并恢复 CNN v6 为单头策略网络

| 字段 | 值 |
|------|-----|
| intention_id | 17 |
| 版本 | v1.0 |
| 创建日期 | 2026-06-22 |
| 状态 | 待评审 |

---

## 1. 目标与背景

### 1.1 背景

意图 #15 创建了 cnn_v6：一个带 BatchNorm 和残差连接的**单头策略网络**，`predict()` 返回 225 维概率分布。随后意图 #16 给 cnn_v6 增加了价值头，使其成为**双头网络**（策略 + 价值），`predict()` 返回 `(List[float], float)` 元组。

价值头是一个重大的架构变更——它将网络从单纯的策略评估器升级为具备局面评估能力的双头网络。将这一双头架构保留在"cnn_v6"名下存在语义混淆：

- **版本号不匹配**：v6 原意（意图 #15）是"带残差+BN 的策略网络"，现在却包含了价值头
- **对比基线缺失**：v5 是纯策略网络，v6（双头）无法直接与 v5 做架构层面的"残差+BN 是否有效"的公平对比

### 1.2 目标

1. **新增 cnn_v7**：将当前 cnn_v6 的双头架构（策略头 + 价值头）"升级"为 cnn_v7，独立成新的模型版本
2. **恢复 cnn_v6**：将 cnn_v6 回退到意图 #15 定义的单头策略网络（仅策略头，无价值头）

最终效果：

| 模型 | 架构特征 | `predict()` 返回值 |
|------|---------|-------------------|
| cnn_v5 | 5 层纯卷积，无 BN，无残差 | `List[float]`（225 维） |
| **cnn_v6（恢复后）** | 5 层卷积 + BN + 2 残差块 | `List[float]`（225 维） |
| **cnn_v7（新增）** | 5 层卷积 + BN + 2 残差块 + 价值头 | `(List[float], float)`（策略 + 价值） |

### 1.3 不做的事

- 不修改 cnn_v1～v5 的任何代码和权重
- 不修改 cnn_v7 相对于当前 cnn_v6 的网络结构（纯迁移，不加新层）
- 不修改调用方代码（engine / game_manager 已通过 `isinstance(result, tuple)` 通用兼容）
- 不修改训练流程、损失函数、样本格式
- 不涉及 cnn_v6 旧权重文件的迁移或转换

---

## 2. 范围与边界

### 2.1 功能范围

| 编号 | 功能 | 说明 |
|------|------|------|
| FR-1 | 新增 CNNV7Network | 网络结构与当前 CNNV6Network 一致：策略头（LogSoftmax）+ 价值头（FC+Tanh） |
| FR-2 | 新增 CNNV7Model | 继承 NeuralNetworkModel，封装 cnn_v7 网络为可训练的模型类，`predict()` 返回 `(List[float], float)` |
| FR-3 | 新增 CNNV7Trainer | 继承 BaseTrainer，训练器适配 |
| FR-4 | 注册 cnn_v7 | 在 `MODEL_REGISTRY` 中添加 `cnn_v7` 条目，标记为多代模型 |
| FR-5 | 集成导出 | 在 `ai/models/__init__.py` 中导出 cnn_v7 相关类 |
| FR-6 | cnn_v6 移除价值头 | 从 CNNV6Network 中删除价值头相关层（value_fc、value_tanh），恢复纯策略输出 |
| FR-7 | cnn_v6 恢复 predict 接口 | `predict()` 返回值从 `(List[float], float)` 变回 `List[float]` |
| FR-8 | cnn_v6 恢复 forward 返回值 | `forward()` 仅返回策略 log-probability 张量，不再返回价值 |
| FR-9 | 测试更新 | 新增 cnn_v7 测试；修正 cnn_v6 测试以匹配恢复后的单头接口 |

### 2.2 边界

- **调用方零改动**：engine.py 和 game_manager.py 已通过 `isinstance(result, tuple)` 兼容两种返回格式，cnn_v6 恢复后和 cnn_v7 均无需修改调用方
- **权重不兼容**：
  - cnn_v6 恢复后，旧版 cnn_v6 权重文件（包含 `value_fc.weight` / `value_fc.bias`）与恢复后的单头网络不兼容
  - cnn_v7 为全新模型，从初始权重开始训练
- **多代模型**：cnn_v7 也是多代模型，权重存储路径为 `ai/models/cnn_v7/gen{i}/cnn_v7.pth`

---

## 3. 术语定义

| 术语 | 定义 |
|------|------|
| 策略头 (Policy Head) | 网络的输出分支之一，输出 225 维落子概率分布（LogSoftmax），表示在每个空位上落子的优先程度 |
| 价值头 (Value Head) | 网络的输出分支之一，输出一个 [−1, 1] 区间的标量（FC + Tanh），表示从当前落子方视角评估的预估胜率 |
| 单头网络 | 仅包含策略头的网络，`predict()` 直接返回 225 维概率列表 |
| 双头网络 | 同时包含策略头和价值头的网络，`predict()` 返回 `(策略列表, 价值标量)` 元组 |
| BatchNorm (BN) | 批归一化层 |
| 残差块 | 由两层 3×3 卷积 + BN 组成，主路径输出与输入相加后过 ReLU |

---

## 4. 功能说明

### 4.1 新增 cnn_v7

#### 4.1.1 网络结构

cnn_v7 的网络结构与**当前** cnn_v6 完全一致（双头架构）：

```
输入: (batch, 3, 15, 15)
  │
  ├─ Conv1 (3→32, 3×3) → BN → ReLU
  │
  ├─ [残差块 1] ─────────────────────┐
  │   Conv2 (32→32, 3×3) → BN → ReLU │
  │   Conv3 (32→32, 3×3) → BN ────[+]→ ReLU
  │                                  ↑ shortcut
  │
  ├─ [残差块 2] ─────────────────────┐
  │   Conv4 (32→32, 3×3) → BN → ReLU │
  │   Conv5 (32→32, 3×3) → BN ────[+]→ ReLU
  │                                  ↑ shortcut
  │
  ├─ Conv_out (32→1, 1×1)
  ├─ Flatten → 225
      ├─ LogSoftmax → 策略 (batch, 225)
      └─ FC(225→1) → Tanh → 价值 (batch, 1)

输出: (policy, value) 元组
```

#### 4.1.2 模型接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `predict` | `(board_vector: List[int]) -> (List[float], float)` | 返回元组：225 维策略概率（值域 (0,1)，总和为 1）+ 价值标量（值域 [−1, 1]） |
| `save` | `(model_path: str = None) -> str` | 保存网络 state_dict |
| `load` | `(model_path: str)` | 加载网络 state_dict |
| `name` | `-> str` | 返回 `"cnn_v7"` |
| `full_name` | `-> str` | 多代时返回 `"cnn_v7.N"`（如 `cnn_v7.1`） |

#### 4.1.3 文件结构

```
ai/models/cnn_v7/
├── __init__.py        # 包声明，导出 CNNV7Model, CNNV7Trainer
├── model.py           # CNNV7Network + CNNV7Model
└── trainer.py         # CNNV7Trainer
```

#### 4.1.4 参数量

与当前 cnn_v6 一致，约 **38,467** 参数（原 cnn_v5 约 38,241 + BN 参数约 320 + 价值头 FC 参数 226）。

### 4.2 恢复 cnn_v6

#### 4.2.1 网络结构变更

**变更前**（当前 cnn_v6，双头）：

```
输入 → [Conv+BN+残差块] → Conv_out → Flatten → 225
    ├─ LogSoftmax → 策略 (batch, 225)
    └─ FC(225→1) → Tanh → 价值 (batch, 1)
```

**变更后**（恢复后，单头）：

```
输入 → [Conv+BN+残差块] → Conv_out → Flatten → 225
    └─ LogSoftmax → 策略 (batch, 225)
```

具体删除内容：

| 删除项 | 说明 |
|--------|------|
| `value_fc`（nn.Linear(225, 1)） | 价值头全连接层权重 + 偏置 |
| `value_tanh`（nn.Tanh()） | 价值头激活函数 |
| forward 中价值头分支 | `value = self.value_tanh(self.value_fc(x))` |
| forward 返回值元组包装 | 从 `return policy, value` 变为 `return policy` |
| predict 返回值元组包装 | 从返回 `(policy_list, value_float)` 变为返回 `policy_list` |

#### 4.2.2 恢复后的模型接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `predict` | `(board_vector: List[int]) -> List[float]` | 恢复为 225 维概率列表，值域 (0,1)，总和为 1 |
| `forward` | `(x: Tensor) -> Tensor` | 恢复为仅返回策略 log-probability 张量 (batch, 225) |
| 其他接口 | 不变 | `save`、`load`、`name`、`full_name` 保持不变 |

#### 4.2.3 恢复后的参数量

删除价值头后，cnn_v6 参数量回到意图 #15 定义的水平，约 **38,241** 参数。

---

## 5. 验收标准

### AC-1：cnn_v7 文件结构完整性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-1.1 | 目录存在 | `ai/models/cnn_v7/` 目录存在，包含 `__init__.py`、`model.py`、`trainer.py` |
| AC-1.2 | 模块可导入 | `from ai.models.cnn_v7 import CNNV7Model, CNNV7Trainer` 无报错 |
| AC-1.3 | 注册表可解析 | `get_model("cnn_v7.1")` 可以正确加载模型 |
| AC-1.4 | 模型名称正确 | `model.name` 返回 `"cnn_v7"`；`model.full_name` 返回 `"cnn_v7.1"` |

### AC-2：cnn_v7 网络结构正确性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-2.1 | BN 层存在 | 网络中存在 5 个 `BatchNorm2d` 层 |
| AC-2.2 | 残差连接存在 | forward 中存在两次加法操作（shortcut + main path） |
| AC-2.3 | 价值头存在 | 网络中存在 `nn.Linear(225, 1)` + `nn.Tanh()` |
| AC-2.4 | 参数量范围 | 约 38,467（范围 38,000～39,000） |
| AC-2.5 | 策略输出为 LogSoftmax | 策略分支最后一层为 `nn.LogSoftmax(dim=1)` |

### AC-3：cnn_v7 双头输出正确性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-3.1 | forward 返回元组 | `network(input_tensor)` 返回 `(policy_tensor, value_tensor)` |
| AC-3.2 | 策略输出形状 | policy_tensor 形状为 (batch, 225)，值 ≤ 0 |
| AC-3.3 | 价值输出形状和值域 | value_tensor 形状为 (batch, 1)，所有值 ∈ [−1, 1] |
| AC-3.4 | predict 返回元组 | `model.predict(board)` 返回 `(List[float], float)` 元组 |
| AC-3.5 | 策略概率总和 | 策略部分 225 维概率总和 ≈ 1（误差 < 1e-5） |

### AC-4：cnn_v6 价值头已移除

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-4.1 | value_fc 不存在 | 网络中不存在 `nn.Linear(225, 1)` |
| AC-4.2 | value_tanh 不存在 | 网络中不存在 `nn.Tanh()` |
| AC-4.3 | forward 返回单一张量 | `network(input_tensor)` 返回形状为 (batch, 225) 的张量，非元组 |
| AC-4.4 | 参数量回到单头水平 | 约 38,241（范围 38,000～38,500） |

### AC-5：cnn_v6 单头输出正确性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-5.1 | predict 返回列表 | `model.predict(board)` 返回 `list`，非元组 |
| AC-5.2 | 列表长度 225 | `len(result) == 225` |
| AC-5.3 | 元素均为 float | 列表中每个元素 `isinstance(x, float)` 为 True |
| AC-5.4 | 概率总和 | 225 维概率总和 ≈ 1（误差 < 1e-5） |
| AC-5.5 | 概率值域 | 每个概率 ∈ (0, 1) |

### AC-6：调用方兼容性

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-6.1 | cnn_v6 对战正常 | `python ai/battle.py --ai1 cnn_v6.1 --ai2 random --games 5` 正常完成 |
| AC-6.2 | cnn_v7 对战正常 | `python ai/battle.py --ai1 cnn_v7.1 --ai2 random --games 5` 正常完成 |
| AC-6.3 | engine 兼容两种返回 | engine 对 cnn_v6（列表）和 cnn_v7（元组）均正常落子 |
| AC-6.4 | game_manager 兼容两种返回 | game_manager 对 cnn_v6 和 cnn_v7 均正常记录和对战 |

### AC-7：测试覆盖

| 编号 | 验收项 | 预期结果 |
|------|--------|---------|
| AC-7.1 | cnn_v7 网络测试类存在 | `TestCNNV7Network` 存在，包含结构、输出格式、值域测试 |
| AC-7.2 | cnn_v7 模型测试类存在 | `TestCNNV7Model` 存在，包含 predict 双头输出测试 |
| AC-7.3 | cnn_v6 测试已修正 | `TestCNNV6Model` 的 predict 测试不再期望元组返回（回退到 Mixin 默认或覆盖为单头断言） |
| AC-7.4 | cnn_v6 集成测试已修正 | `test_cnn_v6_integration.py` 中 predict 测试不再期望 `(列表, 价值)` 元组 |
| AC-7.5 | 全量测试通过 | `python -m pytest tests/ -v` 所有测试通过（含新增和已有） |

---

## 6. 约束与注意事项

1. **cnn_v6 旧权重作废**：恢复后的 cnn_v6 不包含 `value_fc` 和 `value_tanh`，加载包含价值头参数的旧权重文件会因 key 不匹配而失败。这是预期行为，旧权重应被视为 cnn_v7 的前身
2. **cnn_v7 全新开始**：cnn_v7 作为新模型，从随机初始化开始训练，不与任何已有权重兼容
3. **调用方已兼容**：engine.py 和 game_manager.py 均已通过 `isinstance(result, tuple)` 做模型版本感知，cnn_v6 恢复和 cnn_v7 新增均无需修改调用方
4. **cnn_v5 不受影响**：本需求不修改 cnn_v5 及之前模型的任何代码和权重
5. **训练器行为不变**：CNNV6Trainer 恢复后仍仅训练策略头；CNNV7Trainer 当前也仅训练策略头，价值头训练由后续意图独立处理
6. **多代模型声明**：cnn_v7 在 `MODEL_REGISTRY` 中 `supports_multi_gen = True`

---

## 7. 修订记录

| 日期 | 修订内容 | 涉及章节 |
|------|---------|---------|
| 2026-06-22 | 初版创建 | — |
