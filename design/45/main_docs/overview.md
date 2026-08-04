# CNN v11 双头模型设计

> 意图 45 | 关联模型：cnn_v11 | 基座模型：cnn_v9

---

## 1. 概述

在 cnn_v9 基础上，参照 AlphaGo Zero 论文网络结构，创建更大规模的双头模型 cnn_v11。

**核心变动（对照 cnn_v9）**：

| 组件 | cnn_v9 | cnn_v11 |
|------|--------|---------|
| 残差块数 | 2 | **7** |
| 卷积通道数 | 48 | **256** |
| 策略头 | conv(48→1) → flatten(225) → FC(225→64) → ReLU → FC(64→225) → LogSoftmax | conv(256→**2**) → BN → ReLU → flatten(**450**) → FC(**450→225**) → LogSoftmax |
| 价值头 | conv(48→1) → flatten(225) → FC(225→64) → ReLU → FC(64→1) → Tanh | conv(256→1) → BN → ReLU → flatten(225) → FC(225→**256**) → ReLU → FC(**256→1**) → Tanh |
| 参数量 | ~12.8 万 | **~843 万**（约 66×） |

**与现有体系关系**：完全复用 `DualHeadTrainer` / `PolicyValueAiEngine` / `MCTSSearcher` 基础设施，仅新增模型文件 + 注册表条目。自对弈脚本、训练脚本、对战脚本均通过 registry 自动兼容。

---

## 2. 目录结构

```
src/ai/models/cnn_v11/
├── __init__.py        # 新增：导出 CNNV11Model, CNNV11Network, CNNV11Trainer
├── model.py           # 新增：CNNV11Network(nn.Module) + CNNV11Model(NeuralNetworkModel)
└── trainer.py         # 新增：CNNV11Trainer(DualHeadTrainer)

src/ai/
└── registry.py        # 修改：MODEL_REGISTRY 新增 cnn_v11 条目
```

---

## 3. 网络结构

### 3.1 共享卷积体

与 cnn_v9 共享相同的残差块结构（每块 2 层 Conv + BN + shortcut），块数从 2 增加到 7，通道数从 48 提升到 256：

```
输入 (batch, 3, 15, 15)
  ↓
Conv1(3→256, 3×3, pad=1) → BN → ReLU
  ↓
┌─ 残差块 1 ──────────────────────────┐
│ Conv2(256→256, 3×3, pad=1) → BN → ReLU  │
│ Conv3(256→256, 3×3, pad=1) → BN        │
│ + shortcut (identity)                  │
└→ ReLU ────────────────────────────────┘
  ↓
  ... (残差块 2~6，结构同残差块 1，编号 conv4~conv15)
  ↓
┌─ 残差块 7 ──────────────────────────┐
│ Conv14(256→256, 3×3, pad=1) → BN → ReLU  │
│ Conv15(256→256, 3×3, pad=1) → BN        │
│ + shortcut (identity)                  │
└→ ReLU ────────────────────────────────┘
  ↓
共享特征 (batch, 256, 15, 15)
```

**总计**：1 层初始 Conv + 7×2 = **15 层卷积**（Conv1~Conv15），全部 256 通道。

### 3.2 策略头

参照 AlphaGo Zero 策略头结构，使用 2 通道 1×1 卷积增加空间特征，单层 FC 直接映射到 225 维：

```
cnnv_out(256→2, 1×1, pad=0)
  ↓ (batch, 2, 15, 15)
BN → ReLU
  ↓
Flatten → (batch, 450)
  ↓
FC(450→225)
  ↓
LogSoftmax → (batch, 225)
```

**设计决策**：相比 cnn_v9 的策略头（两层 FC + hidden_size=64），cnn_v11 的策略头仅用一层 FC。AGZ 论文的 policy head 即是 `conv → BN → ReLU → FC → softmax` 结构，无中间隐藏层。256 通道共享特征已提供足够非线性变换能力，策略头保持简洁。

### 3.3 价值头

```
value_cnnv_out(256→1, 1×1, pad=0)
  ↓ (batch, 1, 15, 15)
BN → ReLU
  ↓
Flatten → (batch, 225)
  ↓
FC(225→256) → ReLU
  ↓
FC(256→1)
  ↓
Tanh → (batch, 1)，值域 [-1, 1]
```

**与 cnn_v9 的区别**：隐藏层维度从可配置的 `hidden_size`（默认 64）固定为 256，移除可配置性以简化接口。

### 3.4 前向传播返回值

`forward(x)` → `(policy, value)` 元组：
- `policy`: `(batch, 225)` LogSoftmax log-probability
- `value`: `(batch, 1)` Tanh 胜率估值，值域 [-1, 1]

### 3.5 参数量估算

| 组件 | 参数量（约） |
|------|-------------|
| Conv1 (3→256) + BN | 7,100 |
| 每个残差块（2×Conv + 2×BN） | 118.0 万 |
| **7 个残差块小计** | **~826.1 万** |
| **卷积体小计（含 Conv1）** | **~826.8 万** |
| Policy conv_out (256→2) + BN | 500 |
| Policy FC (450→225) | 101,500 |
| **策略头小计** | **~10.2 万** |
| Value conv_out (256→1) + BN | 260 |
| Value FC1 (225→256) | 57,900 |
| Value FC2 (256→1) | 260 |
| **价值头小计** | **~5.8 万** |
| **总计** | **~843 万** |

> 参数量分布：卷积体占 98%，策略头 1.2%，价值头 0.7%。训练时 GPU 显存主要消耗在 15 层 256 通道的特征图及其梯度。

---

## 4. 类设计

### 4.1 CNNV11Network (nn.Module)

| 项 | 内容 |
|----|------|
| 位置 | `src/ai/models/cnn_v11/model.py` |
| 父类 | `torch.nn.Module` |
| 输入 | `(batch, 3, 15, 15)` |
| 输出 | `(policy: (batch, 225), value: (batch, 1))` |
| 对外方法 | `forward(x: Tensor) -> Tuple[Tensor, Tensor]` |

**层定义**（`__init__` 中注册）：

| 属性名 | 类型 | 形状 |
|--------|------|------|
| `conv1` | `nn.Conv2d` | 3→256, k=3, p=1 |
| `bn1` | `nn.BatchNorm2d` | 256 |
| `relu1` | `nn.ReLU` | — |
| `conv2`~`conv15` | `nn.Conv2d` | 256→256, k=3, p=1（14 层） |
| `bn2`~`bn15` | `nn.BatchNorm2d` | 256（14 层） |
| `relu2`~`relu15` | `nn.ReLU` | —（14 层，仅残差块内第一层 ReLU 和块输出 ReLU） |
| `conv_out` | `nn.Conv2d` | 256→2, k=1, p=0 |
| `policy_bn` | `nn.BatchNorm2d` | 2 |
| `policy_relu` | `nn.ReLU` | — |
| `policy_fc` | `nn.Linear` | 450→225 |
| `log_softmax` | `nn.LogSoftmax(dim=1)` | — |
| `value_conv_out` | `nn.Conv2d` | 256→1, k=1, p=0 |
| `value_bn` | `nn.BatchNorm2d` | 1 |
| `value_relu1` | `nn.ReLU` | — |
| `value_fc1` | `nn.Linear` | 225→256 |
| `value_relu2` | `nn.ReLU` | — |
| `value_fc2` | `nn.Linear` | 256→1 |
| `value_tanh` | `nn.Tanh` | — |

> 残差块内 ReLU 使用规则（每块 2 层 Conv + BN）：
> - 块内第一层 Conv 后：BN → **ReLU**（如 relu2、relu4、...、relu14）
> - 块内第二层 Conv 后：仅 BN（无独立 ReLU），与 shortcut 相加后 → **ReLU**（如 relu3、relu5、...、relu15）
> - 即每个残差块使用 2 个 ReLU 实例，7 块共 14 个 ReLU

**forward 流程**：参见 [network_forward.puml](network_forward.puml)

**残差块实现建议**：7 个残差块结构高度重复，建议在 `CNNV11Network.__init__` 中使用 `nn.ModuleList` 存储块级子模块，减少样板代码。每个块封装为：`[Conv2d, BN, ReLU, Conv2d, BN, ReLU]`，forward 中迭代 ModuleList，每块执行 `out = relu(block[4](block[2](relu(block[1](block[0](x))))) + x)`。注意 `relu2`~`relu15` 作为 `nn.ReLU` 实例不可复用（其内部无状态，但每个块需独立引用以满足设计表）。

### 4.2 CNNV11Model (NeuralNetworkModel)

| 项 | 内容 |
|----|------|
| 位置 | `src/ai/models/cnn_v11/model.py` |
| 父类 | `NeuralNetworkModel`（间接实现 `BaseModel`） |
| `name` | `"cnn_v11"` |
| `DEFAULT_MODEL_PATH` | `{model_dir}/cnn_v11.pth` |
| 多代 | `supports_multi_gen = True` |

**对外接口**（继承自 NeuralNetworkModel，覆盖部分方法）：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict` | `board_vector: List[int]` | `ModelPredict(type=TYPE_POLICY_VALUE)` | 单样本推理，返回 policy + value |
| `predict_batch` | `board_vectors: List[List[int]]` | `List[ModelPredict]` | 批量推理 |
| `save` | `model_path: str` | `str` | 保存权重 |
| `load` | `model_path: str` | — | 加载权重 |
| `move_to_device` | `device: torch.device` | `torch.device` | 迁移到 GPU/CPU |

**内部实现**（子类覆盖的抽象方法）：

| 方法 | 说明 |
|------|------|
| `_create_network()` → `CNNV11Network` | 创建网络实例 |
| `_prepare_input(board_vector, device)` → `Tensor(1,3,15,15)` | 使用 `board_vector_to_cnn_tensor_relative`（相对视角编码） |

### 4.3 CNNV11Trainer (DualHeadTrainer)

| 项 | 内容 |
|----|------|
| 位置 | `src/ai/models/cnn_v11/trainer.py` |
| 父类 | `DualHeadTrainer` |
| `model_name` | `"cnn_v11"` |

**对外接口**（继承自 DualHeadTrainer）：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `train_epochs` | `sample_dirs, epochs, batch_size, val_split, shuffle, verbose` | `List[Dict]` | 完整训练流程 |
| `save_model` | `output_path: str` | `str` | 保存模型 |

**内部实现**（子类覆盖的抽象方法）：

| 方法 | 说明 |
|------|------|
| `_create_model(fresh)` → `CNNV11Model` | 创建模型实例 |
| `_convert_inputs(states)` → `Tensor(N,3,15,15)` | 使用 `board_vector_to_cnn_tensor_relative` |

---

## 5. 注册表集成

在 `src/ai/registry.py` 的 `MODEL_REGISTRY` 中新增：

```python
"cnn_v11": RegistryEntry(
    "ai.models.cnn_v11.model.CNNV11Model",
    "ai.models.cnn_v11.trainer.CNNV11Trainer",
    True
)
```

- `supports_multi_gen = True`：支持多代模型（如 `cnn_v11.1`、`cnn_v11.2`）
- 无需修改 `get_model` / `get_trainer` / `create_engine` / `list_models_with_full_name` 等函数

---

## 6. 引擎路由

cnn_v11 的 `predict()` 返回 `ModelPredict(type=TYPE_POLICY_VALUE)`：

```
create_engine("cnn_v11.1")
  → model.predict([0]*226) → type=TYPE_POLICY_VALUE
  → AIEngineBuilder.build_policy_value(model, num_simulations=800, ...)
  → PolicyValueAiEngine
```

**无需修改 `create_engine` 或 `AIEngineBuilder`**。

---

## 7. 交互流程图

| 图 | 文件 | 说明 |
|----|------|------|
| 网络前向传播流程 | [network_forward.puml](network_forward.puml) | 展示 forward() 中数据在各层间的流转 |
| 类关系图 | [class_diagram.puml](class_diagram.puml) | CNNV11 类与现有基础设施的关系 |

---

## 8. 测试用例

参见 [test_cases.md](test_cases.md)

---

## 9. 非功能决策

| 决策 | 内容 |
|------|------|
| 代码隔离 | 仅新增 `cnn_v11/` 包 + 注册表一行；不修改任何现有文件（除 registry.py） |
| 训练流程 | 完全复用 DualHeadTrainer，损失函数 `MSE(value) + CrossEntropy(policy)`，AdamW 优化器 |
| 推理流程 | 完全复用 PolicyValueAiEngine + MCTSSearcher，无需适配 |
| 编码方式 | 相对视角编码（`board_vector_to_cnn_tensor_relative`），与 cnn_v7/v8/v9/v10 一致 |
| 价值头隐藏维度 | 固定 256，不提供 `hidden_size` 可配置参数。理由：参照 AGZ 固定结构，避免超参搜索复杂度 |
| 兼容性 | 权重文件与 cnn_v9 不兼容（通道数不同、残差块数不同，无法加载 cnn_v9 权重），需 `fresh=True` 从零训练 |

---

## 10. 术语

| 术语 | 定义 |
|------|------|
| 双头 (dual-head) | 同时输出策略分布和价值评估的模型结构 |
| 共享卷积体 | 策略头和价值头共用的卷积特征提取层 |
| 残差块 | 两层 Conv + BN + 跳跃连接的结构单元 |
| 相对视角编码 | 以当前落子方视角编码棋盘（我/对方/颜色），保证颜色互换不变性 |
| LogSoftmax | 策略头输出 log-probability（值 ≤0），训练时直接用于 CrossEntropy |
