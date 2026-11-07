# cnn_v12 接口设计

> 意图 61 | 只定义对外暴露的接口。私有方法不在本文档设计范围内。

---

## 0. 模块与文件

| 文件 | 定义 |
|------|------|
| `src/ai/models/cnn_v12/model.py` | `CNNV12Network`、`CNNV12Model` |
| `src/ai/models/cnn_v12/trainer.py` | `CNNV12Trainer` |
| `src/ai/models/cnn_v12/__init__.py` | 导出三者，`__all__` 列出 |

`__init__.py` 必须具备：

```python
from .model import CNNV12Model, CNNV12Network
from .trainer import CNNV12Trainer

__all__ = ["CNNV12Model", "CNNV12Network", "CNNV12Trainer"]
```

---

## 1. CNNV12Network

`class CNNV12Network(torch.nn.Module)`

128 通道 / 4 残差块的双头卷积网络。架构固定，不提供超参。

### 1.1 类常量

| 名称 | 类型 | 值 | 说明 |
|------|------|----|------|
| `CHANNELS` | `int` | `128` | 全部卷积通道数 |
| `NUM_RES_BLOCKS` | `int` | `4` | 残差块个数 |

### 1.2 构造

| 签名 | 说明 |
|------|------|
| `__init__(self)` | 构建全部层；不接受任何参数（架构固定） |

### 1.3 方法

| 签名 | 返回值 | 说明 |
|------|--------|------|
| `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]` | `(policy, value)` | `x` 形状 `(B, 3, 15, 15)`；`policy` 为 `(B, 225)` 的 LogSoftmax 对数概率（值 ≤ 0），`value` 为 `(B, 1)` 的 Tanh 输出（值域 `[-1, 1]`） |

### 1.4 契约

| 项 | 要求 |
|----|------|
| 参数量 | 恰好 `1,346,541` |
| 层清单 | `conv1(3→128,3×3,p=1)`、`bn1`、`relu1`、`res_blocks`（`ModuleList` 长度 4，元素为 `ResBlock(128)`）、策略头 5 层、价值头 6 层（详见 overview §3） |
| TorchScript | 必须能被 `torch.jit.script` 脚本化，脚本化后输出形状与 eager 一致 |
| 无状态复用 | 各 `nn.ReLU` 作为独立模块实例注册（保持层清单完整，与 cnn_v11 一致） |

---

## 2. CNNV12Model

`class CNNV12Model(CNNV11Model)`

### 2.1 类属性

| 名称 | 类型 | 值 | 说明 |
|------|------|----|------|
| `DEFAULT_MODEL_PATH` | `str` | `{cnn_v12 包目录}/cnn_v12.pth` | 基准权重路径；指定 `gen=N` 时由基类改写为 `{包目录}/gen{N}/cnn_v12.pth` |

### 2.2 属性

| 签名 | 返回值 | 说明 |
|------|--------|------|
| `name`（property） | `str` | 恒为 `"cnn_v12"` |

### 2.3 方法

本类**新增方法为零**。以下公开方法全部从 `CNNV11Model` 继承，语义不变，实施时禁止复制粘贴重写：

| 签名 | 返回值 | 说明 |
|------|--------|------|
| `predict(self, board_vector: List[int]) -> ModelPredict` | `ModelPredict(type="policy_value")` | 单样本推理。`board_vector` 为 226 维（前 225 位棋盘行展开，末位当前落子方）；返回 `policy` 为 225 维概率分布（和 ≈ 1），`value` ∈ `[-1, 1]`（当前落子方视角） |
| `predict_batch(self, board_vectors: List[List[int]]) -> List[ModelPredict]` | 等长列表 | 批量推理；空列表返回空列表 |
| `save(self, model_path: str = None) -> str` | 实际写入路径 | 保存 `state_dict`；目录不存在时自动创建 |
| `load(self, model_path: str)` | — | 加载 `state_dict` 并置 `eval()` |
| `move_to_device(self, device: torch.device = None) -> torch.device` | 最终设备 | `None` 时 CUDA 可用则 GPU，否则 CPU |
| `full_name`（property） | `str` | `gen` 非空时返回 `"cnn_v12.{gen}"`，否则 `"cnn_v12"` |

### 2.4 异常与加载行为

| 场景 | 行为 |
|------|------|
| `gen=N` 但 `gen{N}/cnn_v12.pth` 不存在 | 抛 `FileNotFoundError` |
| `model_path` 显式指定但文件不存在 | 抛 `FileNotFoundError` |
| `fresh=True` | 不加载任何权重（随机初始化），不抛异常 |
| 既不传 `gen` 也不传 `model_path`、且 `fresh=False` | 随机初始化，不抛异常（`NeuralNetworkModel.__init__` 的既有行为） |
| `gen=N` 同时传 `model_path` | 以 `model_path` 为准，同时 `DEFAULT_MODEL_PATH` 已被改写为 `gen{N}/` 路径，影响后续 `save()` 目标 |

---

## 3. CNNV12Trainer

`class CNNV12Trainer(CNNV11Trainer)`

### 3.1 属性

| 签名 | 返回值 | 说明 |
|------|--------|------|
| `model_name`（property） | `str` | 恒为 `"cnn_v12"`。用于损失曲线图文件名 `mcts_loss_logs/cnn_v12_{timestamp}.png` |

### 3.2 方法

本类**新增方法为零**。以下公开方法全部从 `DualHeadTrainer`（经 `CNNV11Trainer`）继承，实施时禁止复制粘贴重写：

| 签名 | 返回值 | 说明 |
|------|--------|------|
| `train_epochs(self, sample_dirs: List[str], epochs: int, batch_size: int, val_split: float, shuffle: bool, verbose: bool, log_every_updates: int = 0) -> List[Dict[str, float]]` | 每 epoch 的指标历史 | 全量多 epoch 训练 + 验证集评估 + 每 epoch 打印 + 自动生成曲线图 |
| `save_model(self, output_path: str) -> str` | 实际写入路径 | 保存模型权重 |

损失函数、优化器与训练循环均由基类提供，无需改造：`MSE(value) + CrossEntropy(policy)`，`AdamW(weight_decay=1e-4)`。

---

## 4. 父类契约覆盖点

以下为**必须覆盖的父类抽象契约**，不构成新增公共接口，仅为实施完整性的必要说明。

| 类 | 覆盖目标 | 行为要求 |
|----|----------|----------|
| `CNNV12Model` | `_create_network()` | 返回 `CNNV12Network()` 实例 |
| `CNNV12Trainer` | `_create_model(fresh: bool)` | 返回 `CNNV12Model`；本训练器持有 `gen` 属性时须原样透传（与 `CNNV11Trainer` 同构） |

> `CNNV12Model._prepare_input` 与 `CNNV12Trainer._convert_inputs` **不覆盖**，直接继承相对视角编码实现，保证与 cnn_v7/v8/v9/v10/v11 编码一致。

---

## 5. 注册表条目

`src/ai/registry.py::MODEL_REGISTRY` 新增键 `"cnn_v12"`，值为：

```python
RegistryEntry(
    "ai.models.cnn_v12.model.CNNV12Model",
    "ai.models.cnn_v12.trainer.CNNV12Trainer",
    True,
)
```

对外可通过以下已存在的接口访问，均无需修改：

| 调用 | 结果 |
|------|------|
| `get_model("cnn_v12.1")` | `CNNV12Model` 实例，`gen=1` |
| `get_model("cnn_v12")` | 抛 `ValueError`（多代模型必须指定代数） |
| `get_trainer("cnn_v12.1")` | `CNNV12Trainer` 实例 |
| `get_highest_gen("cnn_v12")` | 最高有效代数（`gen{N}/` 下存在 `.pth` 才算有效） |
| `list_models_with_full_name()` | 含 `"cnn_v12.{N}"`（仅当对应 gen 目录下有 `.pth`） |
| `create_engine("cnn_v12.1")` | `PolicyValueAiEngine`（Python MCTS；C++ 路由不在本次范围） |
