# 接口变更设计

## 1. 网络输出层变更

### 1.1 FCv1Network（`ai/models/fc_v1/model.py`）

| 项目 | 修改前 | 修改后 |
| ---- | ------ | ------ |
| 属性 `self.softmax` | `nn.Softmax(dim=1)` | `nn.LogSoftmax(dim=1)` |
| forward 最后一行 | `x = self.softmax(x)` | `x = self.softmax(x)`（调用名不变，实例类型变） |

### 1.2 CNNV1Network（`ai/models/cnn_v1/model.py`）

同 §1.1，`self.softmax = nn.Softmax(dim=1)` → `nn.LogSoftmax(dim=1)`。

### 1.3 CNNV2Network（`ai/models/cnn_v2/model.py`）

同 §1.1。

### 1.4 CNNV3Network（`ai/models/cnn_v3/model.py`）

同 §1.1。

> 四个网络类的变更完全一致：仅将 `nn.Softmax(dim=1)` 替换为 `nn.LogSoftmax(dim=1)`，forward 方法代码无需改动。
>
> **属性命名决策**：保留 `self.softmax` 作为属性名而非改为 `self.log_softmax` 或 `self.output_layer`，权衡如下：
> - 保留 `softmax`：减少改动量（仅改赋值右侧），forward 调用 `self.softmax(x)` 无需改动，但语义与实际类型（LogSoftmax）不一致，可能造成阅读混淆
> - 改为 `log_softmax`：语义准确，但需同时修改 forward 中的调用代码和所有引用处
> - 改为 `output_layer`：语义中立，但需同上修改调用代码
>
> 选择保留 `softmax`，理由是本次改动范围明确且仅涉及激活函数类型替换，forward 代码无需改动可降低出错风险。开发者理解此属性为 LogSoftmax 后不会产生实际误用，因 forward 流程不变且 predict 方法已统一做 exp 还原。

---

## 2. 损失函数与辅助接口变更

### 2.1 BaseTrainer（`ai/base_trainer.py`）

| 项目 | 修改前 | 修改后 |
| ---- | ------ | ------ |
| `self.criterion` | `nn.L1Loss(reduction='none')` | `nn.KLDivLoss(reduction='none')` |
| `_masked_loss` 文档字符串 | "计算 masked L1 loss" | "计算 masked KL 散度 loss" |
| `_masked_loss` 逻辑 | 不变 | 不变（criterion 替换后自动适配） |
| `__init__` 注释 | "损失函数使用 L1Loss（MAE）" | "损失函数使用 KLDivLoss（KL 散度）" |

`_masked_loss` 方法签名不变：

```
def _masked_loss(self, outputs, targets_tensor, masks_tensor) -> torch.Tensor
```

### 2.2 compute_masked_kl_loss（`ai/sample_loader.py`）

函数重命名并重写计算逻辑：

| 项目 | 修改前 | 修改后 |
| ---- | ------ | ------ |
| 函数名 | `compute_masked_l1_loss` | `compute_masked_kl_loss` |
| 签名 | `(estimated: List[float], sample: List[float], mask: List[int]) -> float` | `(log_estimated: List[float], target: List[float], mask: List[int]) -> float` |
| 参数1含义 | 模型预测的 reward 列表 | 模型输出的 log-probability 列表 |
| 参数2含义 | sample.json 中的 reward 列表 | sample.json 中的概率分布 |

> **调用者注意**：参数1从概率值改为 log-probability，类型签名仍为 `List[float]`，无法在类型层面区分。调用者必须确保传入的是网络 LogSoftmax 输出而非 Softmax 概率，误传概率值会导致计算结果完全错误。建议在函数文档字符串中加粗标注此语义差异。
>
> **运行时防御**：为防止误用，建议在函数入口添加运行时校验——若 `log_estimated` 中存在 > 0 的元素（概率值特征，log-probability 应 ≤ 0），则 raise ValueError 并提示"应传入 log-probability 而非概率值"。此校验仅在调用时执行，不影响训练性能。
>
> **运行时防御**：为防止误用，建议在函数入口添加运行时校验——若 `log_estimated` 中存在 > 0 的元素（概率值特征，log-probability 值域为 (-∞, 0]），则 raise ValueError 并提示"应传入 log-probability 而非概率值"。此校验仅在调用时执行，不影响训练性能。

> **target 浮点偏差**：sample.json 的 output 由 `masked_softmax` 生成，mask=1 位置概率总和理论上为 1.0，但浮点运算可能导致微小偏差（如 0.9999999 或 1.0000001）。`nn.KLDivLoss` 对此容忍度良好，不影响训练结果。
| 计算公式 | `Σ\|estimated[i] - sample[i]\| / count` | `Σ target[i] * (log(target[i]) - log_estimated[i]) / count` |
| 返回值 | mask=1 位置的平均绝对误差 | mask=1 位置的平均 KL 散度 |
| 特殊值 | mask 全 0 → 0.0 | mask 全 0 → 0.0 |
| `0 * log(0)` 处理 | 不涉及 | target[i]==0 时跳过 log(0) 计算，该项贡献为 0 |
| `log(0)` 异常防御 | 不涉及 | 不直接调用 `math.log(0)`，改为 `if target[i] == 0: continue` 或 `if target[i] > 0: term = target[i] * (math.log(target[i]) - log_estimated[i])` |

### 2.3 NeuralNetworkModel.predict（`ai/base_model.py`）

| 项目 | 修改前 | 修改后 |
| ---- | ------ | ------ |
| 输出转换 | `output_tensor.squeeze(0).tolist()` | `output_tensor.squeeze(0).exp().tolist()` |

predict 方法签名和返回值语义不变：

```
def predict(self, board_vector: List[int]) -> List[float]
```

返回值仍为 225 维概率分布（值域 (0,1)，总和为 1）。

### 2.4 LR Range Test（`ai/lr_range_test.py`）

| 项目 | 修改前 | 修改后 |
| ---- | ------ | ------ |
| `criterion` | `nn.L1Loss(reduction='none')` | `nn.KLDivLoss(reduction='none')` |

`run_lr_range_test` 函数签名不变，内部损失计算流程不变（与 BaseTrainer._masked_loss 一致）。

### 2.5 样本检视器删除

`ai/models/fc_v1/sample_inspector.py` 整体删除（该工具不再需要）。同时删除其入口脚本 `ai/models/fc_v1/inspect_main.py` 和测试文件 `tests/ai/test_fc_v1_inspect.py`。

---

## 3. 外部引用更新清单

以下文件的 import 语句需同步更新：

| 文件 | 旧引用 | 新引用 |
| ---- | ------ | ------ |
| `tests/ai/test_sample_loader.py` | `from ai.sample_loader import ..., compute_masked_l1_loss, ...` | `from ai.sample_loader import ..., compute_masked_kl_loss, ...` |
| `tests/ai/test_fc_v1_trainer.py` | `from ai.sample_loader import compute_masked_l1_loss` | `from ai.sample_loader import compute_masked_kl_loss` |
| `tests/ai/test_cnn_trainers.py` | 注释提及 `compute_masked_l1_loss`（第 7 行） | 注释更新为 `compute_masked_kl_loss` |

## 4. 注释与文档字符串更新清单

以下位置的注释/文档字符串需同步更新（代码逻辑不变，仅语义描述变更）：

| 文件 | 位置 | 修改内容 |
| ---- | ---- | -------- |
| `ai/base_model.py` | 模块注释（开头第 2 行） | "所有模型输出统一为 Softmax 概率分布" → "所有模型输出层使用 LogSoftmax，predict 返回 Softmax 概率分布" |
| `ai/base_model.py` | predict 方法文档字符串 | "所有概率之和为 1（Softmax 输出）" → "所有概率之和为 1（LogSoftmax + exp 还原）" |
| `ai/base_model.py` | predict 方法行内注释 | "(值域 (0, 1)，Softmax 概率)" → "(值域 (0, 1)，exp(log-probability) 还原为概率)" |
| `ai/base_model.py` | predict 方法代码 | `output_values = output_tensor.squeeze(0).tolist()` → `output_values = output_tensor.squeeze(0).exp().tolist()` |
| `ai/base_trainer.py` | `__init__` 注释 | "损失函数使用 L1Loss（MAE）" → "损失函数使用 KLDivLoss（KL 散度）" |
| `ai/base_trainer.py` | `__init__` 代码 | `nn.L1Loss(reduction='none')` → `nn.KLDivLoss(reduction='none')` |
| `ai/base_trainer.py` | `_masked_loss` 文档字符串 | "计算 masked L1 loss" → "计算 masked KL 散度 loss" |
| `ai/lr_range_test.py` | criterion 赋值代码 | `nn.L1Loss(reduction='none')` → `nn.KLDivLoss(reduction='none')` |
| `ai/models/fc_v1/MODEL.md` | Softmax 输出层描述 | "输出层后新增 Softmax(dim=1)" → "输出层使用 LogSoftmax(dim=1)" |
| `tests/ai/test_neural_network_model.py` | DummyNetwork 属性 | `self.softmax = nn.Softmax(dim=1)` → `self.softmax = nn.LogSoftmax(dim=1)` |
（原 `sample_inspector.py` 相关注释已随文件删除，无需更新）
