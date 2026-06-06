# 设计文档：训练损失函数从 L1 迁移到 KL 散度

## 1. 元信息

| 项目       | 值                           |
| ---------- | ---------------------------- |
| intention_id | 4                          |
| PRD        | `requirements/4/PRD.md`     |
| 文档版本   | v1.0                         |
| 创建日期   | 2026-06-05                   |

---

## 2. 设计概览

本次改动将训练损失函数从 L1 Loss 统一替换为 KL 散度，同时将模型网络输出层从 Softmax 改为 LogSoftmax 以适配 `nn.KLDivLoss` 的输入要求。

改动分为四条主线：

1. **网络输出层变更**：所有模型的 forward 最后一层 `nn.Softmax(dim=1)` → `nn.LogSoftmax(dim=1)`
2. **损失函数替换**：`nn.L1Loss` → `nn.KLDivLoss`，涉及训练器、LR Range Test、纯 Python 辅助函数
3. **predict 兼容**：`NeuralNetworkModel.predict` 将 log-probability 还原为 probability
4. **样本检视器删除**：删除 `sample_inspector.py` 及入口脚本 `inspect_main.py`、测试文件 `test_fc_v1_inspect.py`（该工具不再需要）

交互图参见 [training_loss_flow.puml](./training_loss_flow.puml)

---

## 3. 变更范围

### 3.1 需修改的文件

| 文件 | 变更内容 | 子文档引用 |
| ---- | -------- | ---------- |
| `ai/base_model.py` | `NeuralNetworkModel.predict` 中对网络输出取 exp 还原概率 | [interface_changes.md](./main_docs/interface_changes.md) §2.3 |
| `ai/base_trainer.py` | `self.criterion` 从 `nn.L1Loss` 改为 `nn.KLDivLoss`；`_masked_loss` 文档字符串更新 | [interface_changes.md](./main_docs/interface_changes.md) §2.1 |
| `ai/sample_loader.py` | `compute_masked_l1_loss` → `compute_masked_kl_loss` | [interface_changes.md](./main_docs/interface_changes.md) §2.2 |
| `ai/lr_range_test.py` | `criterion` 从 `nn.L1Loss` 改为 `nn.KLDivLoss` | [interface_changes.md](./main_docs/interface_changes.md) §2.4 |
| `ai/models/fc_v1/model.py` | `FCv1Network`: Softmax → LogSoftmax | [interface_changes.md](./main_docs/interface_changes.md) §1.1 |
| `ai/models/cnn_v1/model.py` | `CNNV1Network`: Softmax → LogSoftmax | [interface_changes.md](./main_docs/interface_changes.md) §1.2 |
| `ai/models/cnn_v2/model.py` | `CNNV2Network`: Softmax → LogSoftmax | [interface_changes.md](./main_docs/interface_changes.md) §1.3 |
| `ai/models/cnn_v3/model.py` | `CNNV3Network`: Softmax → LogSoftmax | [interface_changes.md](./main_docs/interface_changes.md) §1.4 |
| `ai/models/fc_v1/sample_inspector.py` | **删除**（该工具不再需要） | — |
| `ai/models/fc_v1/inspect_main.py` | **删除**（依赖 SampleInspector，删除后无意义） | — |
| `design/fc_v1_design.md` | §1.2 损失函数行（L1 Loss）更新为 KL 散度；§12 SampleInspector 相关描述标记为已废弃 | — |
| `design/cnn_design.md` | Softmax 输出层描述更新为 LogSoftmax | — |
| `design/cnn_v2_design.md` | Softmax 输出层表格更新为 LogSoftmax | — |
| `design/cnn_v3_design.md` | Softmax 输出层描述更新为 LogSoftmax | — |
| `design/sample_visualizer_design.md` | 第 5 行 SampleInspector 引用更新 | — |
| `ai/models/fc_v1/MODEL.md` | Softmax 输出层描述更新为 LogSoftmax | — |
| `README.md` | 第 272 行 inspect_main.py 引用删除 | — |

### 3.2 需更新的测试文件

| 测试文件 | 变更内容 | 子文档引用 |
| -------- | -------- | ---------- |
| `tests/ai/test_sample_loader.py` | `TestComputeMaskedL1Loss` → `TestComputeMaskedKLLoss`，测试逻辑适配 KL 散度 | [test_design.md](./main_docs/test_design.md) §1 |
| `tests/ai/test_fc_v1_trainer.py` | `compute_masked_l1_loss` 引用 → `compute_masked_kl_loss`；masked_loss 一致性测试适配 | [test_design.md](./main_docs/test_design.md) §2 |
| `tests/ai/test_fc_v1_inspect.py` | **删除**（随 SampleInspector 一同删除） | — |
| `tests/ai/test_fc_v1_softmax_input.py` | `_masked_loss` 结果断言适配 KL 散度 | [test_design.md](./main_docs/test_design.md) §4 |
| `tests/ai/test_neural_network_model.py` | DummyNetwork 的 `self.softmax` 从 `nn.Softmax` 改为 `nn.LogSoftmax`；predict 输出仍为概率分布的断言不变，新增 exp(log_softmax) = softmax 的验证 | [test_design.md](./main_docs/test_design.md) §5 |
| `tests/unit/test_lr_range_test.py` | 无需改测试逻辑，因 LR range test 只验证 loss 为有限数、lr 递增等不变性质 | — |

---

## 4. 关键设计决策

### 4.1 KLDivLoss 的 log_target 参数

PyTorch `nn.KLDivLoss` 默认 `log_target=False`（自 PyTorch 1.8+），即 target 为普通概率值。这与我们的 sample.json 数据格式（Softmax 概率分布）一致，**无需额外对 target 取 log**。

### 4.2 predict 方法的 exp 还原

`NeuralNetworkModel.predict` 当前直接将 `output_tensor.squeeze(0).tolist()` 返回。改为 LogSoftmax 后，输出变为 log-probability（负数），需取 exp 还原：

```
修改前：output_values = output_tensor.squeeze(0).tolist()
修改后：output_values = output_tensor.squeeze(0).exp().tolist()
```

调用方（engine、game_manager、move_sampler）无需任何修改，因为 predict 返回值语义不变。

### 4.3 _masked_loss 的兼容性

`BaseTrainer._masked_loss` 当前流程：

```
loss = self.criterion(outputs, targets)  # 逐元素损失
loss = loss * masks_tensor               # 应用 mask
loss = loss.sum() / masks_tensor.sum()   # mask=1 位置求均值
```

改为 KLDivLoss 后流程完全不变，因为 `reduction='none'` 仍返回逐元素损失。唯一差异是 criterion 的计算公式从 L1 变为 KL 散度。

### 4.4 LogSoftmax 的数值稳定性优势

选择 `nn.LogSoftmax` 而非先 Softmax 再取 log，核心原因是数值稳定性。Softmax 计算涉及 `exp(x)` 操作，当某些 logit 值远大于其他位置时，exp 可能溢出；而 LogSoftmax 通过先减去最大值再计算 `log(exp(x - max))`，避免了中间步骤的数值溢出/下溢。PRD §5.5 明确要求使用 LogSoftmax，而非 Softmax + log 的两步操作。

### 4.5 全零 target 的防御

当 target 中 mask=1 位置全为 0 时（极端情况），KL 散度形式上无定义。需区分两层防御：

**PyTorch KLDivLoss 层面**：`nn.KLDivLoss(reduction='none')` 在 `log_target=False` 模式下的计算公式为 `target * (log(target) - input)`。在 IEEE 754 浮点标准下，`0 * log(0)` 即 `0 * (-inf)` 理论上为 NaN，但 PyTorch 内部实现通过 `target.clamp(min=eps)` 机制（eps 为极小正值，如 1e-10）避免了 `log(0) = -inf`，使得 `0 * log(clamp(0)) \approx 0 * log(eps) \approx 0`，不会产生 NaN。因此 PyTorch 侧**无需额外防御**。

**纯 Python compute_masked_kl_loss 层面**：直接调用 `math.log(0)` 会抛 ValueError，必须显式处理。约定：当 `target[i] == 0` 时，该位置的 KL 散度贡献为 0，跳过 `log(0)` 计算（见 interface_changes.md \u00a72.2）。

**_masked_loss 层面**：为满足 PRD §7.1 要求的"合理回退行为"，**必须在** `_masked_loss` 中增加防御——当 mask=1 位置的 target 总和为 0 时直接返回 0.0，避免极端情况下潜在的数值不稳定。此防御为必须措施，与 T9-02 测试断言（_masked_loss 返回 0.0）保持一致。

### 4.6 模型输出含极端值的防御

当 logits 极端不平衡（如某个位置 logit 远大于其他位置）时，LogSoftmax 输出可能包含 -inf（某位置 log-probability 被压到极小值）。此时 `nn.KLDivLoss(reduction='none')` 的计算公式 `target * (log(target) - input)` 中，若 `input = -inf` 且 `target > 0`，则 `log(target) - (-inf) = +inf`，乘以 target 后可能产生无限值。

PyTorch 的 LogSoftmax 实现内部会做数值稳定化处理（减最大值），因此在正常训练场景下不会产生 -inf。但若输入 logits 本身包含 NaN 或极端值，仍需防御。约定：**不添加静默 NaN/inf 防御**，而是通过测试 T9-01 验证正常训练场景下 loss 均为有限数。若出现 NaN/inf 则表明数据或模型有严重问题，应由调试手段排查而非在 `_masked_loss` 中静默忽略。

### 4.7 旧权重继续训练的建议

PRD §7.3 明确指出：虽然旧 .pth 权重可直接加载且 predict 结果一致，但损失函数变更导致训练梯度不同，继续在旧权重上训练时训练动态会改变。**建议从 fresh 模型重新训练**，而非在旧权重上继续训练，以确保训练行为与 KL 散度的梯度特性完全匹配。

---

## 5. 改动影响分析

### 5.1 对已有 .pth 权重的影响

网络结构仅变更激活函数类型（Softmax → LogSoftmax），权重参数完全相同。已有 .pth 文件可直接加载，predict 推理结果与修改前完全一致：

```
exp(log_softmax(z)) = softmax(z)
```

### 5.2 对训练动态的影响

KL 散度的梯度特性与 L1 不同：
- L1 对所有位置的梯度为 ±1（均匀）
- KL 散度对高概率位置的梯度更大（集中在峰值区域）

这意味着训练初期 loss 数值范围会变化，**旧 loss 曲线与新 loss 曲线不可直接比较**。

### 5.3 对 sample_inspector 的影响

`sample_inspector.py` 及其入口脚本 `inspect_main.py` 被整体删除。该工具的 `_compute_loss` 功能不再需要，删除后无需适配 KL 散度。
