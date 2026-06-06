# Bug 报告：CNN v3 训练 grad_norm 始终为零（ReLU 导致梯度消失）

## 发现时间

2026-06-06

## 触发命令

```powershell
python ai/train_main.py --model cnn_v3.5 --fresh --train-from-dir ai/merge_samples/v4 --epochs 300 --batch-size 4096 --shuffle --val-ratio 0.2
```

## 现象

训练 300 个 epoch（4500 个 step），loss 曲线完全水平：

| 指标 | 观察 | 预期 |
|------|------|------|
| **grad_norm** | 全部 4500 步均为 `0.0` | 应随训练逐渐减小但 >0 |
| **val_loss** | 300 次评估全部为 `0.01665099523961544`（精确到 17 位小数不变） | 应随训练逐渐下降 |
| **train loss** | 在 `0.0163~0.0169` 间随机波动，无下降趋势 | 应持续下降 |
| **模型输出** | 全部 225 个位置输出相同值（均匀分布） | 应有区分度 |

日志文件：`ai/ai_data/loss_logs/cnn_v3.5_20260606_233756_a2de9580.jsonl`

## 根因分析

### 直接原因

`CNNV3Network` 在最后的 1×1 卷积 (`conv_out`) 与 `LogSoftmax` 之间多插了一个 ReLU (`relu_out`)：

```python
# ai/models/cnn_v3/model.py (修复前)
self.conv_out = nn.Conv2d(32, 1, kernel_size=1)
self.relu_out = nn.ReLU()          # ← 多余的 ReLU
self.log_softmax = nn.LogSoftmax(dim=1)

def forward(self, x):
    ...
    x = self.relu_out(self.conv_out(x))  # ← ReLU 在 LogSoftmax 之前
    x = x.view(x.size(0), -1)
    x = self.log_softmax(x)
    return x
```

### 梯度消失机制

1. `conv_out` 是 1×1 卷积，将 32 通道压缩为单通道。随机初始化时，bias 和 weight 可能导致**全部 225 个位置**的输出为负值
2. ReLU 将负值全部置零 → LogSoftmax 收到 225 个全零向量
3. LogSoftmax(全零) 输出完全均匀分布 `log(1/225) ≈ -5.416`
4. KL 散度对均匀分布的梯度虽然非零，但 ReLU 的局部梯度为 0（因为输入 ≤ 0）
5. **整个网络的梯度链被 ReLU 切断，`loss.backward()` 后端所有参数 `p.grad` 为全零张量**

### 为什么有时能训练、有时不能

是否触发取决于权重随机初始化的运气：
- bias 初始化为正 → 部分位置 >0 → 有梯度 → 正常训练
- bias 初始化为负 → 全部位置 ≤0 → 梯度消失 → 死网络

用户的训练恰好命中了后者。

### 与设计文档的矛盾

[设计文档](file://d:/projects/gomoku/design/cnn_v3_design.md#L8) 明确描述为：

> Conv → ReLU → 1×1 Conv → LogSoftmax

即三层 Conv+ReLU 之后，1×1 Conv 直接接 LogSoftmax，**不应有额外的 ReLU**。

## 修复方案

删除 `conv_out` 之后的 `relu_out`，1×1 卷积直接输出到 LogSoftmax：

```python
# ai/models/cnn_v3/model.py (修复后)
self.conv_out = nn.Conv2d(32, 1, kernel_size=1)
# relu_out 已删除
self.log_softmax = nn.LogSoftmax(dim=1)

def forward(self, x):
    ...
    x = self.conv_out(x)   # 直接输出，不加 ReLU
    x = x.view(x.size(0), -1)
    x = self.log_softmax(x)
    return x
```

理由：LogSoftmax 可以处理任意实数值，ReLU 在它之前只会消灭信息，没有任何积极作用。

## 修复影响

- 网络参数量减少 0 个（ReLU 无参数），结构变化不改变参数 shape
- **不兼容旧权重**：旧 `.pth` 文件的 `state_dict` 不含 `relu_out`，加载正常；但训练出的旧模型因结构不同，重新训练时需用 `--fresh`
- 设计文档无需修改（设计描述本身正确，是实现的 bug）

## 验证方法

1. 用修复后代码创建新模型，加载 `ai/merge_samples/v4` 数据
2. 执行前向+反向传播
3. 检查 `p.grad.norm()` 是否非零

修复前（batch_size=4096）：
```
outputs min/max: -5.416101 / -5.416101   ← 全部相同，均匀分布
grad_norm total: 0.0000000000             ← 梯度为零
```

修复后（batch_size=4096）：
```
outputs min/max: -5.444489 / -5.385241   ← 有区分度
grad_norm total: 0.0009518983             ← 梯度正常
参数最大变化: ~0.001                      ← 正常学习
```

## 修复文件

- `ai/models/cnn_v3/model.py`：删除 `self.relu_out`，`forward()` 中 `conv_out` 直接接入 `view → LogSoftmax`

## 测试覆盖

- `tests/ai/test_cnn_models.py`：72/72 全部通过（含 CNN v3 全部测试），无需新增用例——现有用例已覆盖前向传播输出范围、LogSoftmax 行和、batch forward 等，修复后均正常通过
