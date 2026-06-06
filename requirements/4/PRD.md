# PRD: 训练损失函数从 L1 迁移到 KL 散度

## 1. 元信息

| 项目       | 值                                       |
| ---------- | ---------------------------------------- |
| intention_id | 4                                      |
| 目标模块   | `ai/base_trainer.py`、`ai/sample_loader.py`、`ai/lr_range_test.py`、`ai/models/` |
| 文档版本   | v1.0                                     |
| 创建日期   | 2026-06-05                               |

---

## 2. 目标与背景

### 2.1 现状

当前所有模型（FCv1、CNNv1/v2/v3）的训练均使用 L1 Loss（MAE）作为损失函数。训练流水线中的损失计算涉及以下位置：

1. **训练器基类** `BaseTrainer`：`criterion = nn.L1Loss(reduction='none')`，在 `_masked_loss` 中计算 masked L1 loss
2. **LR Range Test** `lr_range_test.py`：独立使用 `nn.L1Loss(reduction='none')`
3. **样本加载器** `sample_loader.py`：提供纯 Python 的 `compute_masked_l1_loss` 函数（供测试使用）

模型输出和 target 均为 Softmax 概率分布（225 维，值域 (0,1)，总和为 1）。

### 2.2 问题

L1 Loss 衡量的是两个向量之间的逐元素绝对误差，对概率分布之间的结构性差异不敏感。当模型输出和 target 都是概率分布时，L1 Loss 无法捕捉分布的整体形态差异（如峰值位置偏差、尾部厚度差异等），导致训练信号不够精确。

### 2.3 目标

将训练损失函数从 L1 Loss 统一迁移到 KL 散度（Kullback-Leibler Divergence），使损失函数与数据语义（概率分布）匹配，提供更精确的训练信号。

---

## 3. 范围与边界

### 3.1 范围内

- 训练器基类 `BaseTrainer` 的损失函数从 L1 改为 KL 散度
- 所有模型的网络输出层从 Softmax 改为 LogSoftmax
- `predict` 方法返回值语义不变（仍返回概率分布）
- LR Range Test 的损失函数同步修改
- `compute_masked_l1_loss` 纯 Python 函数改为 KL 散度实现
- 删除 `sample_inspector.py` 及其入口脚本 `inspect_main.py`、测试文件 `test_fc_v1_inspect.py`（该工具不再需要）
- 相关测试用例同步更新

### 3.2 范围外

- 不修改模型网络结构（层数、通道数、隐藏层维度等）
- 不修改 `Evaluator`、`RewardCalculator`、`probability_utils` 等上游模块
- 不修改 `sample.json` 的数据格式（target 仍为 Softmax 概率分布）
- 不修改优化器配置（AdamW、学习率、权重衰减等）
- 不修改 `game_manager`、`move_sampler`、`engine` 等推理侧模块
- 不修改训练数据路径、训练流程（train_single/train_batch/train_mini_batch/train_auto/train_cycle）

---

## 4. 术语定义

| 术语           | 定义                                                                                   |
| -------------- | -------------------------------------------------------------------------------------- |
| L1 Loss        | 平均绝对误差（MAE），计算两个向量逐元素绝对差的均值                                    |
| KL 散度        | Kullback-Leibler 散度，衡量概率分布 P 相对于参考分布 Q 的信息损失                      |
| LogSoftmax     | Softmax 的对数形式，数值更稳定，是 PyTorch KLDivLoss 要求的输入格式                    |
| Softmax 概率分布 | 225 维向量，值域 (0,1)，总和为 1，表示 225 个落子位置的概率                            |
| Mask           | 225 维向量，1 表示空位（参与损失计算），0 表示已占位（不参与损失计算）                 |

---

## 5. 功能说明

### 5.1 损失函数替换

将训练过程中使用的损失函数从 L1 Loss 替换为 KL 散度，涵盖以下位置：

#### 5.1.1 训练器基类（BaseTrainer）

- 损失函数从 `nn.L1Loss` 改为 `nn.KLDivLoss`
- `_masked_loss` 方法保持 mask 机制不变（仅对 mask=1 的位置计算损失）
- mask 语义不变：mask=1 的空位参与损失计算，mask=0 的已占位不参与

#### 5.1.2 LR Range Test（lr_range_test.py）

- 损失函数从 `nn.L1Loss` 改为 `nn.KLDivLoss`
- 与 BaseTrainer 的损失计算保持一致

#### 5.1.3 纯 Python 辅助函数（sample_loader.py）

- `compute_masked_l1_loss` 函数重命名为 `compute_masked_kl_loss`
- 计算逻辑从 L1 改为 KL 散度
- 函数签名和返回值语义相应更新

#### 5.1.4 样本检视器删除（sample_inspector.py）

- 删除 `ai/models/fc_v1/sample_inspector.py`（该工具不再需要）
- 删除入口脚本 `ai/models/fc_v1/inspect_main.py`（依赖 SampleInspector，删除后无意义）
- 删除测试文件 `tests/ai/test_fc_v1_inspect.py`

### 5.2 网络输出层修改

所有模型的 `forward` 方法输出层从 `nn.Softmax(dim=1)` 改为 `nn.LogSoftmax(dim=1)`，确保与 `nn.KLDivLoss` 的输入要求匹配。

涉及模型：

| 模型    | 网络类          | 当前输出层           | 修改后输出层             |
| ------- | --------------- | -------------------- | ------------------------ |
| FCv1    | FCv1Network     | nn.Softmax(dim=1)    | nn.LogSoftmax(dim=1)    |
| CNNv1   | CNNV1Network    | nn.Softmax(dim=1)    | nn.LogSoftmax(dim=1)    |
| CNNv2   | CNNV2Network    | nn.Softmax(dim=1)    | nn.LogSoftmax(dim=1)    |
| CNNv3   | CNNV3Network    | nn.Softmax(dim=1)    | nn.LogSoftmax(dim=1)    |

### 5.3 predict 方法兼容性

模型 `predict` 方法的返回值语义不变：仍返回 225 维概率分布（值域 (0,1)，总和为 1）。由于网络输出变为 LogSoftmax，predict 方法需将 log-probability 还原为 probability（取 exp）。

predict 方法的调用方（engine、game_manager、move_sampler 等）无需任何修改。

### 5.4 KLDivLoss 输入格式要求

PyTorch `nn.KLDivLoss` 要求：
- **输入（prediction）**：log-probabilities（即网络直接输出的 LogSoftmax 结果）
- **目标（target）**：普通概率值（即 sample.json 中的 Softmax 概率分布，无需取 log）

target 数据（sample.json 中的 output 字段）格式不变，仍为 Softmax 概率分布。

### 5.5 数值稳定性

- 使用 `nn.LogSoftmax` 而非先 Softmax 再取 log，避免数值下溢
- `nn.KLDivLoss` 的 `reduction='none'` 保持与当前 L1Loss 一致的使用模式，由 `_masked_loss` 统一做 mask 和归约

---

## 6. 验收步骤

### 6.1 损失函数替换验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| A-01 | 确认 `BaseTrainer.__init__` 中 criterion 为 `nn.KLDivLoss(reduction='none')`                       | criterion 类型为 KLDivLoss                                     |
| A-02 | 构造已知概率分布 P 和 Q，调用 `_masked_loss(P, Q, mask)`                                           | 返回值与手动计算的 KL 散度一致（允许浮点误差 1e-5）           |
| A-03 | mask 全为 0 时调用 `_masked_loss`                                                                   | 返回 0（与当前 L1 行为一致）                                   |
| A-04 | mask 部分为 0 时，验证仅 mask=1 位置参与损失计算                                                    | mask=0 位置对损失无贡献                                        |

### 6.2 网络输出层验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| B-01 | 确认所有模型的网络类使用 `nn.LogSoftmax(dim=1)`                                                    | 无任何网络类使用 `nn.Softmax`                                  |
| B-02 | 调用任意模型的 `predict` 方法                                                                       | 返回 225 维列表，值域 (0,1)，总和为 1                          |
| B-03 | 对同一输入，对比修改前后的 `predict` 输出                                                           | 概率分布一致（浮点误差 1e-5 内，仅输出层表示形式不同）        |
| B-04 | 直接调用网络 forward，验证输出为 log-probability                                                    | 输出值 <= 0，exp(output) 的总和为 1                            |

### 6.3 纯 Python 辅助函数验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| C-01 | 确认 `compute_masked_l1_loss` 已重命名/替换为 KL 散度版本                                          | 旧函数名不存在，新函数存在且计算 KL 散度                      |
| C-02 | 构造已知概率分布，调用纯 Python KL 散度函数                                                         | 返回值与 `_masked_loss` 结果一致（允许浮点误差 1e-5）         |
| C-03 | 旧函数名 `compute_masked_l1_loss` 的所有引用已同步更新                                             | 全局搜索 `compute_masked_l1_loss` 无结果                       |

### 6.4 样本检视器删除验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| D-01 | 确认 `ai/models/fc_v1/sample_inspector.py` 已删除                                                 | 文件不存在                                                     |
| D-02 | 确认 `ai/models/fc_v1/inspect_main.py` 已删除                                                     | 文件不存在                                                     |
| D-03 | 确认 `tests/ai/test_fc_v1_inspect.py` 已删除                                                      | 文件不存在                                                     |
| D-04 | 全局搜索 `SampleInspector` 无结果                                                                  | 无残留引用                                                     |

### 6.5 LR Range Test 验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| E-01 | 确认 `lr_range_test.py` 使用 `nn.KLDivLoss(reduction='none')`                                      | criterion 类型为 KLDivLoss                                     |
| E-02 | 执行 LR Range Test（小规模，如 `--rounds 3 --epochs 10`）                                          | 正常完成，无报错，loss 曲线图正常生成                          |

### 6.6 端到端训练验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| F-01 | 使用 `train_auto` 训练任意模型（小规模：10 epochs, 1 sample dir）                                  | 训练正常完成，loss 值为非负数                                  |
| F-02 | 训练前后调用 `predict`，验证输出仍为合法概率分布                                                    | 值域 (0,1)，总和为 1                                           |
| F-03 | 使用 val_ratio > 0 训练，验证 val_loss 正常计算                                                    | val_loss 为非负数，曲线图包含 train/val 两条线                |

### 6.7 向后兼容验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| G-01 | 加载已有 `.pth` 权重文件后调用 `predict`                                                           | 成功加载，输出为合法概率分布                                   |
| G-02 | `predict` 返回值格式与修改前一致                                                                    | 225 维列表，float 类型，值域 (0,1)，总和为 1                  |

### 6.8 全量测试验证

| 编号 | 操作                                                                                               | 期望结果                                                       |
| ---- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| H-01 | 执行 `python -m pytest tests/ -v`                                                                   | 所有测试通过，无失败                                           |

---

## 7. 约束与注意事项

1. **概率分布语义**：KL 散度要求 target 为合法概率分布（非负、总和为 1）。sample.json 中的 output 由 `masked_softmax` 生成，保证满足此条件。但需防御极端情况（如全零 target），此时 KL 散度无定义，应有合理的回退行为
2. **损失值不可直接比较**：KL 散度的数值范围与 L1 Loss 不同，修改后训练日志中的 loss 值与修改前不可直接比较。loss 曲线图的 Y 轴含义变更
3. **旧权重兼容**：网络结构不变（仅激活函数从 Softmax 改为 LogSoftmax），已有 `.pth` 权重文件可直接加载，且 predict 推理结果与修改前完全一致（exp(log_softmax) = softmax）。但损失函数变更导致训练梯度不同，继续在旧权重上训练时训练动态会改变，建议从 fresh 模型重新训练
4. **mask 机制保留**：KL 散度同样需要 mask 机制来排除已占位位置，mask 语义和用法不变
