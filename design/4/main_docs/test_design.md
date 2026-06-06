# 测试用例设计

## 1. compute_masked_kl_loss 测试（`tests/ai/test_sample_loader.py`）（对应 PRD §6.3）

原 `TestComputeMaskedL1Loss` 类重命名为 `TestComputeMaskedKLLoss`，测试逻辑重写。

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T1-01 | 完全相同分布 loss 为 0 | log_estimated = log([0.5, 0.5, 0, ..., 0]), target = [0.5, 0.5, 0, ..., 0], mask = [1,1,0,...,0] | 0.0 |
| T1-02 | 已知分布 KL 散度 | log_estimated = log([0.7, 0.3]), target = [0.5, 0.5], mask 前两位=1 | \u2248 0.0872（手动计算验证） |
| T1-03 | mask 全为 0 返回 0 | 任意值, mask 全 0 | 0.0 |
| T1-04 | target 含零项 | log_estimated = log([0.8, 0.2]), target = [0.0, 1.0], mask = [1,1] | target[0]=0 位置贡献为 0（跳过 log(0) 计算） + target[1]=1 位置贡献为 log(1)-log(0.2) ≈ 1.6094 |
| T1-05 | KL 散度非对称性 | 交换 T1-02 的 log_estimated 和 target | 结果与 T1-02 不同 |
| T1-06 | 与 _masked_loss 一致性 | 同一组数据，分别调用 compute_masked_kl_loss 和 _masked_loss | 两者结果一致（1e-5 内） |

> 修正说明：T1-02 期望值原为 0.0823，经复核 KL([0.5,0.5] || [0.7,0.3]) = 0.5*(ln(0.5)-ln(0.7)) + 0.5*(ln(0.5)-ln(0.3)) \u2248 0.0872，原值为计算笔误，已修正。

## 2. BaseTrainer._masked_loss 测试（`tests/ai/test_fc_v1_trainer.py`）（对应 PRD §6.1）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T2-01 | criterion 类型验证 | 检查 trainer.criterion 类型 | isinstance(trainer.criterion, nn.KLDivLoss) |
| T2-02 | 已知分布 KL 散度 | log_softmax 输出 + Softmax target + mask | 与手动计算一致 |
| T2-03 | mask 全为 0 | 任意输出, mask 全 0 | 0.0 |
| T2-04 | mask 部分为 0 | 输出含 mask=0 和 mask=1 位置 | 仅 mask=1 位置贡献损失 |
| T2-05 | 与 compute_masked_kl_loss 一致 | 同一组数据，两种方式计算 | 1e-5 内一致 |
| T2-06 | KL 散度非负 | 任意合法概率分布 | loss >= 0 |

## 3. 样本检视器删除验证（对应 PRD §6.4）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T3-01 | sample_inspector.py 已删除 | 检查文件是否存在 | 文件不存在 |
| T3-02 | inspect_main.py 已删除 | 检查文件是否存在 | 文件不存在 |
| T3-03 | test_fc_v1_inspect.py 已删除 | 检查文件是否存在 | 文件不存在 |
| T3-04 | SampleInspector 无残留引用 | 全局搜索 `SampleInspector` | 0 个匹配（含代码和注释） |

## 4. _masked_loss 在 Softmax 输入下的一致性测试（`tests/ai/test_fc_v1_softmax_input.py`）（对应 PRD §6.1）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T4-01 | 已知分布 KL 散度值正确性 | 构造已知概率分布 P 和 Q，分别用 _masked_loss 和手动公式计算 | _masked_loss 结果与手动计算的 KL(P||Q) 一致（1e-5 内） |
| T4-02 | mask 机制保留 | 与 L1 版本相同 mask | mask 行为不变（仅计算公式变） |

## 5. predict 输出验证（`tests/ai/test_neural_network_model.py`）（对应 PRD §6.2、§6.7）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T5-01 | predict 返回概率分布 | 任意 board_vector | 225 维，所有值 ∈ (0,1)，总和 ≈ 1 |
| T5-02 | exp(log_softmax) = softmax | 同一输入，对比网络 LogSoftmax 输出取 exp 与直接 Softmax | 1e-5 内一致 |
| T5-03 | 旧权重加载后 predict 一致 | 加载旧 .pth 权重，调用 predict | 结果与旧版本一致 |

## 6. 网络输出层验证（对应 PRD §6.2）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T6-01 | FCv1Network forward 输出为 log-probability | 任意输入 | 输出值 ≤ 0，exp 后总和 ≈ 1 |
| T6-02 | CNNV3Network forward 输出为 log-probability | 任意输入 | 输出值 ≤ 0，exp 后总和 ≈ 1 |
| T6-03 | 无 Softmax 残留 | 全局搜索 `nn.Softmax` | 0 个匹配 |
| T6-04 | 旧函数名清理 | 全局搜索 `compute_masked_l1_loss` | 0 个匹配（含代码和注释） |

## 7. LR Range Test 验证（对应 PRD §6.5）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T7-01 | 所有 loss 为有限数 | 小规模 range test | torch.isfinite(losses) 全为 True |
| T7-02 | criterion 类型 | 检查代码 | 为 KLDivLoss |

## 8. 端到端训练验证（对应 PRD \u00a76.6）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T8-01 | train_auto 正常完成 | 小规模训练（10 epochs, 1 sample dir） | 训练正常完成，loss 为非负数 |
| T8-02 | 训练后 predict 输出合法概率 | train_auto 后调用 predict | 225 维，所有值 ∈ (0,1)，总和 ≈ 1 |
| T8-03 | val_ratio > 0 时 val_loss 正常 | val_ratio=0.2, 小规模训练 | val_loss 为非负数，loss 曲线包含 train/val 两条线 |

## 9. 数值稳定性边界验证（对应 PRD §7.1）

| 编号 | 用例名 | 输入 | 期望输出 |
| ---- | ------ | ---- | -------- |
| T9-01 | 模型输出含极端值时 loss 仍有限 | 构造极端 logits（如同一位置 logit 远大于其他位置） | torch.isfinite(loss) 为 True |
| T9-02 | mask=1 位置 target 全为 0 | 全零 target, mask=1 | _masked_loss 返回 0.0（防御生效） |
| T9-03 | compute_masked_kl_loss 中 target[i]=0 | log_estimated 包含正常值, target 含 0 | target[i]=0 位置贡献为 0，不抛异常 |
