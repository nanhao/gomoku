# 训练 reward 归一化实现

**日期**: 2026-04-03  
**序号**: 11

---

## 1. 需求内容

在 FC v1 训练器内部，对 target reward 做归一化处理后再计算 loss：
- 归一化范围：`[-1000000, 1000000] -> [-1, 1]`
- 归一化仅在 trainer 内部（`_prepare_tensors`）生效
- `sample.json` 中的原始值不变
- 模型 `predict` 输出不受影响

需求来源：`requirements/key_points.md` 第七次追加

---

## 2. 解决思路

在 `FCv1Trainer._prepare_tensors()` 中，将 `targets_tensor` 除以 `REWARD_NORMALIZE_FACTOR = 1_000_000.0`。

这样做的原因：
- 网络初始输出约为 ±几十，target 为 ±1,000,000，量级差 4-5 个数量级
- L1 loss 梯度为 ±1，Adam lr=0.001，收敛到 target 需要约 10 亿步
- 归一化后 target 变为 ±1，量级匹配，大幅加快收敛速度
- 修改范围极小，对外接口（sample.json、predict）完全不变

---

## 3. 执行过程

### 步骤1：更新设计文档
- 新增第 6.2 节「Target Reward 归一化」，说明目的、公式、适用范围
- 更新第 6.1 节损失公式，加入 `normalize()` 函数
- 更新第 7.2 节训练数据流，加入归一化步骤
- 版本升级至 v1.3

### 步骤2：先写测试（TDD）
在 `tests/ai/test_fc_v1_trainer.py` 新增 `TestRewardNormalization` 类，包含：
- **primary**：`test_targets_are_normalized_in_prepare_tensors`：±1,000,000 应归一化为 ±1
- **primary**：`test_zero_reward_stays_zero_after_normalization`：0 归一化后仍为 0
- **primary**：`test_loss_scale_is_reasonable_after_normalization`：归一化后 loss < 100
- **primary**：`test_original_sample_data_unchanged`：不修改原始 sample_data
- **edge**：`test_partial_reward_normalization`：中间值 ±500,000 归一化为 ±0.5
- **edge**：`test_loss_decreases_with_normalization`：50 epoch 平均 loss < 1 epoch loss

### 步骤3：实现归一化逻辑
在 `ai/fc_v1/trainer.py` 中：
1. 新增类常量 `REWARD_NORMALIZE_FACTOR = 1_000_000.0`
2. 在 `_prepare_tensors` 末尾加一行：`targets_tensor = targets_tensor / self.REWARD_NORMALIZE_FACTOR`

### 步骤4：修复一个受影响的原有测试
`test_masked_loss_consistent_with_compute_loss` 原本验证 `_masked_loss` 与 `_compute_loss` 结果相等（均为 50.0）。归一化后两者语义不再相同（一个用原始值，一个用归一化值），更新该测试以反映新语义（归一化后 `_masked_loss` 应约为 100.0）。

### 步骤5：全量测试
运行 `tests.ai.test_fc_v1_trainer`，43 个测试全部通过。

---

## 4. 执行结果

- 新增 6 个归一化测试，全部通过
- 修复 1 个受影响的原有测试
- 全量 43 个测试通过，无回归
- 训练时 loss 量级从 ~258,231 降至 ~0.258，收敛速度大幅提升

---

## 5. 涉及文件

| 文件 | 操作 |
|------|------|
| `design/fc_v1_design.md` | 更新：新增 6.2 归一化章节，更新 6.1/7.2，版本升至 v1.3 |
| `ai/fc_v1/trainer.py` | 修改：新增 `REWARD_NORMALIZE_FACTOR` 常量，`_prepare_tensors` 中归一化 targets |
| `tests/ai/test_fc_v1_trainer.py` | 修改：新增 `TestRewardNormalization` 类（6个用例），更新 `test_masked_loss_consistent_with_compute_loss` |
