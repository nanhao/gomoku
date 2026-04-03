# 修复 trainer.py 中 mask 扩展逻辑错误

## 1. 需求内容

用户发现 `ai/fc_v1/trainer.py` 中 `_prepare_tensors` 方法的 mask 扩展逻辑与 `ai/evaluator.py` 中 output 的实际排列格式不一致，要求修复该 bug。

## 2. 解决思路

通过阅读 `evaluator.py` 确认 output 的实际排列为**交错格式**：
```
[x0, y0, r0,  x1, y1, r1,  ...,  x224, y224, r224]
```
第 i 个位置对应索引：`i*3`（x）、`i*3+1`（y）、`i*3+2`（reward）

而 `_prepare_tensors` 中将 mask 按**分段格式**扩展（`i`、`i+225`、`i+450`），与实际排列不符，导致训练时 mask 作用在错误的位置上，损失计算完全失效（loss 恒为 0）。

修复方案：将 mask 扩展改为交错格式（`i*3`、`i*3+1`、`i*3+2`）。

## 3. 执行过程

### 步骤 1：确认 bug

- 阅读 `evaluator.py` 的 `_initialize_sample` 和 `_find_reward_index` 方法，确认 output 使用交错排列
- 对比 `trainer.py` 的 `_prepare_tensors`（分段扩展）与 `_compute_loss`（交错索引）的矛盾

### 步骤 2：编写测试用例（TDD）

在 `tests/ai/test_fc_v1_trainer.py` 中新增 `TestFCv1TrainerPrepareTensors` 测试类，包含 3 个测试：
- `test_mask_expanded_uses_interleaved_layout`：验证第 i 个位置 mask=1 时，扩展后 `i*3`、`i*3+1`、`i*3+2` 为 1，分段位置为 0
- `test_mask_expanded_only_active_positions_nonzero`：验证 mask=0 的位置扩展后全为 0
- `test_masked_loss_consistent_with_compute_loss`：验证 tensor 损失 > 0（修复前为 0）

运行测试，确认 6 个断言失败（bug 存在）。

### 步骤 3：修复代码

将 `_prepare_tensors` 中的 mask 扩展逻辑从分段格式改为交错格式：

```python
# 修复前（错误）
masks_expanded[:, i] = masks_tensor[:, i]           # x坐标
masks_expanded[:, i + 225] = masks_tensor[:, i]     # y坐标
masks_expanded[:, i + 450] = masks_tensor[:, i]     # reward

# 修复后（正确）
masks_expanded[:, i * 3] = masks_tensor[:, i]           # x坐标
masks_expanded[:, i * 3 + 1] = masks_tensor[:, i]       # y坐标
masks_expanded[:, i * 3 + 2] = masks_tensor[:, i]       # reward
```

### 步骤 4：修正测试边界问题

发现 `mask_idx=224` 时，交错格式的 reward 索引（`224*3+2=674`）与分段格式（`224+450=674`）重合，无法区分，将测试参数改为 `223`。

### 步骤 5：验证

运行全部 35 个测试，全部通过。

## 4. 执行结果

- Bug 已修复：`_prepare_tensors` 的 mask 扩展现在正确使用交错格式
- 新增 3 个测试用例，覆盖 mask 扩展逻辑
- 全部 35 个测试通过（原有 32 个 + 新增 3 个）

## 5. 涉及文件

| 文件 | 操作 |
|------|------|
| `ai/fc_v1/trainer.py` | 修复 `_prepare_tensors` 的 mask 扩展逻辑 |
| `tests/ai/test_fc_v1_trainer.py` | 新增 `TestFCv1TrainerPrepareTensors` 测试类（3个测试） |
