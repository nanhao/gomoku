# 训练周期 epoch 参数更新语义修复

## 1. 需求内容

用户反馈：
> 这里发现了一个新问题：不仅是打印的问题，真正的问题是应该做完一个 epoch 才更新一次参数。train_single 是做完一个 sample 文件就更新一次参数。你不应该在 train_cycle 里面去调用 train_single。

需求来源：`requirements/key_points.md` 第383行：
> 开启训练周期时，要在每个 epoch 结束时输出 loss 的平均值

**真正的问题**：`train_cycle` 中对每个 sample 文件都调用一次 `train_single`，意味着一个 epoch 里参数被更新了 N 次（N = sample 文件数）。正确的语义是将所有 sample 数据合并，每个 epoch 只做一次前向传播 + 一次参数更新。

## 2. 解决思路

新增 `train_batch` 方法：
- 接收 `sample_dirs: list`（多个 sample 目录）
- 将所有 sample.json 数据合并为一个大数据集
- 在合并数据集上训练 `epochs` 次，**每次只做一次参数更新**
- 每个 epoch 结束时输出 loss（verbose=True 时）

`train_cycle` 内部改为调用 `train_batch`，而不是循环调用 `train_single`。

同时将原 `train_single` 中重复的张量准备逻辑提取为私有方法 `_prepare_tensors` 和 `_masked_loss`，供两个训练方法共享。

## 3. 执行过程

### 步骤 1：更新设计文档

文件：`design/fc_v1_design.md`

- 在公共方法表中添加 `train_batch` 方法
- 更新代码框架，添加 `train_batch` 框架
- 更新 5.2 训练周期流程说明：明确"合并数据集，每 epoch 只做一次参数更新"

### 步骤 2：编写测试用例（TDD）

文件：`tests/ai/test_fc_v1_trainer.py`

新增 `TestFCv1TrainerBatch` 测试类，包含：

**主流程测试（5个）**：
- `test_train_batch_returns_float`：验证返回值是 float 且 >= 0
- `test_train_batch_single_sample_dir`：单个 sample 目录能正常训练
- `test_train_batch_multiple_sample_dirs_merged`：3个 sample 目录 × 2 epochs 只输出 2 行（验证合并语义）
- `test_train_batch_verbose_outputs_epoch_loss`：verbose=True 时输出每个 epoch 的 loss
- `test_train_batch_verbose_false_no_epoch_output`：verbose=False 时无 Epoch 输出

**边缘测试（3个）**：
- `test_train_batch_empty_list_raises`：空列表抛出 ValueError
- `test_train_batch_missing_sample_file_raises`：不存在目录抛出 FileNotFoundError
- `test_train_batch_updates_model_params`：训练后模型参数确实改变

### 步骤 3：实现代码

文件：`ai/fc_v1/trainer.py`

重构内容：
1. **新增 `train_batch` 方法**：合并所有 sample 数据，每 epoch 只做一次参数更新
2. **提取 `_prepare_tensors` 私有方法**：将输入/目标/mask 转换为 PyTorch 张量（供 train_single 和 train_batch 共享）
3. **提取 `_masked_loss` 私有方法**：计算 masked L1 loss（供两个训练方法共享）
4. **修改 `train_single`**：改用 `_prepare_tensors` + `_masked_loss`
5. **修改 `train_cycle`**：改为调用 `train_batch(sample_dirs, epochs, verbose=True)`，收集本轮所有 sample 目录，合并训练
6. **保留 `_compute_loss`**：纯 Python 实现，供旧单元测试使用

### 步骤 4：修复测试问题

- **ERROR（3个）**：旧测试调用 `_compute_loss`，该方法在重构中被删除 → 恢复该方法
- **FAIL（1个）**：`test_train_batch_updates_model_params` 使用全零输入导致梯度为零，参数无法更新 → 改为使用非零输入（`[float(i+1)] * 226`）和非零目标（`output[2] = 1000.0`）

## 4. 执行结果

```
Ran 32 tests in 7.652s
OK
```

全部 32 个测试通过，包括 8 个新增的 `TestFCv1TrainerBatch` 测试用例。

`train_cycle` 训练输出现在正确：
```
训练模型 (1 epochs，共 2 个样本合并训练)...
Epoch 1/1, loss: 14922.4395
本轮平均损失: 14922.4395
```
100 个 sample 文件 × 1 epoch = 只输出 **1** 行 Epoch 信息（而不是之前的 100 行）。

## 5. 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `design/fc_v1_design.md` | 修改 | 添加 `train_batch` 方法设计，更新训练周期流程 |
| `ai/fc_v1/trainer.py` | 重写 | 新增 `train_batch`、`_prepare_tensors`、`_masked_loss` 方法；重构 `train_single`；修改 `train_cycle` 调用 `train_batch` |
| `tests/ai/test_fc_v1_trainer.py` | 修改 | 新增 `TestFCv1TrainerBatch` 测试类（8个测试用例）；修复 `_create_sample_dir` 使用非零数据 |
