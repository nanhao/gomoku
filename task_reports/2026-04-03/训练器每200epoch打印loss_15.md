# 训练器每 200 epoch 打印一次 loss

## 1. 需求内容

`ai/fc_v1/train.py`（通过 `trainer.py`）改为每 200 个 epoch 才打印一次 loss，而不是每个 epoch 都打印。

## 2. 解决思路

`trainer.py` 中的 `train_single` 和 `train_batch` 方法各有一处 `verbose` 打印逻辑，将原来的无条件打印改为只在 `(epoch + 1) % 200 == 0` 时才打印。同步更新测试用例，使其验证新的打印频率。

## 3. 执行过程

### 步骤一：修改 `trainer.py`
- 将 `train_single` 和 `train_batch` 中的打印条件从 `if verbose:` 改为 `if verbose and (epoch + 1) % 200 == 0:`
- 注释也同步更新为"每 200 个 epoch 输出一次损失"

### 步骤二：更新测试用例
受影响的 4 个测试用例：
- `test_verbose_true_outputs_epoch_loss`：将 epochs=3 改为 epochs=400，验证第 200、400 个 epoch 有输出且只有 2 行
- `test_verbose_output_format`：将 epochs=2 改为 epochs=200，验证第 200 个 epoch 有输出
- `test_train_batch_multiple_sample_dirs_merged`：epochs=2 均小于 200，预期行数改为 0
- `test_train_batch_verbose_outputs_epoch_loss`：将 epochs=3 改为 epochs=400，验证第 200、400 个 epoch 有输出

### 步骤三：运行测试
- 全量运行 `tests/ai/test_fc_v1_trainer.py`
- 结果：43 passed，0 failed

## 4. 执行结果

功能按需求实现，所有测试通过。训练时每 200 个 epoch 打印一次 loss，减少了日志噪音。

## 5. 涉及文件

| 文件 | 操作 |
|------|------|
| `ai/fc_v1/trainer.py` | 修改 `train_single` 和 `train_batch` 中的打印频率逻辑 |
| `tests/ai/test_fc_v1_trainer.py` | 更新 4 个 verbose 相关测试用例 |
| `task_reports/2026-04-03/训练器每200epoch打印loss_15.md` | 新建（本报告） |
