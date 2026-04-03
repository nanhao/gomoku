# 训练周期 epoch loss 输出功能实现

## 1. 需求内容

用户需求（来自 `requirements/key_points.md` 第 383 行）：
> 开启训练周期时，要在每个 epoc 结束时输出 loss 的平均值

## 2. 解决思路

采用 TDD 开发流程：
1. 先更新设计文档，明确 `verbose` 参数和输出行为
2. 编写测试用例覆盖 verbose 功能
3. 修改代码实现
4. 运行测试确保全部通过

## 3. 执行过程

### 3.1 更新设计文档

修改 `design/fc_v1_design.md`：
- 更新 `train_single` 方法签名，添加 `verbose: bool = False` 参数
- 更新单轮次训练流程，添加 epoch loss 输出步骤
- 更新训练周期流程，强调使用 `verbose=True` 调用

### 3.2 编写测试用例

在 `tests/ai/test_fc_v1_trainer.py` 添加 `TestFCv1TrainerVerboseOutput` 测试类：
- `test_verbose_false_no_output`：测试默认不输出 epoch loss
- `test_verbose_true_outputs_epoch_loss`：测试 verbose=True 输出每个 epoch loss
- `test_verbose_output_format`：测试输出格式正确（包含 "Epoch X/Y, loss: Z.ZZZZ"）

### 3.3 修改代码实现

**trainer.py**：

1. `train_single` 方法：
   - 添加 `verbose: bool = False` 参数
   - 在每个 epoch 结束后，若 `verbose=True` 则输出 loss

2. `train_cycle` 方法：
   - 调用 `train_single` 时传入 `verbose=True`

### 3.4 运行测试

执行命令：
```bash
python -m unittest tests.ai.test_fc_v1_trainer -v
```

结果：**24 个测试全部通过**

实际输出示例：
```
训练模型 (1 epochs)...
Epoch 1/1, loss: 13997.0498
Epoch 1/1, loss: 26125.7500
本轮平均损失: 20061.3999
```

## 4. 执行结果

- 设计文档已更新，明确了 verbose 参数和输出行为
- 代码实现符合需求：训练周期时每个 epoch 结束输出平均 loss
- 所有测试通过，功能验证完成

## 5. 涉及文件

| 文件路径 | 操作类型 |
|----------|----------|
| `design/fc_v1_design.md` | 修改 |
| `ai/fc_v1/trainer.py` | 修改 |
| `tests/ai/test_fc_v1_trainer.py` | 修改 |
