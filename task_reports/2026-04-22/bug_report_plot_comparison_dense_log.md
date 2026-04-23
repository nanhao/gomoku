# Bug 报告：plot_comparison 不支持密集日志格式

## 发现时间

2026-04-22

## Bug 描述

在使用 mini-batch 训练模式（`--batch-size`）时，系统启用密集日志（dense log），记录每个 batch 的 `(step, loss, grad_norm)` 数据。训练结束后，`SelfPlayCycle` 调用 `LossLogger.plot_comparison()` 生成各轮次 loss 的合并对比图，但绘图失败并抛出 RuntimeWarning：

```
RuntimeWarning: matplotlib 合并绘图失败，跳过对比图生成。
```

## 根因分析

`LossLogger.plot_comparison()` 方法在读取 `round_N_epoch_level.jsonl` 时，假设所有日志条目都包含 `epoch` 字段：

```python
epochs = [entry["epoch"] for entry in entries]
```

但密集日志的数据格式使用 `step` 字段而非 `epoch`：

```json
{"step": 1234, "loss": 0.0015, "grad_norm": 0.012}
```

当 `plot_comparison` 尝试读取 `entry["epoch"]` 时，触发 `KeyError`，被外层 `except Exception` 捕获后返回 None，导致合并对比图生成失败。

## 影响范围

- **仅影响 mini-batch 训练模式**（启用 `--batch-size` 时）
- 全量更新模式（train_batch）使用稀疏日志（含 `epoch` 字段），不受影响
- 单轮次训练（rounds=1）时对比图本身意义不大，但多轮次时无法查看各轮 loss 对比

## 修复方案

在 `plot_comparison` 中检测数据格式：

1. 如果第一条记录包含 `step` 字段，使用 `step` 作为 x 轴数据，x 轴标签设为 "Step"
2. 否则回退到 `epoch` 字段，x 轴标签设为 "Epoch"

```python
if entries and "step" in entries[0]:
    xs = [entry["step"] for entry in entries]
    xlabel = "Step"
else:
    xs = [entry["epoch"] for entry in entries]
```

## 测试覆盖

新增测试用例 `test_plot_comparison_with_dense_log_step_format`：

- 构造包含 `step`/`loss`/`grad_norm` 的密集日志文件
- 调用 `plot_comparison`
- 断言返回路径非 None 且图片文件存在

## 修复文件

- `ai/loss_logger.py`：`plot_comparison()` 方法
- `tests/ai/test_loss_logger.py`：新增测试用例

## 验证结果

全量测试 500/500 通过（212 基础 + 288 AI）。
