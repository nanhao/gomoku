# LossLogger 改进设计文档

## 问题背景

当前 `train_cycle` 产生的 loss 日志和图表存在以下问题：

1. **文件散乱**：所有日志直接堆放在 `ai/ai_data/loss_logs/` 根目录，文件名只有时间戳+UUID，无法直观区分哪份是外层轮次级、哪份是第几轮的内层 epoch 级。
2. **图表可读性差**：每张图标题都是统一的 `Loss Curve - cnn_v1`，没有轮次、层级等语义信息。
3. **无法跨轮对比**：不同内层图使用各自自动缩放的 Y 轴，肉眼无法直接对比第1轮和第3轮的 loss 走势。

## 设计目标

1. **目录隔离**：每次 `train_cycle` 训练创建独立的子目录，所有相关日志（1份外层 + N份内层）收拢在一起。
2. **语义化命名**：文件名和图表标题携带足够信息（层级、轮次、总轮数、对局数等）。
3. **合并对比图**：支持把多轮 epoch 级数据画到同一张图里，统一 Y 轴，方便对比。
4. **向后兼容**：`LossLogger` 旧调用方式（仅传 `model_name`）不受影响。

## API 变更

### LossLogger.__init__

新增以下可选参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_dir` | `str` | 训练会话子目录名。提供时，日志文件存放在 `{log_dir}/{session_dir}/` 下。 |
| `log_type` | `str` | 日志层级标识。可选 `"round_level"` / `"epoch_level"`。 |
| `round_num` | `int` | 当前轮次编号（从1开始）。 |
| `total_rounds` | `int` | 总轮次数。 |
| `epochs` | `int` | 该轮次训练的总 epoch 数。 |
| `games` | `int` | 每轮对局数。 |

### 文件命名规则

| log_type | round_num | 文件名 |
|----------|-----------|--------|
| `"round_level"` | — | `round_level.jsonl` |
| `"epoch_level"` | 有 | `round_{round_num}_epoch_level.jsonl` |
| `None`（旧行为） | — | `{model_name}_{timestamp}_{uuid}.jsonl` |

### 图表标题规则

| log_type | 标题示例 |
|----------|---------|
| `"round_level"` | `Loss Curve - cnn_v1 (Round Level, 3 rounds, 50 games/round)` |
| `"epoch_level"` | `Loss Curve - cnn_v1 (Round 1/3 Epoch Level, 1000 epochs)` |
| `None` | `Loss Curve - cnn_v1` |

### 新增类方法：plot_comparison

```python
@classmethod
def plot_comparison(cls, session_dir: str, model_name: str = None, 
                    log_dir: str = None, output_path: str = None) -> str:
    """读取 session_dir 下所有 epoch_level 日志，合并绘制到一张对比图。

    Returns:
        生成的图片路径；无数据或 matplotlib 不可用时返回 None。
    """
```

合并图特征：
- 每轮一条不同颜色的折线
- X 轴为 Epoch，Y 轴为 Loss，统一 Y 轴范围
- 带图例（Legend）标识每条线属于第几轮
- 标题：`Loss Comparison - {model_name}`

## 目录结构示例

```
ai/ai_data/loss_logs/
├── cnn_v1_20260423_105200_3a68bc36/     # session_dir
│   ├── round_level.jsonl
│   ├── round_level.png
│   ├── round_1_epoch_level.jsonl
│   ├── round_1_epoch_level.png
│   ├── round_2_epoch_level.jsonl
│   ├── round_2_epoch_level.png
│   ├── round_3_epoch_level.jsonl
│   ├── round_3_epoch_level.png
│   └── comparison.png                    # 合并对比图
```

## 调用链修改

### SelfPlayCycle.run

1. 在训练开始前生成 `session_dir`：
   ```python
   session_dir = f"{self.trainer.model_name}_{timestamp}_{short_uuid}"
   ```
2. 创建外层 `LossLogger` 时传入 `session_dir` 和 `log_type="round_level"`：
   ```python
   loss_logger = LossLogger(
       model_name=self.trainer.model_name,
       session_dir=session_dir,
       log_type="round_level",
       total_rounds=rounds,
       games=games
   )
   ```
3. 每轮调用 `train_batch` 时追加轮次信息：
   ```python
   avg_loss = self.trainer.train_batch(
       sample_dirs,
       epochs=epochs,
       verbose=True,
       session_dir=session_dir,
       round_num=round_num,
       total_rounds=rounds,
       epochs_count=epochs
   )
   ```
4. 全部轮次结束后调用合并对比图：
   ```python
   LossLogger.plot_comparison(session_dir, model_name=self.trainer.model_name)
   ```

### BaseTrainer

- `train_batch`、`train_single`、`train_mini_batch` 签名增加可选参数：`session_dir`、`round_num`、`total_rounds`。
- `_setup_training` 修改逻辑：
  - 若 `self.loss_logger` 已注入（测试场景），优先使用注入实例。
  - 否则若调用方传入了 session_dir 等参数，用这些参数新建 `LossLogger`。
  - 否则回退到旧行为（默认 `LossLogger`）。

```python
def _setup_training(self, **kwargs):
    if self.loss_logger is not None:
        loss_logger = self.loss_logger
    elif kwargs:
        loss_logger = LossLogger(model_name=self.model_name, **kwargs)
    else:
        loss_logger = LossLogger(model_name=self.model_name)
    self.model.network.train()
    return loss_logger
```

## 测试策略

### 新增测试用例

1. **`test_session_dir_creates_subdirectory`**：
   - 传入 `session_dir="test_session"` 后，`get_log_path()` 应位于子目录下。

2. **`test_round_level_filename`**：
   - `log_type="round_level"` 时，文件名为 `round_level.jsonl`。

3. **`test_epoch_level_filename_with_round_num`**：
   - `log_type="epoch_level", round_num=2` 时，文件名为 `round_2_epoch_level.jsonl`。

4. **`test_backward_compatibility`**：
   - 不传入任何新参数时，文件名保持旧格式 `{model_name}_{timestamp}_{uuid}.jsonl`。

5. **`test_plot_title_with_log_type`**：
   - 验证图表标题携带轮次和层级信息（通过 mock `plt.title` 断言传入字符串）。

6. **`test_plot_comparison_generates_merged_image`**：
   - 在临时目录下创建多个 `round_N_epoch_level.jsonl`，调用 `plot_comparison`，断言生成图片且包含多条线。

7. **`test_plot_comparison_empty_returns_none`**：
   - session_dir 下无 epoch_level 文件时返回 None。

### 回归测试

- 运行 `tests/ai/test_loss_logger.py` 原有测试，确保全部通过。
- 运行全量测试 `tests/run_tests.py`，确保 474 个测试全部通过。
