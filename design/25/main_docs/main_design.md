# 设计文档-25：MCTS 训练损失曲线图

> 意图 25 | 日期：2026-06-25 | 关联模块：mcts_train.py, dual_head_trainer.py

---

## 1. 概述

为 `mcts_train.py` 训练脚本增加训练结束后自动生成**双 Y 轴损失曲线图**的能力，与旧体系 `train_main.py` 的图表输出行为对齐。

图表在训练完成后自动生成并保存为 PNG 文件，无需用户额外操作。

---

## 2. 配置

图表输出目录通过配置文件指定，默认值为 `mcts_loss_logs`（相对于项目根目录）。

**配置位置**：`src/config/mcts.py`

**新增常量**：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MCTS_LOSS_LOG_DIR` | `"mcts_loss_logs"` | MCTS 训练损失曲线图输出目录 |

目录在图表生成时自动创建（如不存在）。

---

## 3. 图表规格

### 3.1 数据指标

图表展示以下指标随时间（epoch）的变化趋势。所有 epoch 级指标均为 epoch 内各 batch 的**均值**。

| 指标 | Y 轴 | 颜色/线型 | 聚合方式 | 说明 |
|------|------|----------|---------|------|
| `total_loss` | 左轴 | 蓝实线 | epoch 内各 batch 均值 | \( \text{value\_loss} + \text{policy\_loss} \) |
| `value_loss` | 左轴 | 青虚线 (#17becf) | epoch 内各 batch 均值 | 价值头 MSE 损失 |
| `policy_loss` | 左轴 | 橙虚线 (#ff7f0e) | epoch 内各 batch 均值 | 策略头交叉熵损失 |
| `val_total_loss` | 左轴 | 绿实线 | —（验证集只前向一次） | 仅当 `--val-split > 0` 时存在 |
| 梯度范数 | 右轴 | 红实线 | epoch 内各 batch 的 `sum(grad_norms.values())` 取均值 | 所有参数的 L2 梯度范数均值 |

### 3.2 图表布局

```
┌──────────────────────────────────────────────────────┐
│  Loss Curve - cnn_v7 (Epoch Level, N epochs)         │
│                                                      │
│  Loss ▲                                        Grad  │
│       │  ── total_loss (蓝实线)               Norm  │
│       │  - - value_loss (青虚线)                     │
│       │  - - policy_loss (橙虚线)                    │
│       │  ── val_total_loss (绿实线)                  │
│       │                                    ░░░░ ── (红实线)
│       │                              ░░░░░░░░░░░░░░░ │
│       │                        ░░░░░░░░░░░░░░░░░░░░░ │
│       │                  ░░░░░░                      │
│       │            ░░░░░░                            │
│       │      ░░░░░░                                  │
│       │░░░░░░                                        │
│       └──────────────────────────────────▶ Epoch     │
└──────────────────────────────────────────────────────┘
```

- **左 Y 轴**（蓝色）：Loss，标签 "Loss"
  - `total_loss`：蓝色实线
  - `value_loss`：青色虚线 (#17becf)
  - `policy_loss`：橙色虚线 (#ff7f0e)
  - `val_total_loss`：绿色实线（仅验证集存在时）
- **右 Y 轴**（红色）：Gradient Norm，标签 "Gradient Norm"
- **X 轴**：Epoch 编号（1-based）
- **图例**：合并左右轴图例，置于右上角
- **网格**：虚线网格（左轴）
- **图片尺寸**：12×7 英寸，DPI 150

### 3.3 输出位置

图表保存为 PNG 文件，路径格式：

```
{MCTS_LOSS_LOG_DIR}/{model_name}_{timestamp}.png
```

示例：`mcts_loss_logs/cnn_v7_20260625_133000.png`

---

## 4. 验收用例

### 4.1 基本功能

| 编号 | 场景 | 前置条件 | 预期结果 |
|------|------|---------|---------|
| AC-1 | 无验证集训练后生成图表 | `--val-split 0`，训练完成 | 图表包含蓝线（total_loss）和红线（grad_norm），无绿线（val_total_loss） |
| AC-2 | 有验证集训练后生成图表 | `--val-split 0.1`，训练完成 | 图表包含蓝线（total_loss）、绿线（val_total_loss）、红线（grad_norm） |
| AC-3 | 图表文件存在 | 训练完成后 | `mcts_loss_logs/` 目录下存在对应的 `.png` 文件 |
| AC-4 | 图表自动创建目录 | `mcts_loss_logs/` 目录不存在 | 训练完成后自动创建目录并生成图表 |

### 4.2 数据正确性

| 编号 | 场景 | 验证方法 | 预期结果 |
|------|------|---------|---------|
| AC-5 | total_loss 与日志一致 | 对比图表中最后一个 epoch 的 total_loss 值与控制台输出的最终 total_loss | 数值一致（误差 < 0.001） |
| AC-6 | val_total_loss 仅在有验证集时出现 | 分别运行 `--val-split 0` 和 `--val-split 0.1` | 前者无绿线，后者有绿线 |
| AC-7 | 梯度范数 > 0 | 查看图表中 grad_norm 曲线 | 所有 epoch 的 grad_norm 值 > 0（训练中梯度不应全零） |

### 4.3 兼容性

| 编号 | 场景 | 预期结果 |
|------|------|---------|
| AC-8 | 训练 Ctrl+C 中断 | 打印 `[中断] 检测到 Ctrl+C，正在生成图表...`，已完成的 epoch 数据生成图表（与旧体系行为一致） |
| AC-9 | matplotlib 不可用 | 训练正常完成，跳过图表生成，不报错不崩溃 |

---

## 5. 变更范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/config/mcts.py` | 修改 | 新增 `MCTS_LOSS_LOG_DIR` 常量 |
| `src/ai/dual_head_trainer.py` | 修改 | `train_epochs` 中聚合 epoch 级梯度范数（均值）；新增 `_plot_loss_curve(history)` 私有方法；`train_epochs` 末尾 + `except KeyboardInterrupt` 调用绘图 |
| `src/ai/scripts/mcts_train.py` | 无修改 | 图表生成由 `trainer.train_epochs()` 内部自动完成 |

---

## 6. 实现约束

1. **梯度范数聚合**：epoch 级取各 batch `sum(grad_norms.values())` 的**均值**
2. **绘图方法**：`DualHeadTrainer` 新增私有方法 `_plot_loss_curve(history)`，不复用旧体系 `LossLogger`
3. **matplotlib 降级**：`try/except ImportError` + `Agg` 后端（参照 `loss_logger.py` 第 15-23 行）
4. **中断处理**：`train_epochs` 中 `except KeyboardInterrupt` 打印提示后调用 `_plot_loss_curve`
5. **`_train_step` 不修改**：grad_norms 仍返回 per-parameter dict，epoch 级聚合全在 `train_epochs` 内完成

---

## 7. 关联文档

- 意图 24 设计文档：[design/24/main_docs/main_design.md](../24/main_docs/main_design.md) — MCTS 训练体系总体设计
- `train_main.py` 图表实现：`src/ai/loss_logger.py` → `LossLogger.plot()` — 旧体系双 Y 轴图表参考
