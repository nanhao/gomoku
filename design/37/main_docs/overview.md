# 设计文档：意图 37 — 价值头轻量化（分支独立 conv_out）

## 1. 概述

优化 cnn_v7 价值头网络结构，将价值头从共享 `conv_out` 后分支改为在共享特征层独立分支，大幅降低参数量。

**变更类型**：模型结构优化（cnn_v7 直接替换）  
**影响范围**：`CNNV7Network.__init__`、`CNNV7Network.forward`  
**接口变化**：无（`forward` 签名不变）

## 2. 现状

价值头在当前 `conv_out`（32→1, 1×1 卷积）之后分支，通过两层 FC 输出：

```
共享特征(32ch) → conv_out(32→1) → Flatten(225) →
  ├── LogSoftmax → 策略
  └── FC(225→128)→ReLU→FC(128→1)→Tanh → 价值   | 29,057 参数
```

**问题**：价值头 FC 层占 ~29K 参数（总参数 ~67K 的 43%），而价值头任务（标量回归）远比策略头简单。

## 3. 新设计

价值头在 `conv_out` **之前**分支，拥有独立的 1×1 卷积层：

```
共享特征(32ch) →
  ├── conv_out(32→1) → Flatten(225) → LogSoftmax → 策略       | 33 参数
  └── value_conv_out(32→1) → Flatten(225) → ReLU → FC(225→1) → Tanh → 价值  | 259 参数
```

**关键变化**：
| 项 | 旧 | 新 |
|----|----|----|
| 价值头分支点 | `conv_out` 之后（1 通道特征） | `conv_out` 之前（32 通道特征） |
| 价值头结构 | FC(225→128) + ReLU + FC(128→1) + Tanh | Conv2d(32→1,1×1) + ReLU + FC(225→1) + Tanh |
| 价值头参数量 | 29,057 | 259 |
| 总参数量 | 67,298 | 38,500 |

**设计依据**：
- 价值头独立 `conv_out` 使其能从 32 通道特征中学习价值专属的空间表示，而非依赖策略头压缩后的单通道特征
- 单层 FC(225→1) 对 15×15 棋盘的局面估值已足够（对标 AlphaGo Zero 的价值头设计）

## 4. 接口设计

### CNNV7Network

对外接口不变：

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `forward(x)` | `(batch, 3, 15, 15)` 张量 | `(policy, value)` 元组 | policy: `(batch, 225)` LogSoftmax; value: `(batch, 1)` Tanh |

**内部层变更**：
- 移除：`value_fc1`、`value_fc2`
- 新增：`value_conv_out`（Conv2d 32→1, 1×1）、`value_fc`（Linear 225→1）
- 保留（重命名）：`value_relu`（位置从 FC 层间移至 Flatten 之后）

## 5. 关键流程

参见架构图：[architecture.puml](architecture.puml)

## 6. 类图与依赖

参见类图：[class_diagram.puml](class_diagram.puml)

## 7. 相关文件

| 文件 | 变更 |
|------|------|
| `src/ai/models/cnn_v7/model.py` | 修改 `CNNV7Network.__init__` 和 `forward` |
| `tests/ai/test_cnn_v7_models.py` | 更新参数计数和结构断言测试 |
