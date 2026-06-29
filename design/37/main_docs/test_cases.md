# 测试用例集

测试文件位置：`tests/ai/test_cnn_v7_models.py`

---

## T37-01 ~ T37-03：网络结构变更验证

### T37-01 — 价值头仅含单个 FC 层

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例 |
| **操作** | 遍历所有模块，收集 `nn.Linear` 实例 |
| **预期** | 恰好 1 个 Linear 层：`value_fc`，且 `in_features=225`、`out_features=1` |

### T37-02 — 价值头包含独立卷积层

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例 |
| **操作** | 确认 `value_conv_out` 属性存在 |
| **预期** | `value_conv_out` 为 `nn.Conv2d`，`in_channels=32`、`out_channels=1`、`kernel_size=(1,1)` |

### T37-03 — 网络不含 value_fc1 和 value_fc2

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例 |
| **操作** | 检查 `hasattr(network, 'value_fc1')` 和 `hasattr(network, 'value_fc2')` |
| **预期** | 两者均为 `False` |

---

## T37-04 ~ T37-06：参数量与输出验证

### T37-04 — 总参数量在 38,000~39,000 之间

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例 |
| **操作** | `sum(p.numel() for p in network.parameters())` |
| **预期** | 结果 ∈ [38,000, 39,000]，精确值 38,500 |

### T37-05 — forward 输出形状不变

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例 |
| **操作** | `policy, value = network(torch.randn(4, 3, 15, 15))` |
| **预期** | `policy.shape == (4, 225)`；`value.shape == (4, 1)`；`value` 值域 ⊆ [-1, 1] |

### T37-06 — 策略头不受影响

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Network` 实例，同一输入 |
| **操作** | 运行两次 forward，对比 policy 输出 |
| **预期** | 相同输入策略输出一致（确定性），LogSoftmax 输出 ≤ 0，exp 后行和为 1 |

---

## T37-07 ~ T37-08：模型级测试

### T37-07 — predict 价值范围

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Model(fresh=True)` |
| **操作** | `result = model.predict([0]*225 + [1])` |
| **预期** | `result.type == "policy_value"`；`result.value ∈ [-1.0, 1.0]` |

### T37-08 — predict 相同输入一致性

| 项 | 内容 |
|----|------|
| **前提** | 创建 `CNNV7Model(fresh=True)` |
| **操作** | 同一输入 `board_vector` 调用两次 `predict` |
| **预期** | 两次返回的 `policy` 和 `value` 完全相等 |
