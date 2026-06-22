# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/test_cnn_models.py` | 单元测试 | 新增 `TestCNNV6Network` 和 `TestCNNV6Model` 两个测试类 |

> cnn_v6 遵循与已有 CNN 模型完全相同的测试模式，在同一个测试文件中追加测试类即可，无需新建测试文件。

---

## 2. 网络结构测试（TestCNNV6Network）

**继承**：`CNNNetworkSharedMixin` + `unittest.TestCase`  
**类属性**：`NETWORK_CLS = CNNV6Network`

`CNNNetworkSharedMixin` 提供以下共享测试：
- `test_network_structure`：前向输出形状为 (batch, 225)
- `test_network_log_softmax_output_range`：LogSoftmax 输出 ≤ 0，exp 后在 (0, 1)
- `test_network_log_softmax_row_sum`：exp(输出) 每行总和为 1
- `test_network_forward_single`：单样本前向
- `test_network_forward_batch`：不同 batch size 前向

### 2.1 v6 特有结构测试

| 用例编号 | 用例描述 | 验证点 |
|----------|---------|--------|
| UT-NET-1 | 网络中 **包含** BatchNorm 层 | 遍历 `network.modules()`，至少存在 1 个 `nn.BatchNorm2d` 实例（与 v3/v4/v5 的 `test_network_no_bn` 相反） |
| UT-NET-2 | 网络中 **不包含** Dropout 层 | 遍历 `network.modules()`，不存在 `nn.Dropout` 实例 |
| UT-NET-3 | 网络中 **不包含** FC 层 | 遍历 `network.modules()`，不存在 `nn.Linear` 实例 |
| UT-NET-4 | BN 层数量 = 5 | 遍历 `network.modules()`，统计 `nn.BatchNorm2d` 数量，期望 5（Conv1~5 各一个 BN） |
| UT-NET-5 | 卷积通道数 | `conv1`: 3→32; `conv2~5`: 32→32; `conv_out`: 32→1, kernel_size=(1,1) |
| UT-NET-6 | 参数量范围 | `sum(p.numel())` 应在 37,000～41,000 之间（精确值 38,241） |
| UT-NET-7 | 残差连接存在性验证 | 构造一个简单输入张量，对比：① 正常 forward；② 手动移除残差连接后的输出（将 Conv3 的 BN 输出中减去 Conv2 的输入，或直接对比残差块输入输出是否相等）—正常 forward 输出应不同于移除残差的输出 |

### 2.2 残差连接完整性测试

| 用例编号 | 用例描述 | 输入 | 预期结果 |
|----------|---------|------|---------|
| UT-RES-1 | 残差块 1：输入输出形状一致 | (1, 3, 15, 15) 随机张量 | 遍历残差块 1 的输入和输出，shape 均为 (1, 32, 15, 15) |
| UT-RES-2 | 残差块 2：输入输出形状一致 | 同上 | 残差块 2 输入输出 shape 均为 (1, 32, 15, 15) |
| UT-RES-3 | 残差连接对梯度的影响 | 反向传播后 | Conv2.weight.grad 不为 None 且不为全零 |

---

## 3. 模型行为测试（TestCNNV6Model）

**继承**：`CNNModelSharedMixin` + `unittest.TestCase`  
**类属性**：`MODEL_CLS=CNNV6Model`, `MODEL_NAME="cnn_v6"`, `DEFAULT_FILENAME="cnn_v6.pth"`

`CNNModelSharedMixin` 提供以下共享测试：
- `test_model_name`：name 属性正确
- `test_model_predict_shape`：输出长度 225
- `test_model_predict_format`：元素均为 float
- `test_model_predict_probability_range`：值域 (0, 1)
- `test_model_predict_probability_sum`：总和 ≈ 1
- `test_model_predict_consistency`：相同输入输出一致
- `test_model_save_and_load` / `test_model_save_default_path` / `test_default_load_when_model_exists` 等：save/load 往返
- `test_fresh_parameter_skips_load`：fresh=True 跳过加载
- `test_vector_to_tensor_conversion`：输入格式转换正确

### 3.1 v6 特有模型测试

| 用例编号 | 用例描述 | 验证点 |
|----------|---------|--------|
| UT-MOD-1 | `model.name` 返回 `"cnn_v6"` | 字符串相等 |
| UT-MOD-2 | fresh=True 不加载任何权重 | 创建 `CNNV6Model(fresh=True)`，检查 model.network 的参数为随机初始化（非全零） |
| UT-MOD-3 | eval 模式下 BN 使用全局统计量 | `model.network.eval()` 后，多次 forward 相同输入，输出完全一致（无随机性） |
| UT-MOD-4 | train 模式下 BN 使用 batch 统计量 | `model.network.train()` 后，多次 forward 相同输入，输出可能有微小差异（取决于 BN 的 batch 统计） |

---

## 4. 集成测试

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| UT-INT-1 | `get_model("cnn_v6.1")` 返回 CNNV6Model 实例 | 创建 gen1 目录及权重文件 | 返回对象 `isinstance(CNNV6Model)`，`gen==1`，`full_name=="cnn_v6.1"` |
| UT-INT-2 | `get_trainer("cnn_v6.1")` 返回 CNNV6Trainer 实例 | 同上 | 返回对象 `isinstance(CNNV6Trainer)` |
| UT-INT-3 | `get_model("cnn_v6.1").predict(board)` 不报错 | gen1 权重存在 | 正常返回 225 维列表 |
| UT-INT-4 | `list_models_with_full_name()` 包含 `cnn_v6.N` | gen1～genN 存在 .pth | 输出列表中包含 `cnn_v6.1` 等条目 |
| UT-INT-5 | 与其他模型共存不冲突 | 已有 v1～v5 的测试通过 | 全量测试 `python -m pytest tests/` 全部通过 |

---

## 5. 验收用例集（与 PRD §5 对齐）

### AC-1：文件结构完整性

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-1.1 "目录存在" | 代码审查：检查 `ai/models/cnn_v6/` 含三个文件 |
| AC-1.2 "模块可导入" | UT-INT-1（`get_model` 成功即验证导入链） |
| AC-1.3 "注册表可解析" | UT-INT-1, UT-INT-2 |

### AC-2：网络结构正确性

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-2.1 "包含 BN 层" | UT-NET-1, UT-NET-4 |
| AC-2.2 "卷积通道数" | UT-NET-5 |
| AC-2.3 "无 FC 层" | UT-NET-3 |
| AC-2.4 "无 Dropout" | UT-NET-2 |
| AC-2.5 "残差连接存在" | UT-RES-1, UT-RES-2, UT-RES-3 |
| AC-2.6 "参数量范围" | UT-NET-6 |
| AC-2.7 "输出层为 LogSoftmax" | `CNNNetworkSharedMixin.test_network_log_softmax_output_range`、`test_network_log_softmax_row_sum` |

### AC-3：模型功能正确性

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-3.1 "predict 输出格式" | `test_model_predict_shape`、`test_model_predict_probability_range`、`test_model_predict_probability_sum` |
| AC-3.2 "predict 可被已占位过滤" | 由调用方 mask 逻辑保证（不在本测试范围） |
| AC-3.3 "save/load 往返一致" | `test_model_save_and_load`、`test_default_load_when_model_exists` |
| AC-3.4 "forward 输出 ≤ 0" | `test_network_log_softmax_output_range`、`test_network_log_softmax_row_sum` |

### AC-4：训练可用性

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-4.1 "小规模训练不报错" | 手动验证：用少量样本训练 5 epoch |
| AC-4.2 "梯度正常流动" | `test_network_structure`（前向通过即隐含验证梯度路径存在）；额外在 v6 特有测试 UT-RES-3 中显式验证 |
| AC-4.3 "BN 处于训练模式" | UT-MOD-3, UT-MOD-4 |

### AC-5：测试覆盖

| PRD ID | 覆盖用例 |
|--------|---------|
| AC-5.1 "网络结构测试类" | TestCNNV6Network（全部 UT-NET-*, UT-RES-*） |
| AC-5.2 "模型行为测试类" | TestCNNV6Model（全部 UT-MOD-*） |
| AC-5.3 "全量测试通过" | UT-INT-5 |

---

## 6. Mock 策略

本次新增测试**不使用 Mock**。所有测试直接构造网络/模型实例并调用真实方法：

- **CNNV6Network 测试**：直接实例化 `CNNV6Network()`，传入随机 Tensor 验证 forward 行为
- **CNNV6Model 测试**：直接实例化 `CNNV6Model(fresh=True)`，验证 predict/save/load
- **集成测试**：在临时目录下创建 gen 子目录和虚拟 .pth 文件，使用真实 `get_model`/`get_trainer` 验证注册表集成
