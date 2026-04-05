# FC v1 Softmax 与 Reward [0,1] 值域调整设计文档

## 需求概述

根据 key_points.md 第545-548行，需要完成以下三项修改：

1. fc_v1 的网络结构最后加一层 softmax
2. trainer.py 在计算 loss 的时候，把 reward 归一化到 0-1 之间
3. 热力图的 reward 值域调整 0-1

## 设计决策

### 1. FCv1Network 添加 Softmax

**修改位置**: `ai/models/fc_v1/model.py`

**修改内容**:
- 在 `FCv1Network.__init__()` 中添加 `self.softmax = nn.Softmax(dim=1)`
- 在 `forward()` 方法最后对输出应用 softmax

**影响分析**:
- softmax 输出范围是 (0, 1)，且所有位置的概率和为 1
- 这与之前线性输出的语义完全不同，需要同步调整 trainer 的归一化逻辑
- predict() 方法需要将 softmax 输出映射回 [-1000000, 1000000] 范围以保持接口兼容性

### 2. Trainer Reward 归一化到 [0,1]

**修改位置**: `ai/models/fc_v1/trainer.py`

**修改内容**:
- 修改 `REWARD_NORMALIZE_FACTOR` 或调整归一化公式
- 原逻辑: `targets_tensor = targets_tensor / 1_000_000.0` (映射到 [-1, 1])
- 新逻辑: `targets_tensor = (targets_tensor + 1_000_000.0) / 2_000_000.0` (映射到 [0, 1])

**归一化映射**:
- -1000000 -> 0
- 0 -> 0.5
- 1000000 -> 1

### 3. Heatmap 值域调整到 [0,1]

**修改位置**: `gui/components/heatmap.py`

**修改内容**:
- 修改 `GLOBAL_MAX_ABS` 常量或调整颜色映射逻辑
- 原逻辑: 值域 [-1000000, 1000000]，0 对应白色
- 新逻辑: 值域 [0, 1]，0.5 对应白色（或调整颜色映射使 0->黑, 1->红）

**颜色映射调整**:
- 由于 softmax 输出范围是 [0, 1]，且是概率分布
- 需要重新定义颜色映射：0 对应黑色，1 对应红色，中间渐变
- 或者保持 0.5 为白色，向两边渐变

## 主流程伪代码

### FCv1Network.forward()
```python
def forward(x):
    # 隐藏层 1-5
    x = relu(fc1(x))
    x = relu(fc2(x))
    x = relu(fc3(x))
    x = relu(fc4(x))
    x = relu(fc5(x))
    
    # 输出层
    x = fc_output(x)
    
    # 新增: softmax 激活
    x = softmax(x, dim=1)
    
    return x  # 范围 (0, 1)，总和为 1
```

### FCv1Model.predict()
```python
def predict(board_vector):
    # 前向传播
    output = network(input)
    # output 范围 (0, 1)
    
    # 映射到 [-1000000, 1000000] 保持接口兼容
    # 但注意: softmax 输出是概率，需要重新思考映射方式
    # 方案: 将 [0, 1] 映射到 [-1000000, 1000000]
    # 0 -> -1000000, 0.5 -> 0, 1 -> 1000000
    reward = (output - 0.5) * 2 * 1000000
    
    return reward as int list
```

### Trainer._prepare_tensors()
```python
def _prepare_tensors(sample_data):
    # 加载 input, output, mask
    # output 原始值范围 [-1000000, 1000000]
    
    # 归一化到 [0, 1]
    # -1000000 -> 0, 0 -> 0.5, 1000000 -> 1
    targets_tensor = (targets_tensor + 1000000.0) / 2000000.0
    
    return inputs_tensor, targets_tensor, masks_tensor
```

### Heatmap._get_reward_color()
```python
def _get_reward_color(reward):
    # reward 范围 [0, 1]（来自 softmax 输出）
    # 或者保持 [-1000000, 1000000] 但通过 trainer 归一化到 [0,1]
    
    # 新映射: 0 -> 黑色, 1 -> 红色
    # 或者: 0 -> 黑色, 0.5 -> 白色, 1 -> 红色（保持对称）
    
    # 方案: 使用 0.5 为中点，向两边渐变
    normalized = reward  # 假设输入已经是 [0, 1]
    level = normalized * 40  # [0, 1] -> [0, 40]
    
    # 颜色插值...
```

## 主流程测试用例

### 测试 1: FCv1Network Softmax 输出
```python
def test_network_softmax_output():
    network = FCv1Network()
    input_tensor = torch.randn(2, 226)
    output = network(input_tensor)
    
    # 验证输出范围 (0, 1)
    assert torch.all(output > 0)
    assert torch.all(output < 1)
    
    # 验证每行和为 1
    row_sums = output.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(2), atol=1e-6)
```

### 测试 2: Trainer Reward 归一化到 [0,1]
```python
def test_reward_normalization_to_0_1():
    trainer = FCv1Trainer(fresh=True)
    sample_data = [{
        "input": [0] * 226,
        "output": [1000000] + [-1000000] + [0] * 223,
        "mask": [1] * 225
    }]
    
    _, targets_tensor, _ = trainer._prepare_tensors(sample_data)
    
    # 1000000 -> 1.0
    assert abs(targets_tensor[0][0].item() - 1.0) < 1e-5
    # -1000000 -> 0.0
    assert abs(targets_tensor[0][1].item() - 0.0) < 1e-5
    # 0 -> 0.5
    assert abs(targets_tensor[0][2].item() - 0.5) < 1e-5
```

### 测试 3: Heatmap [0,1] 值域颜色映射
```python
def test_heatmap_color_mapping_0_to_1():
    heatmap = HeatmapCanvas(root)
    
    # 测试 0 对应黑色
    color_0 = heatmap._get_reward_color(0)
    assert color_0 == "#000000"
    
    # 测试 1 对应红色
    color_1 = heatmap._get_reward_color(1)
    assert color_1 == "#FF0000"
    
    # 测试 0.5 对应白色
    color_half = heatmap._get_reward_color(0.5)
    assert color_half == "#FFFFFF"
```

## 边缘测试用例

1. **Softmax 数值稳定性**: 测试极大/极小输入值时的 softmax 行为
2. **归一化边界值**: 测试 -1000000 和 1000000 的精确映射
3. **Heatmap 非法值**: 测试超出 [0,1] 范围的输入处理
4. **预测接口兼容性**: 验证 predict() 仍返回 [-1000000, 1000000] 范围的整数

## 文件清单

### 修改文件
- `ai/models/fc_v1/model.py` - 添加 softmax
- `ai/models/fc_v1/trainer.py` - reward 归一化到 [0,1]
- `gui/components/heatmap.py` - 值域调整到 [0,1]

### 测试文件
- `tests/ai/test_fc_v1_model.py` - 添加 softmax 相关测试
- `tests/ai/test_fc_v1_trainer.py` - 更新归一化测试
- `tests/gui/test_heatmap.py` - 更新颜色映射测试
