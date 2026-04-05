# FC v1 Softmax 与 Reward [0,1] 值域调整任务报告

## 1. 需求内容

根据 `key_points.md` 第545-548行，完成以下三项修改：

1. **fc_v1 的网络结构最后加一层 softmax**
2. **trainer.py 在计算 loss 的时候，把 reward 归一化到 0-1 之间**
3. **热力图的 reward 值域调整 0-1**

## 2. 解决思路

### 2.1 Softmax 层添加
- 在 `FCv1Network` 的 `__init__` 中添加 `nn.Softmax(dim=1)` 层
- 在 `forward()` 方法最后应用 softmax，使输出值域为 (0, 1) 且每行和为 1
- 修改 `predict()` 方法，将 softmax 输出映射回 [-1000000, 1000000] 保持接口兼容

### 2.2 Reward 归一化调整
- 原逻辑：将 [-1000000, 1000000] 映射到 [-1, 1]
- 新逻辑：将 [-1000000, 1000000] 映射到 [0, 1]
- 映射公式：`(reward + 1000000) / 2000000`
- 即：-1000000 -> 0, 0 -> 0.5, 1000000 -> 1

### 2.3 热力图值域调整
- 原逻辑：值域 [-1000000, 1000000]，0 对应白色
- 新逻辑：值域 [0, 1]，0.5 对应白色
- 颜色映射：0 -> 黑色, 0.5 -> 白色, 1 -> 红色

## 3. 执行过程

### 步骤 1: 查看现有测试文件
- 阅读 `tests/ai/test_fc_v1_model.py` 了解模型测试覆盖
- 阅读 `tests/ai/test_fc_v1_trainer.py` 了解训练器测试覆盖
- 阅读 `tests/gui/test_heatmap.py` 了解热力图测试覆盖

### 步骤 2: 编写设计文档
- 创建 `design/fc_v1_softmax_and_reward_0_1_design.md`
- 包含伪代码、主流程测试用例、边缘测试用例

### 步骤 3: 修改 model.py 添加 Softmax
- 在 `FCv1Network.__init__()` 添加 `self.softmax = nn.Softmax(dim=1)`
- 在 `forward()` 最后应用 softmax
- 修改 `predict()` 添加映射逻辑：`(output - 0.5) * 2 * 1000000`

### 步骤 4: 修改 trainer.py 归一化到 [0,1]
- 修改常量定义：`REWARD_MIN = -1_000_000.0`, `REWARD_MAX = 1_000_000.0`, `REWARD_RANGE = 2_000_000.0`
- 修改归一化公式：`(targets_tensor - self.REWARD_MIN) / self.REWARD_RANGE`

### 步骤 5: 修改 heatmap.py 值域到 [0,1]
- 修改常量：`REWARD_MIN = 0.0`, `REWARD_MAX = 1.0`
- 修改 `_get_reward_color()`：将值限制在 [0, 1] 并映射到 40 档颜色

### 步骤 6: 更新测试用例
- `test_fc_v1_model.py`: 添加 `test_network_softmax_output_range` 和 `test_network_softmax_row_sum`
- `test_fc_v1_trainer.py`: 更新归一化测试，验证 [-1000000, 0, 1000000] 映射到 [0, 0.5, 1]
- `test_heatmap.py`: 更新颜色映射测试，验证 0->黑色, 0.5->白色, 1->红色

### 步骤 7: 跑通全量测试
- 模型测试：22 个全部通过
- 训练器测试：43 个全部通过
- 热力图测试：22 个通过（1 个 tkinter 环境问题，非代码问题）

## 4. 执行结果

### 修改文件
1. `ai/models/fc_v1/model.py` - 添加 Softmax 层，修改 predict 映射
2. `ai/models/fc_v1/trainer.py` - reward 归一化到 [0,1]
3. `gui/components/heatmap.py` - 值域调整到 [0,1]

### 测试文件更新
1. `tests/ai/test_fc_v1_model.py` - 添加 Softmax 相关测试
2. `tests/ai/test_fc_v1_trainer.py` - 更新归一化测试
3. `tests/gui/test_heatmap.py` - 更新颜色映射测试

### 设计文档
- `design/fc_v1_softmax_and_reward_0_1_design.md`

### 测试结果
```
tests/ai/test_fc_v1_model.py: 22 passed
tests/ai/test_fc_v1_trainer.py: 43 passed  
tests/gui/test_heatmap.py: 22 passed, 1 error (tkinter 环境问题)
```

## 5. 涉及文件清单

### 核心修改文件
- `d:/projects/gomoku/ai/models/fc_v1/model.py`
- `d:/projects/gomoku/ai/models/fc_v1/trainer.py`
- `d:/projects/gomoku/gui/components/heatmap.py`

### 测试文件
- `d:/projects/gomoku/tests/ai/test_fc_v1_model.py`
- `d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py`
- `d:/projects/gomoku/tests/gui/test_heatmap.py`

### 设计文档
- `d:/projects/gomoku/design/fc_v1_softmax_and_reward_0_1_design.md`

### 任务报告
- `d:/projects/gomoku/task_reports/2026-04-05/fc_v1_softmax_reward_0_1_调整_1.md`
