# 任务报告：新 Reward 反向传播算法实现

**任务日期**: 2026-04-04  
**任务编号**: 1  
**工作边界**: ai/ 包（ai/evaluator.py）、tests/ai/ 目录

---

## 1. 需求内容

根据 requirements/key_points.md 第 429-442 行的描述，实现新的 reward 反向传播算法：

**算法规则**:
- 假设黑方获胜，落子序列为 a1 → a2 → ... → an（win）
- 其中 a(j1)、a(j2)、a(j3)、a(j4)、a(j5) 五步构成一条五子连线
- 则：
  - reward(a(j5)) = max_reward
  - reward(a(j4)) = r * reward(a(j5))
  - reward(a(j3)) = r * reward(a(j4))
  - reward(a(j2)) = r * reward(a(j3))
  - reward(a(j1)) = r * reward(a(j2))
- **失败方不做 reward 处理**（保持初始化值）

**与旧算法的区别**:
- 旧算法：获胜方所有落子按时间顺序反向传播 reward
- 新算法：只找出构成五子连线的**5步**，对这5步进行 reward 反向传播

---

## 2. 解决思路

### 2.1 核心问题
需要实现一个 `_find_five_in_a_row` 方法，从获胜方的落子序列中找出构成五子连线的5个步骤索引。

### 2.2 算法设计
1. 获取最后一步的棋盘状态（包含所有落子）
2. 从最后一步位置出发，向四个方向（水平、垂直、主对角、副对角）搜索
3. 在每个方向上收集属于获胜方的连续落子
4. 如果某个方向上有 >= 5 个落子，返回按时间排序后的最后5个步骤索引

### 2.3 实现策略
- 采用 TDD 开发流程：先写测试，再实现代码
- 主流程测试放在 `tests/ai/primary/`
- 边缘测试放在 `tests/ai/edge/`
- 设计文档采用追加方式，不修改原有文档

---

## 3. 执行过程

### 步骤 1：阅读现有代码（已完成）
- 阅读 `ai/evaluator.py` 现有实现
- 阅读 `design/ai_design.md` 设计文档
- 阅读 `tests/ai/test_evaluator.py` 现有测试

### 步骤 2：编写设计文档（已完成）
- 创建 `design/ai_evaluator_new_reward.md`
- 包含算法描述、类设计、主流程伪代码、测试用例设计

### 步骤 3：编写主流程测试（已完成）
- 创建 `tests/ai/primary/test_evaluator_new_reward.py`
- 7 个测试用例：
  - 水平/垂直/主对角/副对角五子连线识别
  - 只给5步分配 reward（其他获胜方落子保持 min_reward）
  - 失败方保持初始化值
  - reward 衰减序列验证

### 步骤 4：编写边缘测试（已完成）
- 创建 `tests/ai/edge/test_evaluator_new_reward_edge.py`
- 9 个测试用例：
  - 五子连线步骤不连续（中间夹杂对方落子）
  - 超过五子连线（六子、七子）
  - 获胜方落子少于5步
  - 找不到五子连线
  - 白方获胜
  - 十字形连线

### 步骤 5：实现代码（已完成）
修改 `ai/evaluator.py`：
1. 新增 `_find_five_in_a_row` 方法
2. 重写 `_compute_rewards` 方法

**关键代码**:
```python
def _find_five_in_a_row(self, winning_actions, history):
    # 从获胜方落子序列中找出构成五子连线的5个步骤索引
    # 实现逻辑：从最后一步向四个方向搜索，找到 >=5 个连续落子的方向
    
def _compute_rewards(self, sample_dict, history, winner):
    # 新算法：只给构成五子连线的5步分配 reward
    # 失败方不做处理
```

### 步骤 6：修复测试数据问题（已完成）
测试中发现的主要问题：
- 测试数据构造时，棋盘更新顺序错误（先记录再更新，导致最后一步的棋盘状态不正确）
- 修复方案：先更新棋盘，再记录历史

### 步骤 7：修复原有测试（已完成）
原有测试 `test_compute_rewards_black_wins` 和 `test_custom_decay_rate_affects_reward_calculation` 使用的是旧算法预期，需要：
- 修改测试数据，使其构成真正的五子连线
- 更新预期值，符合新算法规则

### 步骤 8：运行所有测试（已完成）
- 主流程测试：7 个全部通过
- 边缘测试：9 个全部通过
- 原有 Evaluator 测试：14 个全部通过
- 所有 AI 测试：174 个全部通过

---

## 4. 执行结果

### 测试统计
| 测试文件 | 测试数量 | 结果 |
|---------|---------|------|
| tests/ai/primary/test_evaluator_new_reward.py | 7 | 全部通过 |
| tests/ai/edge/test_evaluator_new_reward_edge.py | 9 | 全部通过 |
| tests/ai/test_evaluator.py | 14 | 全部通过 |
| tests/ai/ 全部 | 174 | 全部通过 |

### 功能验证
- ✅ 水平方向五子连线识别
- ✅ 垂直方向五子连线识别
- ✅ 主对角线方向五子连线识别
- ✅ 副对角线方向五子连线识别
- ✅ 五子连线步骤不连续的情况
- ✅ 超过五子连线的情况（六子、七子）
- ✅ 只给构成连线的5步分配 reward
- ✅ 失败方不做处理（保持初始化值）
- ✅ reward 衰减序列正确（r^0, r^1, r^2, r^3, r^4）

---

## 5. 涉及文件

### 新增文件
1. `design/ai_evaluator_new_reward.md` - 新算法设计文档
2. `tests/ai/primary/test_evaluator_new_reward.py` - 主流程测试（7个测试用例）
3. `tests/ai/edge/test_evaluator_new_reward_edge.py` - 边缘测试（9个测试用例）

### 修改文件
1. `ai/evaluator.py` - 实现新算法
   - 新增 `_find_five_in_a_row` 方法
   - 重写 `_compute_rewards` 方法
2. `tests/ai/test_evaluator.py` - 修复原有测试以符合新算法

### 临时文件
- 测试过程中生成的临时目录（已自动清理）

---

## 6. 算法变更说明

### 旧算法行为
```python
# 获胜方所有落子都有 reward，从最后一步向前衰减
for step_idx, position in reversed(winning_actions):
    sample_dict[step_idx]["output"][reward_idx] = int(current_reward)
    current_reward *= self.decay_rate

# 失败方所有落子都有负向 reward
for step_idx, position in reversed(losing_actions):
    sample_dict[step_idx]["output"][reward_idx] = int(current_reward)
    current_reward *= self.decay_rate
```

### 新算法行为
```python
# 只找出构成五子连线的5步
five_steps = self._find_five_in_a_row(winning_actions, history)

# 只对这5步进行 reward 反向传播
if five_steps:
    current_reward = self.MAX_REWARD
    for step_idx in reversed(five_steps):
        sample_dict[step_idx]["output"][reward_idx] = int(current_reward)
        current_reward *= self.decay_rate

# 失败方不做处理（保持初始化值）
```

---

## 7. 后续建议

1. **fc_v1 和 cnn_v1 训练**：新算法会影响训练数据生成，建议在训练前清理旧的 sample 数据
2. **性能考虑**：当前算法时间复杂度为 O(n)，n 为获胜方落子数，性能良好
3. **边界情况**：已处理超过五子连线的情况（返回最后5步）

---

**任务完成**
