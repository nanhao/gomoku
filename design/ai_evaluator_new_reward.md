# Evaluator 新 Reward 反向传播算法设计文档

**版本**: v1.0  
**创建日期**: 2026-04-04  
**关联文档**: [ai_design.md](ai_design.md) - 基础 Evaluator 设计

---

## 1. 概述

本文档描述 Evaluator 模块的新 reward 反向传播算法。该算法与旧算法的核心区别在于：

- **旧算法**: 获胜方所有落子按时间顺序反向传播 reward
- **新算法**: 只找出构成五子连线的**5步**，对这5步进行反向传播 reward

---

## 2. 算法描述

### 2.1 核心规则

假设一局游戏中：
- 黑方（获胜）：a1 → a2 → a3 → a4 → ... → an（win）
- 白方（失败）：b1 → b2 → b3 → ... → b(n-1)

其中，a(j1)、a(j2)、a(j3)、a(j4)、a(j5) 这五步的落子构成了一条线，则：

```
reward(a(j5)) = max_reward
reward(a(j4)) = r * reward(a(j5))
reward(a(j3)) = r * reward(a(j4))
reward(a(j2)) = r * reward(a(j3))
reward(a(j1)) = r * reward(a(j2))
```

**约束条件**:
- j5 = n（最后一步）
- j1 < j2 < j3 < j4 < j5（按时间顺序）
- j1,j2,j3,j4,j5 不一定连续（中间可能有对方落子）

### 2.2 失败方处理

**失败方的动作不做 reward 处理**，保持初始化值：
- 已落子位置: `min_reward`
- 空位: `0`

---

## 3. 类设计

### 3.1 Evaluator 类扩展

**文件**: `ai/evaluator.py`

在现有 Evaluator 类基础上，新增/修改以下方法：

| 方法 | 类型 | 说明 |
|------|------|------|
| `_compute_rewards` | 修改 | 重写 reward 计算逻辑，调用 `_find_five_in_a_row` |
| `_find_five_in_a_row` | 新增 | 从获胜方落子序列中找出构成五子连线的5步索引 |

### 3.2 _find_five_in_a_row 方法

**职责**: 从获胜方的落子序列中，找出构成五子连线的5个步骤索引

**输入**:
- `winning_actions`: List[Tuple[int, Tuple[int, int]]] - 获胜方落子列表，每项为 (step_index, (x, y))
- `history`: List[Dict] - 完整历史记录

**输出**:
- `List[int]` - 构成五子连线的5个步骤索引 [j1, j2, j3, j4, j5]，如果找不到返回空列表

**算法逻辑**:
```
函数 _find_five_in_a_row(winning_actions, history):
    如果 winning_actions 少于 5 步:
        返回空列表
    
    获取最后一步的落子位置 (x, y) 和步骤索引 last_step_idx
    从 history[last_step_idx] 获取最终棋盘状态
    
    四个方向 = [(0,1), (1,0), (1,1), (1,-1)]  // 水平、垂直、主对角、副对角
    
    对于每个方向 (dx, dy):
        从最后一步位置 (x, y) 出发，沿该方向找出所有属于获胜方的落子
        使用棋盘状态确认这些位置确实都是获胜方的棋子
        
        如果该方向上有至少 5 个获胜方棋子:
            按时间顺序排序这 5+ 个落子
            返回最后 5 个的 step_index [j1, j2, j3, j4, j5]
    
    返回空列表  // 未找到五子连线
```

---

## 4. 主流程伪代码

### 4.1 新的 _compute_rewards 流程

```
函数 _compute_rewards(sample_dict, history, winner):
    // 分离黑白双方落子
    black_actions = []
    white_actions = []
    对于每个 step in history:
        position = step.position
        player = step.input[225]  // 最后一位是当前落子方
        如果 player == 1:
            black_actions.append((step_index, position))
        否则:
            white_actions.append((step_index, position))
    
    // 确定获胜方
    如果 winner == 1:
        winning_actions = black_actions
    否则:
        winning_actions = white_actions
    
    // 找出构成五子连线的5步
    five_steps = _find_five_in_a_row(winning_actions, history)
    
    // 对这5步进行 reward 反向传播
    如果 five_steps 不为空:
        current_reward = MAX_REWARD
        对于 step_idx in reversed(five_steps):  // 从 j5 到 j1
            position = history[step_idx].position
            reward_idx = position[0] * 15 + position[1]
            sample_dict[step_idx].output[reward_idx] = int(current_reward)
            current_reward = current_reward * decay_rate
    
    // 失败方不做处理（保持初始化值）
```

---

## 5. 测试用例

### 5.1 主流程测试用例

**文件**: `tests/ai/primary/test_evaluator_new_reward.py`

```python
class TestEvaluatorNewReward:
    """新 reward 算法主流程测试"""
    
    def test_find_five_in_a_row_horizontal(self):
        """测试识别水平方向五子连线"""
        # 构造获胜方水平五子连线的历史记录
        # 黑方在 (7,3), (7,4), (7,5), (7,6), (7,7) 形成五子连线
        # 验证 _find_five_in_a_row 返回正确的5个步骤索引
        pass
    
    def test_find_five_in_a_row_vertical(self):
        """测试识别垂直方向五子连线"""
        # 构造获胜方垂直五子连线的历史记录
        pass
    
    def test_find_five_in_a_row_diagonal(self):
        """测试识别对角线方向五子连线"""
        # 构造获胜方对角线五子连线的历史记录
        pass
    
    def test_compute_rewards_only_five_steps(self):
        """测试只给构成五子连线的5步分配 reward"""
        # 黑方获胜，共下了9步，其中5步构成连线
        # 验证只有这5步有正向 reward，其他获胜方落子保持 0
        pass
    
    def test_compute_rewards_losing_side_unchanged(self):
        """测试失败方 reward 保持初始化值"""
        # 验证失败方的落子位置保持 min_reward，空位保持 0
        pass
    
    def test_reward_decay_sequence(self):
        """测试 reward 衰减序列正确"""
        # j5 = MAX_REWARD
        # j4 = MAX_REWARD * r
        # j3 = MAX_REWARD * r^2
        # j2 = MAX_REWARD * r^3
        # j1 = MAX_REWARD * r^4
        pass
```

### 5.2 边缘测试用例

**文件**: `tests/ai/edge/test_evaluator_new_reward_edge.py`

```python
class TestEvaluatorNewRewardEdge:
    """新 reward 算法边缘测试"""
    
    def test_five_steps_not_consecutive(self):
        """测试五子连线步骤不连续的情况"""
        # 获胜方落子序列：a1, a2, a3, a4, a5, a6, a7, a8, a9
        # 其中 a1, a3, a5, a7, a9 构成五子连线（不连续）
        pass
    
    def test_more_than_five_in_line(self):
        """测试超过五子连线的情况（六子、七子）"""
        # 获胜方形成六子连线
        # 应返回构成连线的最后5步
        pass
    
    def test_no_five_in_a_row_found(self):
        """测试找不到五子连线的情况（异常情况）"""
        # 理论上不会出现，因为游戏已结束且有获胜方
        pass
    
    def test_five_steps_with_interleaved_opponent(self):
        """测试五子连线中间夹杂对方落子"""
        # 黑：a1, 白：b1, 黑：a2, 白：b2, 黑：a3, 白：b3, 黑：a4, 白：b4, 黑：a5（获胜）
        # a1-a5 构成五子连线，中间夹杂白方落子
        pass
```

---

## 6. 与旧算法的差异对比

| 场景 | 旧算法 | 新算法 |
|------|--------|--------|
| 获胜方 reward | 所有落子都有 reward | 只有构成五子连线的5步有 reward |
| 失败方 reward | 从 min_reward 反向衰减 | 不做处理（保持初始化值） |
| 适用模型 | fc_v1, cnn_v1 旧版本 | fc_v1, cnn_v1 新版本 |

---

## 7. 实现顺序

1. 编写设计文档（本文档）
2. 编写主流程测试用例
3. 编写边缘测试用例
4. 实现 `_find_five_in_a_row` 方法
5. 修改 `_compute_rewards` 方法
6. 运行所有测试并修复问题
7. 生成任务报告

---

**文档结束**
