# 五连追溯 Reward 反向传播算法实现

## 需求内容

根据 `requirements/key_points.md` 第十二次追加需求：
- 实现五连追溯的 Reward 反向传播算法
- 获胜方五连的那 5 步落子获得额外奖励
- 奖励按衰减系数 r 反向传播
- 与现有棋型评估 reward 叠加

## 解决思路

1. **设计决策**
   - 衰减系数 r = 0.9（复用 `Evaluator.DECAY_RATE`）
   - 融合权重 α = 0.7（棋型评估），1-α = 0.3（五连追溯）
   - 失败方不给负奖励，仅保留棋型评估

2. **算法流程**
   - 在 `end_board` 中找到五连位置
   - 在 `history` 中追溯这 5 步落子
   - 计算反向传播奖励并叠加到棋型评估 reward

3. **Reward 融合公式**
   ```
   final_reward = α × shape_reward + (1-α) × line_bonus
   ```

## 执行过程

### 1. 设计文档
- 创建 `design/winning_line_reward_design.md`

### 2. 测试用例（TDD）
- 创建 `tests/ai/primary/test_evaluator_winning_line.py`
- 包含 15 个测试用例：
  - 五连位置查找（6 个）
  - 五连落子追溯（2 个）
  - 奖励融合（3 个）
  - 无胜者处理（2 个）
  - 集成测试（2 个）

### 3. 代码实现
在 `ai/evaluator.py` 中新增方法：
- `find_five_in_a_row(board, winner)`: 找五连位置
- `trace_winning_line(end_board, winner, history)`: 追溯五连落子步骤
- `blend_rewards(shape_reward, line_bonus, alpha)`: 融合奖励

修改方法：
- `evaluate_record()`: 获取 end_board 并传递
- `_compute_all_rewards()`: 增加 winner 和 end_board 参数，融合五连奖励

### 4. 测试验证
- 15 个新测试全部通过
- 227 个 AI 模块测试全部通过

## 执行结果

- ✅ 五连追溯功能实现完成
- ✅ 所有测试通过
- ✅ 与现有棋型评估正确融合

## 涉及文件

### 修改文件
- `ai/evaluator.py` - 新增 3 个方法，修改 2 个方法

### 新增文件
- `design/winning_line_reward_design.md` - 设计文档
- `tests/ai/primary/__init__.py` - 包初始化
- `tests/ai/primary/test_evaluator_winning_line.py` - 测试用例
