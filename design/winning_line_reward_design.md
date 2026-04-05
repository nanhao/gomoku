# 五连追溯 Reward 反向传播算法设计

## 需求

根据 `requirements/key_points.md` 第十二次追加：
- 获胜方五连的那 5 步落子获得额外奖励
- 奖励按衰减系数 r 反向传播
- 与现有棋型评估 reward 叠加

## 设计决策

1. **衰减系数 r**：复用 `Evaluator.DECAY_RATE = 0.9`
2. **融合策略**：取两者最大值 `max(shape_reward, line_bonus)`
3. **失败方**：不给负奖励，仅保留棋型评估

## 算法流程

```
evaluate_record(record_file):
    1. 读取 record.json
    2. 获取 winner, history, end_board
    3. 计算棋型评估 reward（现有逻辑）
    4. 如果有胜者（winner != 0 and winner != 3）:
        a. 找出五连位置（5个连续同色棋子）
        b. 在 history 中追溯这 5 步落子
        c. 计算五连奖励并反向传播
        d. 叠加到棋型评估 reward
    5. 应用 Softmax
    6. 保存 sample.json
```

## 核心算法：五连追溯

### 伪代码

```python
def trace_winning_line(end_board, winner, history):
    """
    找出形成五连的那 5 步落子
    
    Returns:
        List[Tuple[int, float]]: [(step_index, bonus_reward), ...]
    """
    # 1. 在 end_board 中找到五连位置
    winning_positions = find_five_in_a_row(end_board, winner)
    # 返回 5 个位置坐标，如 [(7,7), (7,8), (7,9), (7,10), (7,11)]
    
    # 2. 在 history 中找出这 5 个位置对应的落子步骤
    step_indices = []
    for pos in winning_positions:
        for i, step in enumerate(history):
            if step["position"] == pos:
                # 检查落子方是否正确
                player = step["input"][225]
                if player == winner:
                    step_indices.append(i)
                break
    
    # 3. 按 step 排序（从小到大）
    step_indices.sort()
    
    # 4. 反向传播奖励
    # j5 是最后一步（索引最大的），奖励 = MAX_REWARD
    # j4 奖励 = r * j5 的奖励
    # ...
    results = []
    reward = MAX_REWARD
    for i in range(len(step_indices) - 1, -1, -1):
        results.append((step_indices[i], reward))
        reward *= DECAY_RATE
    
    return results
```

### find_five_in_a_row 算法

```python
def find_five_in_a_row(board, winner):
    """
    在棋盘上找到五连的位置
    
    Args:
        board: 225 维棋盘向量
        winner: 获胜方 (1 or 2)
    
    Returns:
        List[Tuple[int, int]]: 五连的 5 个位置坐标
    """
    # 四个方向
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for x in range(15):
        for y in range(15):
            if board[x * 15 + y] != winner:
                continue
            
            for dx, dy in directions:
                positions = [(x, y)]
                nx, ny = x + dx, y + dy
                
                while 0 <= nx < 15 and 0 <= ny < 15:
                    if board[nx * 15 + ny] == winner:
                        positions.append((nx, ny))
                        nx += dx
                        ny += dy
                    else:
                        break
                
                if len(positions) >= 5:
                    # 返回前 5 个位置
                    return positions[:5]
    
    return []
```

## Reward 融合策略

```python
def blend_rewards(shape_reward, line_bonus):
    """
    融合棋型评估奖励和五连追溯奖励
    
    取两者最大值，确保五连追溯信号不被稀释。
    """
    return max(shape_reward, line_bonus)
```

**设计决策**：取最大值而非加权平均

| 场景 | 棋型评估 | 五连追溯 | 融合结果 |
|------|---------|---------|---------|
| 五连追溯更高 | 380,000 | 656,100 | 656,100 |
| 棋型评估更高 | 855,000 | 729,000 | 855,000 |
| 两者相等 | 1,000,000 | 1,000,000 | 1,000,000 |

## 文件修改

### ai/evaluator.py

新增方法：
- `find_five_in_a_row(board, winner)`: 找五连位置
- `trace_winning_line(end_board, winner, history)`: 追溯五连落子步骤
- `compute_line_bonus(end_board, winner, history)`: 计算五连奖励

修改方法：
- `evaluate_record()`: 增加 winner 参数传递，调用五连追溯
- `_compute_all_rewards()`: 增加 end_board, winner 参数，融合奖励

## 测试用例

### 主流程测试

1. **test_trace_winning_line_horizontal**: 横向五连追溯
2. **test_trace_winning_line_vertical**: 纵向五连追溯
3. **test_trace_winning_line_diagonal**: 对角线五连追溯
4. **test_blend_rewards**: 奖励融合正确
5. **test_no_winner_no_bonus**: 无胜者时不加五连奖励
6. **test_draw_no_bonus**: 平局时不加五连奖励

### 边缘测试

1. **test_multiple_lines**: 多条五连（取最先找到的）
2. **test_six_in_a_row**: 六连情况（取前 5 个）
