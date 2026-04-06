"""分析 Evaluator 棋型检测"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import json
from ai.evaluator import Evaluator

# 加载样本
with open('d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1/sample.json') as f:
    sample_data = json.load(f)

# 加载对局记录
with open('d:/projects/gomoku/ai/ai_data/cnn_v1_cycle_1_1/1/record.json') as f:
    record = json.load(f)

evaluator = Evaluator()

# Step 22 分析
sample22 = sample_data[22]
board = sample22['input'][:225]
player = sample22['input'][225]

print('=== Step 22 棋型检测分析 ===')
print(f'当前落子方: {"黑" if player == 1 else "白"}')
print()

# 检测 (7,6) 位置的棋型
x, y = 7, 6
print(f'检测位置 ({x}, {y}):')

# 检测黑方在此位置落子后的棋型
shapes_black = evaluator.detect_shapes(board, (x, y), 1)
print(f'  黑方落子后棋型: {shapes_black}')
for shape in shapes_black:
    reward = evaluator.shape_to_reward(board, shape, 1, True)
    print(f'    -> reward: {reward}')

# 检测白方在此位置落子后的棋型
shapes_white = evaluator.detect_shapes(board, (x, y), 2)
print(f'  白方落子后棋型: {shapes_white}')
for shape in shapes_white:
    reward = evaluator.shape_to_reward(board, shape, 2, False)
    print(f'    -> reward: {reward}')

# 计算最终 reward
final_reward = evaluator._compute_position_reward(board, x, y, player)
print(f'  最终 reward: {final_reward}')

# 检查五连位置 (6,5)-(10,5) 的棋型
print('\n五连位置棋型检测:')
for pos in [(6,5), (7,5), (8,5), (9,5), (10,5)]:
    px, py = pos
    idx = px * 15 + py
    print(f'  ({px},{py}): board={board[idx]}, sample_output={sample22["output"][idx]:.4f}')
