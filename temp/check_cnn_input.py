"""检查 CNN 输入转换"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import torch
from ai.models.cnn_v1.model import CNNModel

# 创建模型
model = CNNModel(fresh=True)

# 测试输入转换
board_vector = [0] * 225 + [1]  # 空棋盘，黑方
tensor = model._vector_to_tensor(board_vector)

print('Input tensor shape:', tensor.shape)
print('Expected shape: (1, 3, 15, 15)')
print()

# 测试有棋子的棋盘
board = [0] * 225
board[7*15+7] = 1  # 黑棋
board[7*15+8] = 2  # 白棋
board_vector = board + [1]  # 黑方

tensor = model._vector_to_tensor(board_vector)
print('Board with pieces:')
print('Black channel at (7,7):', tensor[0, 0, 7, 7].item())  # 应该是 1
print('White channel at (7,8):', tensor[0, 1, 7, 8].item())  # 应该是 1
print('Player channel (all 1 for black):', tensor[0, 2, 0, 0].item())  # 应该是 1

# 测试白方
board_vector = board + [2]  # 白方
tensor = model._vector_to_tensor(board_vector)
print('Player channel (all 0 for white):', tensor[0, 2, 0, 0].item())  # 应该是 0

# 测试模型前向传播
print('\n测试前向传播...')
output = model.predict(board_vector)
print('Output shape:', len(output))
print('Output sum:', sum(output))
print('Output range:', min(output), '-', max(output))
