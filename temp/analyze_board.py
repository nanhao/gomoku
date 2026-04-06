"""分析棋盘和样本"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import json

# 加载样本
with open('d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1/sample.json') as f:
    data = json.load(f)

# Sample 22 的详细分析
sample = data[22]
board = sample['input'][:225]
target = sample['output']
player = sample['input'][225]

print('=== Sample 22 详细分析 ===')
print(f'当前落子方: {"黑" if player == 1 else "白"}')
print()

# 打印整个第 7 行
print('第 7 行 (y=0-14):')
row = ''
for i in range(15):
    pos = 7*15 + i
    cell = board[pos]
    if cell == 0:
        row += '. '
    elif cell == 1:
        row += 'X '  # 黑
    else:
        row += 'O '  # 白
print(row)

print('\n第 7 行详细:')
for i in range(15):
    pos = 7*15 + i
    print(f'  (7,{i}): board={board[pos]}, target={target[pos]:.4f}')

# 找出所有有黑子的位置
black_positions = [(i//15, i%15) for i, v in enumerate(board) if v == 1]
white_positions = [(i//15, i%15) for i, v in enumerate(board) if v == 2]
print(f'\n黑子位置: {black_positions}')
print(f'白子位置: {white_positions}')

# 找出 target 最大的位置
max_pos = target.index(max(target))
print(f'\nTarget 最大位置: ({max_pos//15}, {max_pos%15}), 值: {max(target):.4f}')

# 检查这个位置周围的棋型
print(f'\n检查位置 ({max_pos//15}, {max_pos%15}) 周围:')
x, y = max_pos // 15, max_pos % 15
# 检查水平方向
h_row = ''
for i in range(15):
    pos = x*15 + i
    h_row += f'{board[pos]} '
print(f'  第 {x} 行: {h_row}')
