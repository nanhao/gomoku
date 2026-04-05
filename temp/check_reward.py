"""检查 reward 分布"""
import json
import sys
sys.path.insert(0, 'd:/projects/gomoku')

from ai.evaluator import Evaluator

# 加载一个样本
data = json.load(open('d:/projects/gomoku/ai/train_data/fc_v1_cycle_1_1/1/sample.json'))

# 检查第一步
sample = data[0]
board = sample['input'][:225]
player = sample['input'][225]

# 统计棋盘
empty_count = sum(1 for x in board if x == 0)
black_count = sum(1 for x in board if x == 1)
white_count = sum(1 for x in board if x == 2)
print(f'棋盘: 空={empty_count}, 黑={black_count}, 白={white_count}')
print(f'当前落子方: {player}')

# 直接用 Evaluator 计算原始 reward
evaluator = Evaluator()
for i in range(15):
    x, y = 7, i  # 检查中心行的位置
    idx = x * 15 + y
    if board[idx] == 0:
        reward = evaluator._compute_position_reward(board, x, y, player)
        print(f'位置 ({x}, {y}): raw_reward = {reward}')

# 检查最后一步（可能有明显棋型）
print('\n=== 最后一步 ===')
last = data[-1]
board_last = last['input'][:225]
player_last = last['input'][225]
empty_last = sum(1 for x in board_last if x == 0)
print(f'棋盘: 空位={empty_last}')
print(f'output 分布: 非0数量={sum(1 for x in last["output"] if x > 0)}')
print(f'output 最大值: {max(last["output"]):.4f}')

# 找到 output 最大的位置
max_idx = last['output'].index(max(last['output']))
max_x, max_y = max_idx // 15, max_idx % 15
print(f'最大值位置: ({max_x}, {max_y})')

# 检查一个中局位置（第30步）
print('\n=== 第30步（中局）===')
mid = data[29]  # 第30步
board_mid = mid['input'][:225]
player_mid = mid['input'][225]
empty_mid = sum(1 for x in board_mid if x == 0)
black_mid = sum(1 for x in board_mid if x == 1)
white_mid = sum(1 for x in board_mid if x == 2)
print(f'棋盘: 空={empty_mid}, 黑={black_mid}, 白={white_mid}')
print(f'当前落子方: {player_mid}')

# 统计所有空位的 raw_reward
rewards = []
for idx in range(225):
    if board_mid[idx] == 0:
        x, y = idx // 15, idx % 15
        reward = evaluator._compute_position_reward(board_mid, x, y, player_mid)
        rewards.append((reward, x, y))

# 排序找最高 reward
rewards.sort(reverse=True)
print(f'\n前 10 个最高 reward 位置:')
for r, x, y in rewards[:10]:
    print(f'  ({x}, {y}): {r}')

print(f'\nreward 分布统计:')
print(f'  非零 reward 数量: {sum(1 for r, _, _ in rewards if r > 0)}')
print(f'  零 reward 数量: {sum(1 for r, _, _ in rewards if r == 0)}')

# 检查 sample 的 output 分布
print(f'\n实际 sample output 分布:')
mid_output = mid['output']
mid_nonzero = [(mid_output[i], i // 15, i % 15) for i in range(225) if mid_output[i] > 0]
mid_nonzero.sort(reverse=True)
print(f'前 10 个最高 output:')
for prob, x, y in mid_nonzero[:10]:
    print(f'  ({x}, {y}): {prob:.6f}')
print(f'\noutput 非0数量: {len(mid_nonzero)}')
print(f'output 总和: {sum(mid_output):.6f}')
