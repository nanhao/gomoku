"""随机挑选10个冲四样本，打印棋盘和reward值"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ai.reward_calculator import RewardCalculator, RewardConfig
from ai.shape_detector import detect_shapes

data_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'ai', 'merge_samples', 'chong4.json'))

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

random_gen = random.Random(123)
indices = random_gen.sample(range(len(data)), 10)

calc = RewardCalculator(RewardConfig())

SYMBOLS = {0: '.', 1: 'X', 2: 'O'}

for seq, idx in enumerate(indices, 1):
    s = data[idx]
    board = [int(x) for x in s['input'][:225]]
    player = int(s['input'][225])
    output = s['output']
    mask = s['mask']

    # 计算每个空位的reward
    rewards = [0.0] * 225
    for i in range(225):
        if board[i] == 0:
            x, y = i // 15, i % 15
            rewards[i] = calc.compute_position_reward(board, x, y, player)

    # 找连五位置
    lian5_pos = []
    for i in range(225):
        if board[i] == 0:
            x, y = i // 15, i % 15
            bc = list(board)
            bc[i] = player
            shapes = detect_shapes(bc, (x, y), player)
            for sh in shapes:
                if sh['shape'] == '连五':
                    lian5_pos.append((x, y))
                    break

    print(f"===== 样本 {seq} (index={idx}) =====")
    print(f"当前落子方: {'黑(X)' if player == 1 else '白(O)'}")
    print(f"连五位置: {lian5_pos}")
    print()

    # 打印棋盘
    header = '    ' + ' '.join(f'{j:2d}' for j in range(15))
    print(header)
    for x in range(15):
        row = f'{x:2d}  '
        for y in range(15):
            idx2 = x * 15 + y
            sym = SYMBOLS[board[idx2]]
            # 标记连五位置
            if (x, y) in lian5_pos:
                row += ' * '
            else:
                row += f' {sym} '
        print(row)

    print()

    # 打印有reward的空位（reward > 0）
    print("非零reward空位:")
    nonzero = [(i, rewards[i], output[i]) for i in range(225)
                if board[i] == 0 and rewards[i] > 0]
    nonzero.sort(key=lambda t: t[1], reverse=True)

    for i, r, p in nonzero[:15]:  # 只打印top15
        x, y = i // 15, i % 15
        marker = ' <-- 连五' if (x, y) in lian5_pos else ''
        print(f"  ({x:2d},{y:2d}) reward={r:>10.0f}  prob={p:.6f}{marker}")

    if len(nonzero) > 15:
        print(f"  ... 还有 {len(nonzero) - 15} 个非零reward位置")

    print()
