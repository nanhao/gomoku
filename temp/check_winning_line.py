"""检查五连追溯奖励"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')

import json

# 加载对局记录
with open('d:/projects/gomoku/ai/ai_data/cnn_v1_cycle_1_1/1/record.json') as f:
    record = json.load(f)

print('Winner:', record['winner'])  # 2 = 白方
print('Total steps:', len(record['history']))

# 检查 end_board
end_board = record.get('end_board', [])
print('End board length:', len(end_board))

# 找五连位置
from ai.evaluator import Evaluator
evaluator = Evaluator()

if end_board:
    winning_positions = evaluator.find_five_in_a_row(end_board[:225], record['winner'])
    print(f'五连位置: {winning_positions}')
    
    # 追溯五连的落子步骤
    bonuses = evaluator.trace_winning_line(end_board[:225], record['winner'], record['history'])
    print(f'五连追溯奖励: {bonuses}')

# 检查 step 22 实际落子位置
step22 = record['history'][22]
print(f'\nStep 22:')
print(f'  Position: {step22.get("position")}')
print(f'  Input player: {step22["input"][225]}')

# Step 22 的 sample output 分析
with open('d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1/sample.json') as f:
    sample_data = json.load(f)

sample22 = sample_data[22]
max_pos = sample22['output'].index(max(sample22['output']))
print(f'  Sample output max: ({max_pos//15}, {max_pos%15}), value: {max(sample22["output"]):.4f}')
