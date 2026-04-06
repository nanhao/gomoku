"""分析 CNN 训练样本"""
import json
import sys
sys.path.insert(0, 'd:/projects/gomoku')

# 读取样本数据
with open('d:/projects/gomoku/ai/train_data/cnn_v1_cycle_1_1/1/sample.json') as f:
    data = json.load(f)

print('分析训练样本中的关键位置...\n')

for idx, sample in enumerate(data[20:40]):
    board = sample['input'][:225]
    output = sample['output']
    player = sample['input'][225]
    
    # 找到输出最大的位置
    max_pos = max(range(225), key=lambda i: output[i])
    max_val = output[max_pos]
    
    # 检查棋盘上有多少棋子
    occupied = sum(1 for x in board if x != 0)
    
    # 找出 top 3 位置
    top3 = sorted(enumerate(output), key=lambda x: x[1], reverse=True)[:3]
    
    if max_val > 0.1:  # 只看有明显区分的样本
        print(f'Sample {idx+20}: occupied={occupied}, player={player}')
        print(f'Top 3: {[(pos, round(val, 4)) for pos, val in top3]}')
        
        # 检查 top 位置的棋型
        x, y = max_pos // 15, max_pos % 15
        print(f'Max pos: ({x},{y}), board value there: {board[max_pos]}')
        print()
