"""诊断五连追溯奖励是否正确应用"""
import sys
sys.path.insert(0, '.')
import json
import os

from ai.evaluator import Evaluator

# 找对应的 record.json
record_file = 'ai/ai_data/fc_v1_vs_fc_v1_2/1/record.json'
if os.path.exists(record_file):
    with open(record_file, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    winner = record.get('winner', 0)
    history = record.get('history', [])
    end_board = record.get('end_board', [])
    
    print(f'winner: {winner}')
    print(f'history steps: {len(history)}')
    print(f'end_board length: {len(end_board)}')
    
    if end_board and winner > 0 and winner < 3:
        board = end_board[:225]
        
        ev = Evaluator()
        
        # 找五连
        positions = ev.find_five_in_a_row(board, winner)
        print(f'五连位置: {positions}')
        
        # 追溯五连
        bonuses = ev.trace_winning_line(board, winner, history)
        print(f'五连追溯奖励数量: {len(bonuses)}')
        for idx, reward in bonuses:
            print(f'  step {idx}: {reward:.0f}')
        
        # 检查 _compute_all_rewards 是否正确
        samples = ev._compute_all_rewards(history, winner, end_board)
        
        # 检查有五连奖励的步骤
        for idx, reward in bonuses:
            sample = samples[idx]
            output = sample['output']
            position = history[idx].get('position', [])
            if position:
                px, py = position[0], position[1]
                pos_idx = px * 15 + py
                print(f'\nStep {idx} 落子 ({px}, {py}):')
                print(f'  该位置概率: {output[pos_idx]:.6f}')
                
                # 找最大概率位置
                max_p = max(output)
                max_idx = output.index(max_p)
                print(f'  最大概率: {max_p:.6f} at ({max_idx//15}, {max_idx%15})')
    else:
        print('无胜者或无 end_board')
else:
    print(f'文件不存在: {record_file}')
