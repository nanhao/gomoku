"""深入分析五连奖励与棋型评估的关系"""
import sys
sys.path.insert(0, '.')
import json

from ai.evaluator import Evaluator

record_file = 'ai/ai_data/fc_v1_vs_fc_v1_2/1/record.json'
with open(record_file, 'r', encoding='utf-8') as f:
    record = json.load(f)

winner = record.get('winner', 0)
history = record.get('history', [])
end_board = record.get('end_board', [])

ev = Evaluator()

# 获取五连追溯奖励
bonuses = ev.trace_winning_line(end_board[:225], winner, history)
bonus_dict = dict(bonuses)

print("=== 五连追溯奖励分析 ===\n")

for step_idx, step in enumerate(history):
    if step_idx not in bonus_dict:
        continue
    
    input_vector = step.get('input', [])
    position = step.get('position', [])
    
    if len(input_vector) < 226 or len(position) < 2:
        continue
    
    board = input_vector[:225]
    current_player = input_vector[225]
    px, py = position[0], position[1]
    pos_idx = px * 15 + py
    
    # 计算棋型评估 reward
    shape_reward = ev._compute_position_reward(board, px, py, current_player)
    
    # 五连奖励
    line_bonus = bonus_dict[step_idx]
    
    # 融合后的奖励
    blended = ev.blend_rewards(shape_reward, line_bonus)
    
    print(f"Step {step_idx} 落子 ({px}, {py}):")
    print(f"  棋型评估 reward: {shape_reward:,}")
    print(f"  五连追溯 reward: {line_bonus:,}")
    print(f"  融合后 reward:   {blended:,.0f}")
    print(f"  融合公式: 0.7 * {shape_reward} + 0.3 * {line_bonus}")
    print()

# 找一个非五连但棋型很好的位置对比
print("=== 对比：非五连位置的棋型评估 ===\n")

# 检查 step 58 的棋盘，找棋型最好的位置
step58 = history[58]
board58 = step58['input'][:225]
player58 = step58['input'][225]

# 遍历所有空位，找棋型 reward 最高的
best_rewards = []
for i in range(225):
    if board58[i] == 0:
        x, y = i // 15, i % 15
        r = ev._compute_position_reward(board58, x, y, player58)
        if r > 0:
            best_rewards.append((r, x, y))

best_rewards.sort(reverse=True)
print("Step 58 棋盘上 reward 最高的 5 个位置:")
for r, x, y in best_rewards[:5]:
    print(f"  ({x}, {y}): {r:,}")
