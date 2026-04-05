"""调试：检查模型预测和落子选择"""
import sys
sys.path.insert(0, '.')

import torch
from ai.models.fc_v1.model import FCv1Model
from ai.engine import AIEngine

# 创建模型
model = FCv1Model(fresh=True)
engine = AIEngine(model)

# 空棋盘
board = [[0]*15 for _ in range(15)]

# 获取预测
rewards = model.predict([0]*225 + [1])

# 分析 reward 分布
print("模型预测分析:")
print(f"  最大值: {max(rewards):.8f}")
print(f"  最小值: {min(rewards):.8f}")
print(f"  差值: {max(rewards) - min(rewards):.8f}")
print(f"  平均值: {sum(rewards)/len(rewards):.8f}")
print(f"  期望均匀分布: {1/225:.8f}")

# 找出最大值位置
max_val = max(rewards)
max_positions = [(i//15, i%15) for i, r in enumerate(rewards) if abs(r - max_val) < 1e-7]
print(f"\n最大概率位置数量: {len(max_positions)}")
print(f"最大概率位置（前5个）: {max_positions[:5]}")

# 模拟多次 get_move，看是否总是选择相同位置
print("\n模拟 10 次 get_move（空棋盘）:")
positions = []
for i in range(10):
    x, y = engine.get_move(board, 1)
    positions.append((x, y))
    print(f"  第 {i+1} 次: ({x}, {y})")

print(f"\n位置多样性: {len(set(positions))} 种不同位置")
