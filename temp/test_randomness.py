"""简单测试：验证概率采样的随机性"""
import sys
sys.path.insert(0, '.')

from ai.models.fc_v1.model import FCv1Model
from ai.engine import AIEngine
import random

# 创建模型
model = FCv1Model(fresh=True)
engine = AIEngine(model)

# 空棋盘
board = [[0]*15 for _ in range(15)]

# 模拟 20 次第一步选择
print("模拟 20 次第一步落子（空棋盘）：")
positions = []
for i in range(20):
    x, y = engine.get_move(board, 1)
    positions.append((x, y))
    print(f"  第 {i+1:2d} 次: ({x:2d}, {y:2d})")

unique_positions = len(set(positions))
print(f"\n不同位置数: {unique_positions}/20")
print(f"随机性: {'✅ 正常' if unique_positions >= 15 else '❌ 异常'}")
