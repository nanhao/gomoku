"""验证 Softmax 修复"""
import sys
sys.path.insert(0, 'd:/projects/gomoku')
import importlib

# 强制重新加载模块
import ai.evaluator
importlib.reload(ai.evaluator)
from ai.evaluator import Evaluator

# 创建 Evaluator
evaluator = Evaluator()

# 模拟第30步的 reward 分布（简化版，只测试前几个位置）
raw_rewards = [0.0] * 225
raw_rewards[0] = 950000   # 位置 0 最高
raw_rewards[1] = 950000   # 位置 1 最高
raw_rewards[2] = 900000   # 位置 2 次高
raw_rewards[3] = 900000   # 位置 3 次高
raw_rewards[4] = 500000   # 位置 4 中等
raw_rewards[5] = 0        # 位置 5 零
mask = [1] * 225  # 全部为有效位置

# 调用修复后的 _softmax
output = evaluator._softmax(raw_rewards, mask)

print("=== 修复后的 Softmax 结果 ===")
print(f"输出前 5: {output[:5]}")
print(f"非零数量: {sum(1 for x in output if x > 0.0001)}")
print(f"总和: {sum(output):.6f}")

# 验证最高值位置
sorted_output = sorted(enumerate(output), key=lambda x: x[1], reverse=True)
print(f"\n前 5 个最高概率位置:")
for idx, prob in sorted_output[:5]:
    print(f"  位置 {idx}: {prob:.6f}")

# 对比：如果 reward 全为 0（第一步）
print("\n=== 第一步（全零 reward）===")
zero_rewards = [0.0] * 225
zero_mask = [1] * 225
zero_output = evaluator._softmax(zero_rewards, zero_mask)
print(f"输出前 5: {zero_output[:5]}")
print(f"非零数量: {sum(1 for x in zero_output if x > 0.0001)}")
print(f"每个位置概率: {zero_output[0]:.6f} (应该约为 1/225 = {1/225:.6f})")
