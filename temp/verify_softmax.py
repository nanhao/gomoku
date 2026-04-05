"""验证 Softmax 问题"""
import math

# 模拟第30步的 reward 分布
raw_rewards = [
    950000, 950000,  # 两个最高
    900000, 900000,  # 次高
    500000, 500000, 500000, 500000, 500000,  # 中等
    475000,
    0, 0, 0  # 一些零值
]

print("=== 当前 Softmax 方法 ===")
max_r = max(raw_rewards)
exp_values = [math.exp(r - max_r) for r in raw_rewards]
sum_exp = sum(exp_values)
softmax_result = [e / sum_exp for e in exp_values]
print(f"exp(0) = {math.exp(0)}")
print(f"exp(-500000) ≈ 0 (下溢)")
print(f"Softmax 结果前 5: {softmax_result[:5]}")
print(f"非零数量: {sum(1 for x in softmax_result if x > 0.0001)}")

print("\n=== 修复方案：缩放 reward ===")
# 将 reward 缩放到合理范围
scale = 1.0 / 1000000  # 缩放到 [-1, 1]
scaled_rewards = [r * scale for r in raw_rewards]
print(f"缩放后的 reward 前 5: {scaled_rewards[:5]}")

max_r = max(scaled_rewards)
exp_values = [math.exp(r - max_r) for r in scaled_rewards]
sum_exp = sum(exp_values)
softmax_result = [e / sum_exp for e in exp_values]
print(f"Softmax 结果前 5: {softmax_result[:5]}")
print(f"非零数量: {sum(1 for x in softmax_result if x > 0.0001)}")
print(f"总和: {sum(softmax_result):.6f}")
