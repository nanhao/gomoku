"""通用概率工具模块

从 evaluator.py 提取，提供带 mask 的数值稳定 Softmax 转换。
"""

import math
from typing import List


# Softmax 缩放因子：将 reward 从 [-1000000, 1000000] 缩放到 [-10, 10]
# 避免 exp(x) 在大数值时的下溢问题
DEFAULT_SOFTMAX_SCALE = 1.0 / 100000.0


def masked_softmax(rewards: List[float], mask: List[int],
                   scale: float = DEFAULT_SOFTMAX_SCALE) -> List[float]:
    """对 reward 值应用 Softmax 转换为概率分布

    只考虑 mask=1 的位置（空位），已落子位置保持为 0。

    注意：reward 值可能高达 1000000，直接计算 exp 会导致数值下溢。
    因此先将 reward 缩放到合理范围，再计算 softmax。

    Args:
        rewards: 225 维 reward 列表
        mask: 225 维 mask 列表
        scale: reward 缩放因子，默认 1.0 / 100000.0

    Returns:
        225 维概率分布，值域 (0, 1)，所有 mask=1 位置的概率和为 1
    """
    # 找到所有有效位置（mask=1）
    valid_indices = [i for i in range(225) if mask[i] == 1]

    if not valid_indices:
        # 没有有效位置，返回全 0
        return [0.0] * 225

    # 提取有效位置的 reward 值
    valid_rewards = [rewards[i] for i in valid_indices]

    # 缩放 reward 到合理范围，避免数值下溢
    scaled_rewards = [r * scale for r in valid_rewards]

    # 数值稳定性处理：减去最大值
    max_reward = max(scaled_rewards)
    exp_values = [math.exp(r - max_reward) for r in scaled_rewards]

    # 计算 softmax
    sum_exp = sum(exp_values)
    softmax_values = [exp_val / sum_exp for exp_val in exp_values]

    # 构建完整的 225 维输出
    output = [0.0] * 225
    for idx, prob in zip(valid_indices, softmax_values):
        output[idx] = prob

    return output
