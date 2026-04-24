"""LineTracer 五连追溯奖励测试

测试五连位置查找、追溯步骤索引、衰减奖励计算、融合奖励。
"""

import sys
import os
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.line_tracer import LineTracer


class TestFindFiveInARow(unittest.TestCase):
    """测试 find_five_in_a_row 五连位置查找"""

    def setUp(self):
        self.tracer = LineTracer()

    def test_horizontal_five(self):
        """横向五连"""
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1  # 黑方横向连五

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 5)
        for i, (x, y) in enumerate(positions):
            self.assertEqual(x, 7)
            self.assertEqual(y, 7 + i)

    def test_vertical_five(self):
        """纵向五连"""
        board = [0] * 225
        for x in range(5, 10):
            board[x * 15 + 7] = 1

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 5)
        for i, (x, y) in enumerate(positions):
            self.assertEqual(x, 5 + i)
            self.assertEqual(y, 7)

    def test_diagonal_five(self):
        """主对角线五连"""
        board = [0] * 225
        for i in range(5):
            board[(5 + i) * 15 + (5 + i)] = 1

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 5)

    def test_anti_diagonal_five(self):
        """副对角线五连"""
        board = [0] * 225
        for i in range(5):
            board[(5 + i) * 15 + (9 - i)] = 1

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 5)

    def test_no_five_returns_empty(self):
        """无五连时返回空列表"""
        board = [0] * 225
        board[7 * 15 + 7] = 1
        board[7 * 15 + 8] = 1
        board[7 * 15 + 9] = 1
        # 只有3个，不够五连

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 0)

    def test_wrong_winner_returns_empty(self):
        """查找错误方的五连应返回空"""
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1  # 黑方五连

        positions = self.tracer.find_five_in_a_row(board, 2)  # 查白方

        self.assertEqual(len(positions), 0)

    def test_six_in_a_row_returns_first_five(self):
        """六连时返回前五个位置"""
        board = [0] * 225
        for y in range(7, 13):
            board[7 * 15 + y] = 1  # 六连

        positions = self.tracer.find_five_in_a_row(board, 1)

        self.assertEqual(len(positions), 5)


class TestTraceWinningLine(unittest.TestCase):
    """测试 trace_winning_line 追溯奖励"""

    def setUp(self):
        self.tracer = LineTracer(max_reward=1000000, decay_rate=0.9)

    def test_no_winner_returns_empty(self):
        """无胜者时返回空列表"""
        board = [0] * 225
        history = []
        result = self.tracer.trace_winning_line(board, 0, history)
        self.assertEqual(result, [])

    def test_draw_returns_empty(self):
        """平局时返回空列表"""
        board = [0] * 225
        history = []
        result = self.tracer.trace_winning_line(board, 3, history)
        self.assertEqual(result, [])

    def test_trace_horizontal_five(self):
        """追溯横向五连的步骤索引和奖励"""
        # 构造黑方横向五连的终局棋盘
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1  # 黑方五连
        board[0 * 15 + 0] = 2  # 白方1子

        # 构造 history
        history = [
            {"position": [7, 7], "input": [0] * 225 + [1]},   # 黑 step 0
            {"position": [0, 0], "input": [0] * 225 + [2]},   # 白 step 1
            {"position": [7, 8], "input": [0] * 225 + [1]},   # 黑 step 2
            {"position": [1, 1], "input": [0] * 225 + [2]},   # 白 step 3
            {"position": [7, 9], "input": [0] * 225 + [1]},   # 黑 step 4
            {"position": [2, 2], "input": [0] * 225 + [2]},   # 白 step 5
            {"position": [7, 10], "input": [0] * 225 + [1]},  # 黑 step 6
            {"position": [3, 3], "input": [0] * 225 + [2]},   # 白 step 7
            {"position": [7, 11], "input": [0] * 225 + [1]},  # 黑 step 8
        ]

        result = self.tracer.trace_winning_line(board, 1, history)

        # 应找到 5 个步骤
        self.assertEqual(len(result), 5)

        # 验证步骤索引（从大到小排列，最后落子的步骤索引最大，排在最前）
        step_indices = [r[0] for r in result]
        self.assertEqual(step_indices, [8, 6, 4, 2, 0])

        # 验证最后落子的步骤奖励为 MAX_REWARD（result[0] 是 step 8）
        first_reward = result[0][1]
        self.assertEqual(first_reward, 1000000)

        # 验证衰减：step 6 的奖励 = MAX_REWARD * decay_rate
        second_reward = result[1][1]
        self.assertAlmostEqual(second_reward, 1000000 * 0.9)

    def test_decay_formula(self):
        """验证衰减公式：第 i 步奖励 = MAX_REWARD * decay_rate^i（从最后落子往前衰减）"""
        tracer = LineTracer(max_reward=1000000, decay_rate=0.9)

        # 构造五连
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1

        history = [
            {"position": [7, 7], "input": [0] * 225 + [1]},
            {"position": [0, 0], "input": [0] * 225 + [2]},
            {"position": [7, 8], "input": [0] * 225 + [1]},
            {"position": [1, 1], "input": [0] * 225 + [2]},
            {"position": [7, 9], "input": [0] * 225 + [1]},
            {"position": [2, 2], "input": [0] * 225 + [2]},
            {"position": [7, 10], "input": [0] * 225 + [1]},
            {"position": [3, 3], "input": [0] * 225 + [2]},
            {"position": [7, 11], "input": [0] * 225 + [1]},
        ]

        result = tracer.trace_winning_line(board, 1, history)

        # result[0] 是最后落子的步骤（MAX_REWARD），result[1] 是倒数第二步...
        # result[i] 的奖励 = MAX_REWARD * decay_rate^i
        for i, (step_idx, reward) in enumerate(result):
            expected = 1000000 * (0.9 ** i)
            self.assertAlmostEqual(reward, expected)

    def test_insufficient_history_returns_empty(self):
        """history 中不足 5 步五连位置时返回空"""
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1

        # history 中缺少五连步骤
        history = [
            {"position": [7, 7], "input": [0] * 225 + [1]},
            {"position": [0, 0], "input": [0] * 225 + [2]},
        ]

        result = self.tracer.trace_winning_line(board, 1, history)
        self.assertEqual(result, [])

    def test_custom_decay_rate(self):
        """自定义衰减率"""
        tracer = LineTracer(max_reward=1000000, decay_rate=0.5)

        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1

        history = [
            {"position": [7, 7], "input": [0] * 225 + [1]},
            {"position": [0, 0], "input": [0] * 225 + [2]},
            {"position": [7, 8], "input": [0] * 225 + [1]},
            {"position": [1, 1], "input": [0] * 225 + [2]},
            {"position": [7, 9], "input": [0] * 225 + [1]},
            {"position": [2, 2], "input": [0] * 225 + [2]},
            {"position": [7, 10], "input": [0] * 225 + [1]},
            {"position": [3, 3], "input": [0] * 225 + [2]},
            {"position": [7, 11], "input": [0] * 225 + [1]},
        ]

        result = tracer.trace_winning_line(board, 1, history)

        # 验证衰减率 0.5：result[0] 是最后落子步骤
        self.assertEqual(result[0][1], 1000000)
        self.assertAlmostEqual(result[1][1], 500000)
        self.assertAlmostEqual(result[2][1], 250000)


class TestBlendRewards(unittest.TestCase):
    """测试 blend_rewards 融合奖励"""

    def setUp(self):
        self.tracer = LineTracer(max_reward=1000000, min_reward=-1000000, blend_alpha=1.0)

    def test_blend_additive(self):
        """加法融合：shape_reward + alpha * line_bonus"""
        result = self.tracer.blend_rewards(500000, 300000)
        self.assertEqual(result, 800000)  # 500000 + 1.0 * 300000

    def test_blend_with_custom_alpha(self):
        """自定义 alpha 融合"""
        result = self.tracer.blend_rewards(500000, 300000, alpha=0.5)
        self.assertEqual(result, 650000)  # 500000 + 0.5 * 300000

    def test_blend_clipped_to_max(self):
        """融合结果裁剪到 max_reward"""
        result = self.tracer.blend_rewards(800000, 500000)
        self.assertEqual(result, 1000000)  # 裁剪到 max_reward

    def test_blend_clipped_to_min(self):
        """融合结果裁剪到 min_reward"""
        result = self.tracer.blend_rewards(-800000, -500000)
        self.assertEqual(result, -1000000)  # 裁剪到 min_reward

    def test_blend_negative_line_bonus(self):
        """负的五连追溯奖励"""
        result = self.tracer.blend_rewards(500000, -200000)
        self.assertEqual(result, 300000)  # 500000 + 1.0 * (-200000)

    def test_blend_zero_values(self):
        """零值融合"""
        result = self.tracer.blend_rewards(0, 0)
        self.assertEqual(result, 0)

    def test_blend_uses_instance_alpha(self):
        """不传 alpha 时使用实例的 blend_alpha"""
        tracer = LineTracer(blend_alpha=0.5)
        result = tracer.blend_rewards(400000, 200000)
        self.assertEqual(result, 500000)  # 400000 + 0.5 * 200000


if __name__ == '__main__':
    unittest.main()
