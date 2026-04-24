"""Evaluator 五连追溯奖励测试 - 主流程"""
import unittest
import sys
import os
import tempfile
import shutil

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestFindFiveInARow(unittest.TestCase):
    """测试五连位置查找"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_horizontal_five(self):
        """横向五连"""
        # 构造棋盘：黑方在 (7,7)-(7,11) 形成横向五连
        board = [0] * 225
        for y in range(7, 12):
            board[7 * 15 + y] = 1  # 黑方棋子
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        self.assertEqual(len(positions), 5)
        # 验证是横向的
        for i, (x, y) in enumerate(positions):
            self.assertEqual(x, 7)
            self.assertEqual(y, 7 + i)

    def test_vertical_five(self):
        """纵向五连"""
        board = [0] * 225
        for x in range(5, 10):
            board[x * 15 + 7] = 1  # 黑方棋子
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        self.assertEqual(len(positions), 5)
        for i, (x, y) in enumerate(positions):
            self.assertEqual(x, 5 + i)
            self.assertEqual(y, 7)

    def test_diagonal_five(self):
        """主对角线五连"""
        board = [0] * 225
        for i in range(5):
            board[(3 + i) * 15 + (3 + i)] = 1
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        self.assertEqual(len(positions), 5)

    def test_anti_diagonal_five(self):
        """副对角线五连"""
        board = [0] * 225
        for i in range(5):
            board[(3 + i) * 15 + (11 - i)] = 1
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        self.assertEqual(len(positions), 5)

    def test_no_five_in_a_row(self):
        """没有五连"""
        board = [0] * 225
        # 只有 4 个棋子
        for y in range(4):
            board[7 * 15 + y] = 1
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        self.assertEqual(len(positions), 0)

    def test_six_in_a_row(self):
        """六连情况（取前 5 个）"""
        board = [0] * 225
        for y in range(6):
            board[7 * 15 + y] = 1
        
        positions = self.evaluator.find_five_in_a_row(board, 1)
        
        # 返回前 5 个
        self.assertEqual(len(positions), 5)


class TestTraceWinningLine(unittest.TestCase):
    """测试五连落子追溯"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_trace_horizontal_winning(self):
        """追溯横向五连的落子步骤"""
        # 构造 history：黑方依次在 (7,7), (7,8), (7,9), (7,10), (7,11) 落子
        # 白方在其他位置落子
        history = []
        
        # 模拟对局历史
        # step 0: 黑方 (7,7)
        history.append({
            "input": [0] * 225 + [1],  # 黑方回合
            "output": [],
            "position": [7, 7]
        })
        # step 1: 白方 (0,0)
        history.append({
            "input": [0] * 225 + [2],  # 白方回合
            "output": [],
            "position": [0, 0]
        })
        # step 2: 黑方 (7,8)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [7, 8]
        })
        # step 3: 白方 (1,1)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [1, 1]
        })
        # step 4: 黑方 (7,9)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [7, 9]
        })
        # step 5: 白方 (2,2)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [2, 2]
        })
        # step 6: 黑方 (7,10)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [7, 10]
        })
        # step 7: 白方 (3,3)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [3, 3]
        })
        # step 8: 黑方 (7,11) - 最后一步，形成五连
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [7, 11]
        })
        
        # 构造 end_board
        end_board = [0] * 225
        for y in range(7, 12):
            end_board[7 * 15 + y] = 1  # 黑方五连
        end_board[0] = 2
        end_board[16] = 2
        end_board[32] = 2
        end_board[48] = 2
        
        # 追溯五连
        bonuses = self.evaluator.trace_winning_line(end_board, 1, history)
        
        # 应该找到 5 个黑方落子步骤
        self.assertEqual(len(bonuses), 5)
        
        # 验证步骤索引
        step_indices = [idx for idx, _ in bonuses]
        self.assertEqual(sorted(step_indices), [0, 2, 4, 6, 8])
        
        # 验证奖励衰减
        # 最后一步 (step 8) 应该获得 MAX_REWARD
        for idx, reward in bonuses:
            if idx == 8:
                self.assertEqual(reward, self.evaluator.MAX_REWARD)
            else:
                # 其他步骤奖励应该小于 MAX_REWARD
                self.assertLess(reward, self.evaluator.MAX_REWARD)
                self.assertGreater(reward, 0)

    def test_trace_non_consecutive_steps(self):
        """五连的 5 步不连续"""
        # 五连的落子步骤不一定连续（中间穿插对手落子）
        history = []
        
        # 黑方 (0,0), (0,1), (0,2), (0,3), (0,4) - 但中间穿插白方
        # step 0: 黑方 (0,0)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [0, 0]
        })
        # step 1: 白方 (14,14)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [14, 14]
        })
        # step 2: 黑方 (0,1)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [0, 1]
        })
        # step 3: 白方 (13,13)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [13, 13]
        })
        # step 4: 黑方 (0,2)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [0, 2]
        })
        # step 5: 白方 (12,12)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [12, 12]
        })
        # step 6: 黑方 (0,3)
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [0, 3]
        })
        # step 7: 白方 (11,11)
        history.append({
            "input": [0] * 225 + [2],
            "output": [],
            "position": [11, 11]
        })
        # step 8: 黑方 (0,4) - 形成五连
        history.append({
            "input": [0] * 225 + [1],
            "output": [],
            "position": [0, 4]
        })
        
        # 构造 end_board
        end_board = [0] * 225
        for y in range(5):
            end_board[0 * 15 + y] = 1  # 黑方五连
        
        bonuses = self.evaluator.trace_winning_line(end_board, 1, history)
        
        # 应该找到 5 个黑方落子步骤
        self.assertEqual(len(bonuses), 5)
        
        # 验证步骤索引是 [0, 2, 4, 6, 8]
        step_indices = [idx for idx, _ in bonuses]
        self.assertEqual(sorted(step_indices), [0, 2, 4, 6, 8])


class TestBlendRewards(unittest.TestCase):
    """测试奖励融合（加法融合）"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_blend_with_line_bonus(self):
        """有五连奖励时，加法融合两种信号（Bug M2: 裁剪到 MAX_REWARD）"""
        shape_reward = 500000
        line_bonus = 1000000
        
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus)
        
        # 加法融合：shape_reward + alpha * line_bonus = 500000 + 1000000 = 1500000
        # Bug M2 修复后裁剪到 MAX_REWARD = 1000000
        self.assertEqual(blended, self.evaluator.MAX_REWARD)

    def test_blend_without_line_bonus(self):
        """无五连奖励时，结果等于棋型评估"""
        shape_reward = 500000
        line_bonus = 0
        
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus)
        
        # 加法融合：shape_reward + 0 = shape_reward
        self.assertEqual(blended, 500000)

    def test_blend_shape_higher(self):
        """棋型评估更高时，两者仍然叠加（Bug M2: 裁剪到 MAX_REWARD）"""
        shape_reward = 855000
        line_bonus = 729000
        
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus)
        
        # 加法融合：855000 + 729000 = 1584000
        # Bug M2 修复后裁剪到 MAX_REWARD = 1000000
        self.assertEqual(blended, self.evaluator.MAX_REWARD)
        # 结果仍大于单独的 shape_reward
        self.assertGreater(blended, shape_reward)

    def test_blend_equal(self):
        """两者相等时，加法融合后裁剪到 MAX_REWARD
        
        Bug M2 修复：blend_rewards 裁剪到 [MIN_REWARD, MAX_REWARD] 范围。
        当 shape_reward = line_bonus = MAX_REWARD 时，
        理论上 1000000 + 1000000 = 2000000，但裁剪后为 MAX_REWARD。
        """
        shape_reward = 1000000
        line_bonus = 1000000
        
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus)
        
        # 裁剪到 MAX_REWARD
        self.assertEqual(blended, self.evaluator.MAX_REWARD)

    def test_blend_early_step_not_dropped(self):
        """五连追溯早期步骤的信号不被丢弃
        
        这是 max 融合改为加法融合的核心修复验证：
        早期步骤的 line_bonus 较小（衰减后），
        加法融合确保它不会被完全丢弃。
        
        Bug M2 修复：融合结果裁剪到 [MIN_REWARD, MAX_REWARD]，
        超出 MAX_REWARD 的值会被截断。
        """
        shape_reward = 500000  # 活三的评估值
        line_bonus = 656100    # 五连第1步（0.9^4 * MAX_REWARD）
        
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus)
        
        # 加法融合：两种信号都贡献
        self.assertGreater(blended, shape_reward)
        self.assertGreater(blended, line_bonus)
        # 旧 max 融合会返回 max(500000, 656100) = 656100
        # 加法融合理论上返回 500000 + 656100 = 1156100
        # 但 Bug M2 修复后裁剪到 MAX_REWARD = 1000000
        self.assertEqual(blended, self.evaluator.MAX_REWARD)

    def test_blend_custom_alpha(self):
        """自定义 alpha 参数控制融合权重"""
        shape_reward = 500000
        line_bonus = 1000000
        
        # alpha = 0.5
        blended = self.evaluator.blend_rewards(shape_reward, line_bonus, alpha=0.5)
        expected = shape_reward + 0.5 * line_bonus
        self.assertEqual(blended, expected)
        
        # alpha = 0（不考虑五连追溯）
        blended_zero = self.evaluator.blend_rewards(shape_reward, line_bonus, alpha=0)
        self.assertEqual(blended_zero, shape_reward)


class TestNoWinnerNoBonus(unittest.TestCase):
    """测试无胜者时的处理"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_no_winner_empty_bonus(self):
        """无胜者时五连奖励为空"""
        history = [{"input": [0] * 226, "output": [], "position": [0, 0]}]
        end_board = [0] * 225
        
        bonuses = self.evaluator.trace_winning_line(end_board, 0, history)
        self.assertEqual(len(bonuses), 0)

    def test_draw_empty_bonus(self):
        """平局时五连奖励为空"""
        history = [{"input": [0] * 226, "output": [], "position": [0, 0]}]
        end_board = [0] * 225
        
        # winner = 3 表示平局
        bonuses = self.evaluator.trace_winning_line(end_board, 3, history)
        self.assertEqual(len(bonuses), 0)


class TestComputeAllRewardsIntegration(unittest.TestCase):
    """测试 _compute_all_rewards 的五连追溯集成"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_compute_all_rewards_with_winning_line(self):
        """测试完整的 reward 计算流程（含五连追溯）"""
        # 构造一个简单的获胜历史
        history = []
        
        # 模拟对局历史：黑方横向五连获胜
        for y in range(5):
            # 黑方落子
            board = [0] * 225
            # 设置之前黑方棋子
            for prev_y in range(y):
                board[7 * 15 + (7 + prev_y)] = 1
            # 设置之前白方棋子（在其他位置）
            for prev_y in range(y):
                board[8 * 15 + prev_y] = 2
            
            input_vector = board + [1]  # 黑方回合
            history.append({
                "input": input_vector,
                "output": [],
                "position": [7, 7 + y]
            })
            
            # 白方落子（除了最后一步）
            if y < 4:
                board2 = board.copy()
                board2[7 * 15 + (7 + y)] = 1  # 刚才黑方的落子
                input_vector2 = board2 + [2]  # 白方回合
                history.append({
                    "input": input_vector2,
                    "output": [],
                    "position": [8, 5 + y]
                })
        
        # 构造 end_board
        end_board = [0] * 225
        for y in range(5):
            end_board[7 * 15 + (7 + y)] = 1  # 黑方五连
        for y in range(5):
            end_board[8 * 15 + y] = 2  # 白方棋子
        
        # 计算 rewards
        samples = self.evaluator._compute_all_rewards(history, winner=1, end_board=end_board)
        
        # 验证返回了正确数量的样本
        self.assertEqual(len(samples), len(history))
        
        # 验证每个样本都有正确的字段
        for sample in samples:
            self.assertIn("input", sample)
            self.assertIn("output", sample)
            self.assertIn("mask", sample)
            self.assertEqual(len(sample["output"]), 225)
            self.assertEqual(len(sample["mask"]), 225)

    def test_compute_all_rewards_without_winner(self):
        """测试无胜者时的 reward 计算（不叠加五连奖励）"""
        history = [
            {
                "input": [0] * 225 + [1],
                "output": [],
                "position": [7, 7]
            }
        ]
        
        samples = self.evaluator._compute_all_rewards(history, winner=0, end_board=[])
        
        # 应该正常计算，只是没有五连奖励
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0]["output"]), 225)


if __name__ == '__main__':
    unittest.main()
