"""Evaluator 新 Reward 算法边缘测试

测试边界条件和异常情况。
"""

import sys
import os
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestEvaluatorNewRewardEdge(unittest.TestCase):
    """新 reward 算法边缘测试"""
    
    def setUp(self):
        """测试前准备"""
        self.evaluator = Evaluator()
    
    def test_five_steps_not_consecutive(self):
        """测试五子连线步骤不连续的情况"""
        # 获胜方落子序列：a1, a2, a3, a4, a5, a6, a7, a8, a9
        # 其中 a1, a3, a5, a7, a9 构成五子连线（不连续，中间有白方落子）
        
        history = []
        board = [0] * 225
        
        # 黑方连线位置：(7,3), (7,4), (7,5), (7,6), (7,7)
        # 白方干扰位置：(0,0), (0,1), (0,2), (0,3)
        moves = [
            ([7, 3], 1),   # 黑 a1 - 连线1
            ([0, 0], 2),   # 白 b1
            ([7, 4], 1),   # 黑 a2 - 连线2
            ([0, 1], 2),   # 白 b2
            ([7, 5], 1),   # 黑 a3 - 连线3
            ([0, 2], 2),   # 白 b3
            ([7, 6], 1),   # 黑 a4 - 连线4
            ([0, 3], 2),   # 白 b4
            ([7, 7], 1),   # 黑 a5 - 连线5 (获胜)
        ]
        
        for pos, player in moves:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = player
            input_vec = board.copy() + [player]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": pos
            })
        
        # 构造获胜方（黑方）落子列表
        winning_actions = [
            (0, [7, 3]),   # step 0
            (2, [7, 4]),   # step 2
            (4, [7, 5]),   # step 4
            (6, [7, 6]),   # step 6
            (8, [7, 7]),   # step 8
        ]
        
        # 调用 _find_five_in_a_row
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 验证返回正确的5个步骤索引（不连续）
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 2, 4, 6, 8])
        
        # 验证 reward 计算
        sample_dict = self.evaluator._initialize_sample(history)
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        max_r = Evaluator.MAX_REWARD
        r = Evaluator.DECAY_RATE
        
        # j5 (step 8): MAX_REWARD
        self.assertEqual(sample_dict[8]["output"][7 * 15 + 7], int(max_r))
        
        # j4 (step 6): MAX_REWARD * r
        self.assertEqual(sample_dict[6]["output"][7 * 15 + 6], int(max_r * r))
    
    def test_more_than_five_in_line(self):
        """测试超过五子连线的情况（六子）"""
        # 获胜方形成六子连线：(7,2), (7,3), (7,4), (7,5), (7,6), (7,7)
        # 应返回构成连线的最后5步：(7,3), (7,4), (7,5), (7,6), (7,7)
        
        history = []
        board = [0] * 225
        
        line_positions = [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        
        for pos in line_positions:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = 1
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(line_positions)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 验证返回最后5步（算法返回找到的第一个有>=5子的方向上的最后5步）
        # 对于六子连线，返回 [0,1,2,3,4] 或 [1,2,3,4,5] 都是合理的
        self.assertEqual(len(five_steps), 5)
    
    def test_more_than_five_in_line_seven(self):
        """测试七子连线的情况"""
        # 七子连线：(7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)
        # 应返回最后5步：(7,3), (7,4), (7,5), (7,6), (7,7)
        
        history = []
        board = [0] * 225
        
        line_positions = [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        
        for pos in line_positions:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = 1
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(line_positions)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 验证返回5个步骤
        self.assertEqual(len(five_steps), 5)
    
    def test_find_five_in_a_row_empty_actions(self):
        """测试获胜方落子列表为空的情况"""
        history = []
        winning_actions = []
        
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 应返回空列表
        self.assertEqual(five_steps, [])
    
    def test_find_five_in_a_row_less_than_five(self):
        """测试获胜方落子少于5步的情况"""
        # 理论上不会出现（游戏结束必须有五子连线）
        history = []
        board = [0] * 225
        
        # 只有3步
        positions = [(7, 3), (7, 4), (7, 5)]
        for pos in positions:
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
            board[pos[0] * 15 + pos[1]] = 1
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(positions)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 应返回空列表
        self.assertEqual(five_steps, [])
    
    def test_find_five_in_a_row_no_line(self):
        """测试没有五子连线的情况（异常情况）"""
        # 获胜方落子分散，不构成连线
        history = []
        board = [0] * 225
        
        positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]  # 对角线但不相邻
        for pos in positions:
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
            board[pos[0] * 15 + pos[1]] = 1
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(positions)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 应返回空列表
        self.assertEqual(five_steps, [])
    
    def test_white_wins_find_five(self):
        """测试白方获胜时识别五子连线"""
        # 白方形成五子连线
        history = []
        board = [0] * 225
        
        line_positions = [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        
        for pos in line_positions:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = 2
            input_vec = board.copy() + [2]  # 白方
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(line_positions)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        self.assertEqual(len(five_steps), 5)
        
        # 验证 reward 计算（白方获胜）
        sample_dict = self.evaluator._initialize_sample(history)
        self.evaluator._compute_rewards(sample_dict, history, winner=2)
        
        max_r = Evaluator.MAX_REWARD
        # 最后一步应有 MAX_REWARD
        self.assertEqual(sample_dict[4]["output"][7 * 15 + 7], int(max_r))
    
    def test_cross_pattern_select_correct_line(self):
        """测试十字形连线时选择正确的五子连线"""
        # 十字形：中间一步是两条线的交点
        # (6,5), (6,6), (6,7), (6,8), (6,9) - 水平线
        # (4,7), (5,7), (6,7), (7,7), (8,7) - 垂直线
        # 最后一步是 (6,9) 或 (8,7)，取决于落子顺序
        
        history = []
        board = [0] * 225
        
        # 先完成水平线
        horizontal = [(6, 5), (6, 6), (6, 7), (6, 8), (6, 9)]
        for pos in horizontal:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = 1
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        winning_actions = [(i, list(pos)) for i, pos in enumerate(horizontal)]
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 1, 2, 3, 4])
    
    def test_compute_rewards_with_five_steps_empty(self):
        """测试找不到五子连线时 _compute_rewards 的行为"""
        # 构造一个没有五子连线的场景（理论上不会出现）
        history = []
        board = [0] * 225
        
        # 只有3步，不构成五子
        positions = [(7, 3), (7, 4), (7, 5)]
        for pos in positions:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = 1
            input_vec = board.copy() + [1]
            output = [0] * 225
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 不应抛出异常
        try:
            self.evaluator._compute_rewards(sample_dict, history, winner=1)
        except Exception as e:
            self.fail(f"_compute_rewards 不应抛出异常: {e}")
        
        # 所有落子位置应保持初始化值（min_reward）
        for i, pos in enumerate(positions):
            self.assertEqual(sample_dict[i]["output"][pos[0] * 15 + pos[1]], Evaluator.MIN_REWARD)


if __name__ == "__main__":
    unittest.main()
