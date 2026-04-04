"""Evaluator 新 Reward 算法主流程测试

测试新的 reward 反向传播算法：
- 只给构成五子连线的5步分配 reward
- 失败方不做处理
"""

import sys
import os
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestEvaluatorNewReward(unittest.TestCase):
    """新 reward 算法主流程测试"""
    
    def setUp(self):
        """测试前准备"""
        self.evaluator = Evaluator()
    
    def _create_history_with_five_in_row(self, line_positions, interleaved_moves=None):
        """
        构造包含五子连线的历史记录
        
        Args:
            line_positions: 构成五子连线的5个位置 [(x,y), (x,y), ...]
            interleaved_moves: 中间夹杂的其他落子 [(position, player), ...]
        
        Returns:
            history 列表
        """
        history = []
        board = [0] * 225
        
        # 合并并排序所有落子
        all_moves = []
        
        # 添加连线落子（黑方=1）
        for i, pos in enumerate(line_positions):
            all_moves.append((pos, 1, i))  # (position, player, line_index)
        
        # 添加夹杂的落子
        if interleaved_moves:
            for move in interleaved_moves:
                all_moves.append((move[0], move[1], None))
        
        # 按落子顺序排序（这里简化处理，假设按输入顺序）
        # 实际测试中需要更精确的排序逻辑
        
        for pos, player, _ in all_moves:
            input_vec = board.copy() + [player]
            output = [0] * 225
            
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
            
            board[pos[0] * 15 + pos[1]] = player
        
        return history, board
    
    def test_find_five_in_a_row_horizontal(self):
        """测试识别水平方向五子连线"""
        # 构造黑方水平五子连线：(7,3), (7,4), (7,5), (7,6), (7,7)
        line_positions = [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        
        history = []
        board = [0] * 225
        
        # 按顺序落子
        for i, pos in enumerate(line_positions):
            # 更新棋盘状态（先更新再记录，这样每步的input反映的是落子后的状态）
            board[pos[0] * 15 + pos[1]] = 1
            input_vec = board.copy() + [1]  # 黑方
            output = [0] * 225
            
            history.append({
                "input": input_vec,
                "output": output,
                "position": list(pos)
            })
        
        # 构造获胜方落子列表
        winning_actions = [(i, list(pos)) for i, pos in enumerate(line_positions)]
        
        # 调用 _find_five_in_a_row
        five_steps = self.evaluator._find_five_in_a_row(winning_actions, history)
        
        # 验证返回5个步骤索引
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 1, 2, 3, 4])
    
    def test_find_five_in_a_row_vertical(self):
        """测试识别垂直方向五子连线"""
        # 构造黑方垂直五子连线：(3,7), (4,7), (5,7), (6,7), (7,7)
        line_positions = [(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]
        
        history = []
        board = [0] * 225
        
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
        
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 1, 2, 3, 4])
    
    def test_find_five_in_a_row_diagonal_main(self):
        """测试识别主对角线方向五子连线"""
        # 构造黑方主对角线五子连线：(3,3), (4,4), (5,5), (6,6), (7,7)
        line_positions = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
        
        history = []
        board = [0] * 225
        
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
        
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 1, 2, 3, 4])
    
    def test_find_five_in_a_row_diagonal_anti(self):
        """测试识别副对角线方向五子连线"""
        # 构造黑方副对角线五子连线：(3,11), (4,10), (5,9), (6,8), (7,7)
        line_positions = [(3, 11), (4, 10), (5, 9), (6, 8), (7, 7)]
        
        history = []
        board = [0] * 225
        
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
        
        self.assertEqual(len(five_steps), 5)
        self.assertEqual(five_steps, [0, 1, 2, 3, 4])
    
    def test_compute_rewards_only_five_steps(self):
        """测试只给构成五子连线的5步分配 reward"""
        # 构造场景：黑方下了9步，其中5步构成连线，另外4步是无关落子
        # 连线位置：(7,3), (7,4), (7,5), (7,6), (7,7)
        # 无关落子：(0,0), (1,1), (2,2), (3,3)
        
        history = []
        board = [0] * 225
        
        # 交错落子：连线步 + 无关步
        moves = [
            ([7, 3], 1),   # 黑 - 连线1
            ([0, 0], 1),   # 黑 - 无关1
            ([7, 4], 1),   # 黑 - 连线2
            ([1, 1], 1),   # 黑 - 无关2
            ([7, 5], 1),   # 黑 - 连线3
            ([2, 2], 1),   # 黑 - 无关3
            ([7, 6], 1),   # 黑 - 连线4
            ([3, 3], 1),   # 黑 - 无关4
            ([7, 7], 1),   # 黑 - 连线5 (获胜)
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
        
        # 初始化 sample_dict
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 计算 reward
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        # 验证只有构成连线的5步有正向 reward
        # 连线步索引：0, 2, 4, 6, 8
        # j5 (step 8): MAX_REWARD
        # j4 (step 6): MAX_REWARD * 0.9
        # j3 (step 4): MAX_REWARD * 0.9^2
        # j2 (step 2): MAX_REWARD * 0.9^3
        # j1 (step 0): MAX_REWARD * 0.9^4
        
        max_r = Evaluator.MAX_REWARD
        r = Evaluator.DECAY_RATE
        
        # 验证 j5 (step 8, position [7,7])
        self.assertEqual(sample_dict[8]["output"][7 * 15 + 7], int(max_r))
        
        # 验证 j4 (step 6, position [7,6])
        self.assertEqual(sample_dict[6]["output"][7 * 15 + 6], int(max_r * r))
        
        # 验证 j3 (step 4, position [7,5])
        self.assertEqual(sample_dict[4]["output"][7 * 15 + 5], int(max_r * r * r))
        
        # 验证无关步没有正向 reward（保持 MIN_REWARD，因为已落子）
        # step 1: position [0,0]
        self.assertEqual(sample_dict[1]["output"][0 * 15 + 0], Evaluator.MIN_REWARD)
        
        # step 3: position [1,1]
        self.assertEqual(sample_dict[3]["output"][1 * 15 + 1], Evaluator.MIN_REWARD)
    
    def test_compute_rewards_losing_side_unchanged(self):
        """测试失败方 reward 保持初始化值"""
        # 构造场景：黑方获胜，白方失败
        # 黑方：(7,3), (7,4), (7,5), (7,6), (7,7) - 五子连线
        # 白方：(0,0), (1,1), (2,2), (3,3)
        
        history = []
        board = [0] * 225
        
        moves = [
            ([7, 3], 1),   # 黑
            ([0, 0], 2),   # 白
            ([7, 4], 1),   # 黑
            ([1, 1], 2),   # 白
            ([7, 5], 1),   # 黑
            ([2, 2], 2),   # 白
            ([7, 6], 1),   # 黑
            ([3, 3], 2),   # 白
            ([7, 7], 1),   # 黑（获胜）
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
        
        # 初始化 sample_dict
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 记录白方落子位置的初始值
        white_positions = [
            (1, 0 * 15 + 0),  # step 1
            (3, 1 * 15 + 1),  # step 3
            (5, 2 * 15 + 2),  # step 5
            (7, 3 * 15 + 3),  # step 7
        ]
        
        initial_white_rewards = []
        for step_idx, pos_idx in white_positions:
            initial_white_rewards.append(sample_dict[step_idx]["output"][pos_idx])
        
        # 计算 reward
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        # 验证白方落子位置保持 min_reward（初始化值）
        for i, (step_idx, pos_idx) in enumerate(white_positions):
            self.assertEqual(sample_dict[step_idx]["output"][pos_idx], Evaluator.MIN_REWARD)
    
    def test_reward_decay_sequence(self):
        """测试 reward 衰减序列正确"""
        # 构造简单的五子连线
        line_positions = [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        
        history = []
        board = [0] * 225
        
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
        
        sample_dict = self.evaluator._initialize_sample(history)
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        max_r = Evaluator.MAX_REWARD
        r = Evaluator.DECAY_RATE
        
        # j5 (step 4): MAX_REWARD
        self.assertEqual(sample_dict[4]["output"][7 * 15 + 7], int(max_r))
        
        # j4 (step 3): MAX_REWARD * r
        self.assertEqual(sample_dict[3]["output"][7 * 15 + 6], int(max_r * r))
        
        # j3 (step 2): MAX_REWARD * r^2
        self.assertEqual(sample_dict[2]["output"][7 * 15 + 5], int(max_r * r * r))
        
        # j2 (step 1): MAX_REWARD * r^3
        self.assertEqual(sample_dict[1]["output"][7 * 15 + 4], int(max_r * r * r * r))
        
        # j1 (step 0): MAX_REWARD * r^4
        self.assertEqual(sample_dict[0]["output"][7 * 15 + 3], int(max_r * r * r * r * r))


if __name__ == "__main__":
    unittest.main()
