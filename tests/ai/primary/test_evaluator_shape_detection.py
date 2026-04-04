"""评价器双向棋型检测法主流程测试

测试 Evaluator 类的双向棋型检测 reward 计算功能。
"""

import sys
import os
import unittest
import json
import shutil

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestEvaluatorShapeDetection(unittest.TestCase):
    """双向棋型检测法主流程测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_evaluator_shape_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        
        # 清理并创建测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
        
        # 创建评价器
        self.evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    # ==================== 连五测试 ====================
    
    def test_live_five_current_player(self):
        """测试连五：当前落子方形成连五，reward = MAX_REWARD"""
        # 构造棋盘：黑方在 (7,7) 落子形成横向连五
        # 黑子位置：(7,3), (7,4), (7,5), (7,6), (7,7) 落子后形成连五
        board = [0] * 225
        # 放置前4个黑子
        for col in range(3, 7):
            board[7 * 15 + col] = 1
        
        # 白子随便放几个
        board[0] = 2
        board[1] = 2
        
        # 检测 (7,7) 位置的 reward（模拟落子后形成连五）
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        self.assertEqual(reward, Evaluator.MAX_REWARD)
    
    def test_live_five_opponent(self):
        """测试连五：对手方形成连五，reward = MAX_REWARD * 0.95"""
        board = [0] * 225
        # 白方前4子
        for col in range(3, 7):
            board[7 * 15 + col] = 2
        
        # 黑子随便放
        board[0] = 1
        
        # 检测 (7,7) 位置对白方的价值
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        # 白方在此落子形成连五，敌方系数 0.95
        expected = int(Evaluator.MAX_REWARD * 0.95)
        self.assertEqual(reward, expected)
    
    # ==================== 活四测试 ====================
    
    def test_live_four_current_player(self):
        """测试活四：当前落子方形成活四，reward = 0.9 * MAX_REWARD"""
        board = [0] * 225
        # 黑方形成横向活四：(7,4), (7,5), (7,6)，落子 (7,7) 后形成活四
        # 两端 (7,3) 和 (7,8) 都是空的
        for col in range(4, 7):
            board[7 * 15 + col] = 1
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int(0.9 * Evaluator.MAX_REWARD)
        self.assertEqual(reward, expected)
    
    def test_live_four_opponent(self):
        """测试活四：对手方形成活四，reward = 0.9 * MAX_REWARD * 0.95"""
        board = [0] * 225
        # 白方形成横向活四
        for col in range(4, 7):
            board[7 * 15 + col] = 2
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)  # 当前黑方
        expected = int(0.9 * Evaluator.MAX_REWARD * 0.95)
        self.assertEqual(reward, expected)
    
    # ==================== 冲四测试 ====================
    
    def test_four_single_direction(self):
        """测试冲四：单方向四子，一端被阻挡"""
        board = [0] * 225
        # 黑方 (7,4), (7,5), (7,6)，白方阻挡 (7,3)
        # 黑方落子 (7,7) 形成冲四
        for col in range(4, 7):
            board[7 * 15 + col] = 1
        board[7 * 15 + 3] = 2  # 白方阻挡左端
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        # 冲四需要递归计算：模拟防守后的棋型
        # 这里先验证返回的是递归后的 reward
        self.assertIsInstance(reward, int)
        self.assertGreater(reward, 0)
    
    def test_multi_four(self):
        """测试多冲四：两个方向形成冲四，reward = 0.9 * MAX_REWARD"""
        board = [0] * 225
        # 黑方在 (7,7) 能形成双冲四
        # 横向冲四：(7,4), (7,5), (7,6) 已有黑子，左端 (7,3) 被白子阻挡
        # 落子 (7,7) 后横向形成 4 子冲四
        for col in range(4, 7):
            board[7 * 15 + col] = 1
        board[7 * 15 + 3] = 2  # 左端阻挡
        # 纵向冲四：(4,7), (5,7), (6,7) 已有黑子，上端 (3,7) 被白子阻挡
        # 落子 (7,7) 后纵向形成 4 子冲四
        for row in range(4, 7):
            board[row * 15 + 7] = 1
        board[3 * 15 + 7] = 2  # 上端阻挡
        # 落子 (7,7) 后，横向和纵向都形成冲四 -> 多冲四
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int(0.9 * Evaluator.MAX_REWARD)
        self.assertEqual(reward, expected)
    
    # ==================== 活三测试 ====================
    
    def test_live_three_single_direction(self):
        """测试活三：单方向三子，两端无阻挡"""
        board = [0] * 225
        # 黑方 (7,5), (7,6)，落子 (7,7) 形成活三
        # 两端 (7,4) 和 (7,8) 都是空的
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int(0.5 * Evaluator.MAX_REWARD)
        self.assertEqual(reward, expected)
    
    def test_multi_live_three(self):
        """测试多活三：两个方向形成活三"""
        board = [0] * 225
        # 黑方在 (7,7) 能形成双活三
        # 横向：(7,5), (7,6)
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        # 纵向：(5,7), (6,7)
        board[5 * 15 + 7] = 1
        board[6 * 15 + 7] = 1
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int(0.8 * Evaluator.MAX_REWARD)
        self.assertEqual(reward, expected)
    
    # ==================== 眠三测试 ====================
    
    def test_sleep_three_single_direction(self):
        """测试眠三：三子，一端被阻挡"""
        board = [0] * 225
        # 黑方 (7,5), (7,6)，白方阻挡 (7,4)
        # 落子 (7,7) 形成眠三
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        board[7 * 15 + 4] = 2  # 左端阻挡
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int((0.1 + 1 * 0.1) * Evaluator.MAX_REWARD)  # count_directions = 1
        self.assertEqual(reward, expected)
    
    def test_sleep_three_multi_direction(self):
        """测试眠三：多个方向形成眠三"""
        board = [0] * 225
        # 横向眠三
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        board[7 * 15 + 4] = 2
        # 纵向眠三
        board[5 * 15 + 7] = 1
        board[6 * 15 + 7] = 1
        board[4 * 15 + 7] = 2
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int((0.1 + 2 * 0.1) * Evaluator.MAX_REWARD)  # count_directions = 2
        self.assertEqual(reward, expected)
    
    # ==================== 活二测试 ====================
    
    def test_live_two_single_direction(self):
        """测试活二：二子，两端无阻挡"""
        board = [0] * 225
        # 黑方 (7,6)，落子 (7,7) 形成活二
        board[7 * 15 + 6] = 1
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int((0.3 + 1 * 0.1) * Evaluator.MAX_REWARD)  # count_directions = 1
        self.assertEqual(reward, expected)
    
    def test_live_two_multi_direction(self):
        """测试活二：多个方向形成活二"""
        board = [0] * 225
        # 横向活二
        board[7 * 15 + 6] = 1
        # 纵向活二
        board[6 * 15 + 7] = 1
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int((0.3 + 2 * 0.1) * Evaluator.MAX_REWARD)  # count_directions = 2
        self.assertEqual(reward, expected)
    
    # ==================== 眠二/独一测试 ====================
    
    def test_sleep_two(self):
        """测试眠二：二子，一端被阻挡，reward = 0"""
        board = [0] * 225
        # 黑方 (7,6)，白方阻挡 (7,5)
        board[7 * 15 + 6] = 1
        board[7 * 15 + 5] = 2
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        # 眠二 reward = 0，但可能有其他方向的棋型
        # 这里应该返回独一 = 0（如果只有这一个眠二方向）
        # 实际上眠二为0，但如果同时有独一，独一也为0
        # 所以最终reward应该是0
        self.assertEqual(reward, 0)
    
    def test_single_piece(self):
        """测试独一：单独一子，reward = 0"""
        board = [0] * 225
        # 空棋盘，任何位置落子都是独一
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        self.assertEqual(reward, 0)
    
    # ==================== Mask 测试 ====================
    
    def test_mask_for_occupied_position(self):
        """测试已落子位置的 mask 为 0"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 有黑子
        board[7 * 15 + 8] = 2  # (7,8) 有白子
        
        history = [{
            "input": board + [1],
            "output": [0] * 225,
            "position": [0, 0]
        }]
        
        sample_dict = self.evaluator._compute_all_rewards(history)
        
        # 已落子位置 mask = 0
        self.assertEqual(sample_dict[0]["mask"][7 * 15 + 7], 0)
        self.assertEqual(sample_dict[0]["mask"][7 * 15 + 8], 0)
        # 空位 mask = 1
        self.assertEqual(sample_dict[0]["mask"][0], 1)
    
    def test_reward_for_occupied_position(self):
        """测试已落子位置的 reward 为 0"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 有黑子
        
        history = [{
            "input": board + [1],
            "output": [0] * 225,
            "position": [0, 0]
        }]
        
        sample_dict = self.evaluator._compute_all_rewards(history)
        
        # 已落子位置 reward = 0
        self.assertEqual(sample_dict[0]["output"][7 * 15 + 7], 0)
    
    # ==================== 双向检测测试 ====================
    
    def test_bidirectional_max_reward(self):
        """测试双向检测：取两边的最大值"""
        board = [0] * 225
        # 使用不同方向避免冲突：
        # 横向：我方黑子 (7,5), (7,6)，落子 (7,7) 后形成活三
        # 纵向：敌方白子 (4,7), (5,7), (6,7)，落子 (7,7) 后形成活四
        
        # 我方横向活三：两端空
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        
        # 敌方纵向活四：(4,7), (5,7), (6,7) + (7,7) = 4子连续
        # 上端 (3,7) 空，下端 (8,7) 空 -> 活四
        board[4 * 15 + 7] = 2
        board[5 * 15 + 7] = 2
        board[6 * 15 + 7] = 2
        
        # 当前落子方是黑方
        # 我方在 (7,7) 落子：横向活三 reward = 0.5 * MAX = 500000
        # 敌方在 (7,7) 落子：纵向活四 reward = 0.9 * MAX * 0.95 = 855000
        # 取 max = 855000
        
        reward = self.evaluator._compute_position_reward(board, 7, 7, 1)
        expected = int(0.9 * Evaluator.MAX_REWARD * 0.95)
        self.assertEqual(reward, expected)


if __name__ == '__main__':
    unittest.main()
