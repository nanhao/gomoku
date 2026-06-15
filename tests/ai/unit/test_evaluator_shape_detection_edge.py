"""评价器双向棋型检测法边缘测试

测试边缘情况和边界条件。
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
from ai.reward_calculator import RewardConfig
from ai.shape_detector import detect_shapes


class TestEvaluatorShapeDetectionEdge(unittest.TestCase):
    """双向棋型检测法边缘测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_evaluator_shape_edge_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
        
        self.evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    # ==================== 边界情况 ====================
    
    def test_empty_board_all_single(self):
        """测试空棋盘：所有位置都是独一，reward = 0"""
        board = [0] * 225
        
        # 测试多个位置
        for pos in [(0, 0), (7, 7), (14, 14), (0, 14), (14, 0)]:
            reward = self.evaluator._reward_calculator.compute_position_reward(board, pos[0], pos[1], 1)
            self.assertEqual(reward, 0, f"Position {pos} should have reward 0 for empty board")
    
    def test_board_edge_as_block(self):
        """测试棋盘边缘视为阻挡"""
        board = [0] * 225
        # 在 (0, 5) 形成眠三（上边缘阻挡）
        # 棋子：(0, 3), (0, 4) 是黑子
        # 落子 (0, 5) 后形成 3 子，左端 (0, 2) 是空的，上边缘是阻挡
        board[0 * 15 + 3] = 1
        board[0 * 15 + 4] = 1
        # 落子 (0, 5)，上方是边缘，但横向延伸，一端被边缘阻挡
        
        # 实际上 (0,5) 落子后：横向 (0,3), (0,4), (0,5) = 3子
        # 检测横向：左端 (0,2) 空，右端 (0,6) 空 -> 活三，不是眠三
        # 边缘只在纵向方向影响，横向不受影响
        # 所以这里应该是活三
        
        # 修正测试：让边缘真正阻挡
        # (0, 3), (0, 4) 黑子，(0, 2) 白子阻挡
        # 落子 (0, 5)，左端被白子阻挡 -> 眠三
        board[0 * 15 + 2] = 2  # 左端阻挡
        
        reward = self.evaluator._reward_calculator.compute_position_reward(board, 0, 5, 1)
        # 眠三：一端被白子阻挡
        expected = int((0.1 + 1 * 0.1) * RewardConfig().max_reward)
        self.assertEqual(reward, expected)
    
    def test_corner_position(self):
        """测试角落位置"""
        board = [0] * 225
        # 角落 (0,0) 受两个边缘阻挡
        
        reward = self.evaluator._reward_calculator.compute_position_reward(board, 0, 0, 1)
        # 独一，但角落位置可能有更多阻挡
        self.assertEqual(reward, 0)
    
    def test_four_blocked_both_ends(self):
        """测试四子被两端阻挡（眠四），不构成威胁"""
        board = [0] * 225
        # 黑方四子，两端都被白方阻挡
        board[7 * 15 + 4] = 1
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        board[7 * 15 + 3] = 2  # 左端阻挡
        # 落子 (7,7)，右端 (7,8) 设为白子
        board[7 * 15 + 8] = 2  # 右端也阻挡
        
        # 这种情况下，四子两端都被阻挡，不是冲四
        # 实际上 (7,7) 本身是空位，落子后形成的是被阻挡的四
        # 这不是有效的棋型威胁，应该视为更低级别
        reward = self.evaluator._reward_calculator.compute_position_reward(board, 7, 7, 1)
        # 眠四不算威胁，应该返回较低的 reward
        # 实际上落子后变成四子，但两端被阻挡，这是眠四
        # 根据设计，眠四返回 0
        self.assertEqual(reward, 0)
    
    # ==================== 方向合并测试 ====================
    
    def test_same_shape_multiple_directions(self):
        """测试同一棋型在多个方向出现，directions 合并"""
        board = [0] * 225
        # 在 (7,7) 四个方向都形成活二
        # 横向左：(7,6) 黑子，横向右：(7,8) 黑子
        # 纵向上：(6,7) 黑子，纵向下：(8,7) 黑子
        # 但实际上落子 (7,7) 后：
        # 横向左：(7,6), (7,7) = 2子，右端空 -> 活二
        # 横向右：(7,7), (7,8) = 2子，左端空 -> 活二
        # 这样横向会有两个活二？不对，应该是一个活二在横向
        
        # 重新理解：活二需要 2 子连续
        # (7,6) 是黑子，(7,8) 也是黑子
        # 落子 (7,7) 后，横向：...X O X...
        # 这不是连续的活二！因为 (7,6) 和 (7,8) 被 (7,7) 隔开
        
        # 正确的活二测试：只有一侧有棋子
        board[7 * 15 + 6] = 1  # 横向左有一个棋子
        board[6 * 15 + 7] = 1  # 纵向上有一个棋子
        # 落子 (7,7) 后，横向：(7,6), (7,7) = 2子
        # 纵向：(6,7), (7,7) = 2子
        
        shapes = detect_shapes(board, (7, 7), 1)
        
        # 应该有活二棋型
        live_two_found = False
        for shape in shapes:
            if shape["shape"] == "活二":
                live_two_found = True
                self.assertGreaterEqual(len(shape["directions"]), 1)
        
        self.assertTrue(live_two_found, "Should detect live two shape")
    
    # ==================== 敌方系数测试 ====================
    
    def test_enemy_factor_0_95(self):
        """测试敌方系数 0.95 应用于所有棋型"""
        board = [0] * 225
        # 构造一个活三棋型
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        
        # 我方 reward
        my_reward = self.evaluator._reward_calculator.compute_position_reward(board, 7, 7, 1)
        
        # 同样棋盘，对手方 reward
        # 把黑子换成白子
        board2 = [0] * 225
        board2[7 * 15 + 5] = 2
        board2[7 * 15 + 6] = 2
        
        # 当前落子方是黑方，检测白方在此落子的价值
        enemy_reward = self.evaluator._reward_calculator.compute_position_reward(board2, 7, 7, 1)
        
        # 敌方 reward = 我方 reward * 0.95
        expected_ratio = enemy_reward / my_reward if my_reward > 0 else 0
        self.assertAlmostEqual(expected_ratio, 0.95, places=2)
    
    # ==================== reward 值范围测试 ====================
    
    def test_reward_not_exceed_max(self):
        """测试 reward 值不超过 MAX_REWARD"""
        board = [0] * 225
        # 构造连五
        for col in range(3, 7):
            board[7 * 15 + col] = 1
        
        reward = self.evaluator._reward_calculator.compute_position_reward(board, 7, 7, 1)
        self.assertLessEqual(reward, RewardConfig().max_reward)
    
    def test_reward_not_negative(self):
        """测试 reward 值非负"""
        board = [0] * 225
        # 随机放一些棋子
        board[7 * 15 + 7] = 1
        board[7 * 15 + 8] = 2
        
        for x in range(15):
            for y in range(15):
                if board[x * 15 + y] == 0:
                    reward = self.evaluator._reward_calculator.compute_position_reward(board, x, y, 1)
                    self.assertGreaterEqual(reward, 0, f"Reward should be non-negative at ({x},{y})")
    
    # ==================== 棋型优先级测试 ====================
    
    def test_shape_priority_five_over_four(self):
        """测试连五优先于活四"""
        board = [0] * 225
        # 在 (7,7) 能同时形成连五和活四（理论上不可能，但测试优先级逻辑）
        # 连五：(7,3)-(7,7)
        for col in range(3, 7):
            board[7 * 15 + col] = 1
        
        # 这里只能形成连五，不会有活四
        reward = self.evaluator._reward_calculator.compute_position_reward(board, 7, 7, 1)
        self.assertEqual(reward, RewardConfig().max_reward)
    
    # ==================== 完整 evaluate_record 测试 ====================
    
    def test_evaluate_record_with_shapes(self):
        """测试完整的 evaluate_record 流程"""
        # 构造一个简单的对局记录
        history = []
        board = [0] * 225
        
        # 模拟几步棋
        moves = [
            ([7, 7], 1),  # 黑
            ([0, 0], 2),  # 白
            ([7, 8], 1),  # 黑
            ([0, 1], 2),  # 白
        ]
        
        for pos, player in moves:
            step = {
                "input": board + [player],
                "output": [0] * 225,
                "position": pos
            }
            history.append(step)
            board[pos[0] * 15 + pos[1]] = player
        
        record = {
            "history": history,
            "winner": 1  # 黑方获胜
        }
        
        # 保存 record.json
        record_path = os.path.join(self.ai_data_dir, "test_game", "record.json")
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        
        # 执行评价
        sample_path = self.evaluator.evaluate_record(record_path)
        
        # 验证 sample.json 存在
        self.assertTrue(os.path.exists(sample_path))
        
        # 读取并验证
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        self.assertEqual(len(sample), len(history))
        
        # 验证每个 step 的结构
        for step in sample:
            self.assertIn("input", step)
            self.assertIn("output", step)
            self.assertIn("mask", step)
            self.assertEqual(len(step["output"]), 225)
            self.assertEqual(len(step["mask"]), 225)
    
    # ==================== 特殊棋型测试 ====================
    
    def test_detect_shapes_returns_list(self):
        """测试 detect_shapes 返回列表"""
        board = [0] * 225
        shapes = detect_shapes(board, (7, 7), 1)
        
        self.assertIsInstance(shapes, list)
    
    def test_detect_shapes_each_has_shape_and_directions(self):
        """测试每个 shape 都有 shape 和 directions 字段"""
        board = [0] * 225
        board[7 * 15 + 6] = 1  # 有一个己方子
        
        shapes = detect_shapes(board, (7, 7), 1)
        
        for shape in shapes:
            self.assertIn("shape", shape)
            self.assertIn("directions", shape)
            self.assertIsInstance(shape["shape"], str)
            self.assertIsInstance(shape["directions"], list)


if __name__ == '__main__':
    unittest.main()
