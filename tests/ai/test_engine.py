"""AI 引擎测试

测试 AIEngine 类的所有公共方法。
"""

import sys
import os
import unittest

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.engine import AIEngine
from ai.models.random.model import RandomModel


class TestAIEngine(unittest.TestCase):
    """AI 引擎测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.model = RandomModel()
        self.engine = AIEngine(self.model)
    
    def test_board_to_vector_empty_board(self):
        """测试空棋盘转向量"""
        board = [[0] * 15 for _ in range(15)]
        vector = AIEngine.board_to_vector(board, 1)
        
        # 验证长度
        self.assertEqual(len(vector), 226)
        # 前 225 位都是 0
        self.assertEqual(vector[:225], [0] * 225)
        # 最后一位是当前落子方
        self.assertEqual(vector[225], 1)
    
    def test_board_to_vector_with_pieces(self):
        """测试有棋子的棋盘转向量"""
        board = [[0] * 15 for _ in range(15)]
        board[7][7] = 1  # 黑子
        board[7][8] = 2  # 白子
        
        vector = AIEngine.board_to_vector(board, 2)
        
        # 验证长度
        self.assertEqual(len(vector), 226)
        # 第 7 行第 7 列 = 索引 7*15 + 7 = 112
        self.assertEqual(vector[112], 1)
        # 第 7 行第 8 列 = 索引 7*15 + 8 = 113
        self.assertEqual(vector[113], 2)
        # 最后一位是当前落子方
        self.assertEqual(vector[225], 2)
    
    def test_get_move_empty_board(self):
        """测试空棋盘获取落子位置"""
        board = [[0] * 15 for _ in range(15)]
        x, y = self.engine.get_move(board, 1)
        
        # 验证坐标范围
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)
        # 验证位置为空
        self.assertEqual(board[x][y], 0)
    
    def test_get_move_partial_board(self):
        """测试部分填充的棋盘"""
        board = [[0] * 15 for _ in range(15)]
        # 填充中心区域
        for i in range(5, 10):
            for j in range(5, 10):
                board[i][j] = 1
        
        x, y = self.engine.get_move(board, 2)
        
        # 验证坐标范围
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)
        # 验证位置为空
        self.assertEqual(board[x][y], 0)
        # 验证不在已填充区域
        self.assertFalse(5 <= x < 10 and 5 <= y < 10)
    
    def test_get_move_full_board(self):
        """测试满棋盘抛出异常"""
        board = [[1] * 15 for _ in range(15)]
        
        with self.assertRaises(ValueError) as context:
            self.engine.get_move(board, 1)
        
        self.assertIn("棋盘已满", str(context.exception))
    
    def test_get_move_returns_empty_position(self):
        """测试返回的位置一定是空位"""
        board = [[0] * 15 for _ in range(15)]
        # 只留一个空位
        for i in range(15):
            for j in range(15):
                if not (i == 14 and j == 14):
                    board[i][j] = 1
        
        x, y = self.engine.get_move(board, 2)
        
        # 必须是唯一的空位
        self.assertEqual(x, 14)
        self.assertEqual(y, 14)


class TestRandomModel(unittest.TestCase):
    """Random 模型测试类"""
    
    def test_predict_output_length(self):
        """测试输出长度"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 验证长度 225
        self.assertEqual(len(result), 225)
    
    def test_predict_all_rewards_zero(self):
        """测试所有 reward 为 0"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 验证所有元素为 0（每个元素对应一个位置的 reward）
        self.assertEqual(len(result), 225)
        self.assertTrue(all(r == 0 for r in result))
    
    def test_predict_index_mapping(self):
        """测试索引映射：索引 i 对应位置 (i//15, i%15)"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 索引 0 -> (0, 0)，索引 1 -> (0, 1)，索引 15 -> (1, 0)
        # random 模型所有 reward 为 0，就验证长度和全为 0
        self.assertEqual(len(result), 225)
        self.assertEqual(result[0], 0)   # (0, 0)
        self.assertEqual(result[1], 0)   # (0, 1)
        self.assertEqual(result[15], 0)  # (1, 0)
    
    def test_name(self):
        """测试模型名称"""
        model = RandomModel()
        self.assertEqual(model.name, "random")


if __name__ == "__main__":
    unittest.main()
