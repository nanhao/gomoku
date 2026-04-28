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

from ai.engine import AIEngine, create_engine
from ai.models.random.model import RandomModel
from ai.models.cnn_v2.model import CNNV2Model


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
    
    def test_predict_probability_format(self):
        """[primary] 测试输出为概率格式（浮点数）"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 验证所有元素为浮点数
        self.assertEqual(len(result), 225)
        for prob in result:
            self.assertIsInstance(prob, float)
    
    def test_predict_probability_sum(self):
        """[primary] 测试空位概率总和为 1"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 验证概率总和为 1
        prob_sum = sum(result)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
    
    def test_predict_empty_cells_uniform_probability(self):
        """[primary] 测试空位均匀分布"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 空棋盘有 225 个空位，每个概率应为 1/225
        expected_prob = 1.0 / 225.0
        for prob in result:
            self.assertAlmostEqual(prob, expected_prob, places=5)
    
    def test_predict_with_pieces(self):
        """[primary] 测试有棋子时的概率分布"""
        model = RandomModel()
        
        # 创建有棋子的棋盘：中心有黑棋
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7, 7) 位置有黑棋
        board.append(1)  # 当前黑方落子
        
        result = model.predict(board)
        
        # 验证概率总和为 1
        prob_sum = sum(result)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        
        # 验证已落子位置概率为 0
        self.assertEqual(result[7 * 15 + 7], 0.0)
        
        # 验证空位概率相同（均匀分布）
        empty_count = 224  # 225 - 1 个已落子
        expected_prob = 1.0 / empty_count
        for i, prob in enumerate(result):
            if board[i] == 0:  # 空位
                self.assertAlmostEqual(prob, expected_prob, places=5)
    
    def test_predict_all_occupied(self):
        """[primary] 测试棋盘满时的输出
        
        修复后的行为：满棋盘时返回均匀分布 [1/225]*225，
        确保概率之和为1（满足接口契约），而非全0。
        """
        model = RandomModel()
        
        # 创建满棋盘
        board = [1] * 225  # 全部有棋子
        board.append(1)
        
        result = model.predict(board)
        
        # 修复后：满棋盘返回均匀分布，概率之和为1
        self.assertEqual(len(result), 225)
        expected_prob = 1.0 / 225
        for prob in result:
            self.assertAlmostEqual(prob, expected_prob, places=5,
                                   msg="满棋盘应返回均匀分布")
        self.assertAlmostEqual(sum(result), 1.0, places=5,
                               msg="概率之和应为1")
    
    def test_predict_index_mapping(self):
        """测试索引映射：索引 i 对应位置 (i//15, i%15)"""
        model = RandomModel()
        board_vector = [0] * 226
        result = model.predict(board_vector)
        
        # 索引 0 -> (0, 0)，索引 1 -> (0, 1)，索引 15 -> (1, 0)
        # 空棋盘每个位置概率相同
        expected_prob = 1.0 / 225.0
        self.assertEqual(len(result), 225)
        self.assertAlmostEqual(result[0], expected_prob, places=5)   # (0, 0)
        self.assertAlmostEqual(result[1], expected_prob, places=5)   # (0, 1)
        self.assertAlmostEqual(result[15], expected_prob, places=5)  # (1, 0)
    
    def test_name(self):
        """测试模型名称"""
        model = RandomModel()
        self.assertEqual(model.name, "random")


class TestCreateEngine(unittest.TestCase):
    """测试 create_engine 函数"""

    def test_create_engine_random(self):
        """测试创建 random 引擎"""
        engine = create_engine("random")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model.name, "random")

    def test_create_engine_fc_v1(self):
        """测试创建 fc_v1 引擎"""
        engine = create_engine("fc_v1")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model.name, "fc_v1")

    def test_create_engine_cnn(self):
        """测试创建 cnn_v1 引擎"""
        engine = create_engine("cnn_v1")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model.name, "cnn_v1")

    def test_create_engine_cnn_v2(self):
        """测试创建 cnn_v2 引擎"""
        engine = create_engine("cnn_v2")
        self.assertIsNotNone(engine)
        self.assertIsInstance(engine.model, CNNV2Model,
                             "cnn_v2 应映射到 CNNV2Model，而非回退到 RandomModel")
        self.assertEqual(engine.model.name, "cnn_v2")

    def test_create_engine_unknown_uses_random(self):
        """测试未知模型名称时使用 random 作为兜底"""
        engine = create_engine("unknown_model")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model.name, "random")

    def test_create_engine_cnn_v2_not_random(self):
        """测试 cnn_v2 不会静默回退到 RandomModel"""
        engine = create_engine("cnn_v2")
        self.assertNotIsInstance(engine.model, RandomModel,
                                 "cnn_v2 不应回退到 RandomModel")


if __name__ == "__main__":
    unittest.main()
