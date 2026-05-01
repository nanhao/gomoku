"""AI 引擎测试

测试 AIEngine 类的所有公共方法。
"""

import sys
import os
import random
import unittest

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import BaseModel
from ai.engine import AIEngine, create_engine
from ai.models.random.model import RandomModel
from ai.models.cnn_v2.model import CNNV2Model


class StubModel(BaseModel):
    """测试桩模型：predict 恒定返回构造时指定的概率向量。

    用于在不依赖真实神经网络的前提下，精确控制 AIEngine 接收到的概率分布，
    从而验证落子策略（argmax / sampling / 未知策略）的行为。
    """

    def __init__(self, probs):
        # 深拷贝避免外部修改影响
        self._probs = list(probs)
        # network 设为 None：BaseModel 约定 network 可为 None（如 RandomModel）
        self.network = None
        # 记录 predict 调用次数，便于验证"传入 probs 时跳过 predict"
        self.predict_call_count = 0

    def predict(self, board_vector):
        self.predict_call_count += 1
        return list(self._probs)

    @property
    def name(self):
        return "stub"

    def save(self, model_path=None):
        raise NotImplementedError("StubModel 不支持保存")

    def load(self, model_path=None):
        raise NotImplementedError("StubModel 不支持加载")


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

    def test_create_engine_unknown_raises_value_error(self):
        """测试未知模型名称时直接抛出 ValueError，不再回退到 RandomModel"""
        with self.assertRaises(ValueError) as ctx:
            create_engine("unknown_model")
        self.assertIn("未知的模型", str(ctx.exception))

    def test_create_engine_cnn_v2_not_random(self):
        """测试 cnn_v2 不会静默回退到 RandomModel"""
        engine = create_engine("cnn_v2")
        self.assertNotIsInstance(engine.model, RandomModel,
                                 "cnn_v2 不应回退到 RandomModel")


class TestGetMoveStrategy(unittest.TestCase):
    """落子策略测试：argmax（默认） / sampling（未实现） / 未知策略。

    对应设计文档 design/engine_move_strategy_design.md 中的主流程测试用例 1-10。
    """

    def _empty_board(self):
        return [[0] * 15 for _ in range(15)]

    def _make_probs(self, pairs, total=225):
        """辅助：根据 (index, prob) 对构造 225 维概率向量，其余位置为 0。"""
        probs = [0.0] * total
        for idx, p in pairs:
            probs[idx] = p
        return probs

    # ---------- argmax ----------

    def test_argmax_picks_unique_max(self):
        """用例 1：唯一最大值时必返回该位置。"""
        board = self._empty_board()
        # (7,7) 索引 = 7*15+7 = 112，概率明显最大
        probs = self._make_probs([(112, 0.9), (0, 0.05), (1, 0.05)])
        engine = AIEngine(StubModel(probs))

        for _ in range(20):
            x, y = engine.get_move(board, 1, strategy="argmax")
            self.assertEqual((x, y), (7, 7))

    def test_argmax_ties_broken_randomly(self):
        """用例 2：多个位置概率并列最大时，从并列候选中随机挑一个。"""
        board = self._empty_board()
        # (0,0)=idx 0 与 (0,1)=idx 1 并列 0.4，(0,2)=idx 2 为 0.2
        probs = self._make_probs([(0, 0.4), (1, 0.4), (2, 0.2)])
        engine = AIEngine(StubModel(probs))

        # 固定随机种子，避免偶发失败；仍要求覆盖两个并列候选
        random.seed(20260501)
        seen = set()
        for _ in range(200):
            seen.add(engine.get_move(board, 1, strategy="argmax"))

        self.assertIn((0, 0), seen)
        self.assertIn((0, 1), seen)
        # (0,2) 概率更低，不应被选中
        self.assertNotIn((0, 2), seen)

    def test_argmax_skips_occupied_max(self):
        """用例 3：概率最大的位置已落子时，返回空位中概率最大的。"""
        board = self._empty_board()
        board[7][7] = 1  # 占住 idx 112
        # idx 112 概率最大，但已占用；idx 0 次大且为空
        probs = self._make_probs([(112, 0.9), (0, 0.5), (1, 0.1)])
        engine = AIEngine(StubModel(probs))

        x, y = engine.get_move(board, 2, strategy="argmax")
        self.assertEqual((x, y), (0, 0))

    def test_argmax_all_zero_probs_falls_back_to_uniform_pick(self):
        """用例 4：所有空位概率都为 0 时仍应挑一个空位，不抛异常。"""
        board = self._empty_board()
        # 模型把所有概率压在 (0,0)，且 (0,0) 已落子 → 空位全 0
        board[0][0] = 1
        probs = self._make_probs([(0, 1.0)])
        engine = AIEngine(StubModel(probs))

        x, y = engine.get_move(board, 2, strategy="argmax")
        self.assertEqual(board[x][y], 0, "回退时必须选空位")
        self.assertNotEqual((x, y), (0, 0), "不能选已占用的位置")

    # ---------- 默认行为 ----------

    def test_default_strategy_is_argmax(self):
        """用例 5：不传 strategy 时，行为等价于 argmax。"""
        board = self._empty_board()
        probs = self._make_probs([(50, 0.8), (51, 0.2)])
        engine = AIEngine(StubModel(probs))

        # 多次调用都应稳定返回概率最大的位置
        for _ in range(20):
            self.assertEqual(engine.get_move(board, 1), (50 // 15, 50 % 15))

    # ---------- sampling 未实现 ----------

    def test_sampling_strategy_raises_not_implemented(self):
        """用例 7：sampling 策略尚未实现，应抛 NotImplementedError。"""
        board = self._empty_board()
        probs = [1.0 / 225] * 225
        engine = AIEngine(StubModel(probs))

        with self.assertRaises(NotImplementedError):
            engine.get_move(board, 1, strategy="sampling")

    # ---------- 未知策略 ----------

    def test_unknown_strategy_raises_value_error(self):
        """用例 8：未知 strategy 值应抛 ValueError。"""
        board = self._empty_board()
        probs = [1.0 / 225] * 225
        engine = AIEngine(StubModel(probs))

        with self.assertRaises(ValueError):
            engine.get_move(board, 1, strategy="foo")

    # ---------- 棋盘已满 ----------

    def test_full_board_raises_value_error_under_argmax(self):
        """用例 9：棋盘已满时仍抛 ValueError（与既有行为一致）。"""
        board = [[1] * 15 for _ in range(15)]
        probs = [1.0 / 225] * 225
        engine = AIEngine(StubModel(probs))

        with self.assertRaises(ValueError) as ctx:
            engine.get_move(board, 2, strategy="argmax")
        self.assertIn("棋盘已满", str(ctx.exception))

    # ---------- 预计算 probs ----------

    def test_precomputed_probs_skip_predict(self):
        """用例 10：传入 probs 时不调用 model.predict，且行为与不传一致。"""
        board = self._empty_board()
        probs_vec = self._make_probs([(100, 0.7), (101, 0.3)])
        stub = StubModel([0.0] * 225)  # 模型自身的概率故意设为全 0
        engine = AIEngine(stub)

        x, y = engine.get_move(board, 1, probs=probs_vec, strategy="argmax")

        self.assertEqual((x, y), (100 // 15, 100 % 15),
                         "应使用传入的 probs，而非模型自身的 predict 输出")
        self.assertEqual(stub.predict_call_count, 0,
                         "传入预计算 probs 时不应再调用 model.predict")


if __name__ == "__main__":
    unittest.main()
