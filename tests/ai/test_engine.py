"""AI 引擎测试

测试 AIEngine 类的所有公共方法。
"""

import sys
import os
import random
import unittest
import pytest

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import BaseModel, ModelPredict
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
        return ModelPredict(policy=list(self._probs))

    @property
    def name(self):
        return "stub"

    def save(self, model_path=None):
        raise NotImplementedError("StubModel 不支持保存")

    def load(self, model_path=None):
        raise NotImplementedError("StubModel 不支持加载")


def _empty_board():
    """返回 15×15 的空棋盘。"""
    return [[0] * 15 for _ in range(15)]


def _make_probs(pairs, total=225):
    """根据 (index, prob) 对构造 225 维概率向量，其余位置为 0。"""
    probs = [0.0] * total
    for idx, p in pairs:
        probs[idx] = p
    return probs


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
        decision = self.engine.get_move(board, 1)
        x, y = decision.get_position()
        
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
        
        decision = self.engine.get_move(board, 2)
        x, y = decision.get_position()
        
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
        
        decision = self.engine.get_move(board, 2)
        x, y = decision.get_position()
        
        # 必须是唯一的空位
        self.assertEqual(x, 14)
        self.assertEqual(y, 14)


class TestCreateEngine(unittest.TestCase):
    """测试 create_engine 函数"""

    @pytest.mark.slow
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


class TestGetMoveStrategy(unittest.TestCase):
    """落子策略测试：argmax（默认） / sampling（未实现） / 未知策略。

    对应设计文档 design/engine_move_strategy_design.md 中的主流程测试用例 1-10。
    """

    # ---------- argmax ----------

    def test_argmax_picks_unique_max(self):
        """用例 1：唯一最大值时必返回该位置。"""
        board = _empty_board()
        # (7,7) 索引 = 7*15+7 = 112，概率明显最大
        probs = _make_probs([(112, 0.9), (0, 0.05), (1, 0.05)])
        engine = AIEngine(StubModel(probs))

        for _ in range(20):
            decision = engine.get_move(board, 1, strategy="argmax")
            x, y = decision.get_position()
            self.assertEqual((x, y), (7, 7))

    def test_argmax_ties_broken_randomly(self):
        """用例 2：多个位置概率并列最大时，从并列候选中随机挑一个。"""
        board = _empty_board()
        # (0,0)=idx 0 与 (0,1)=idx 1 并列 0.4，(0,2)=idx 2 为 0.2
        probs = _make_probs([(0, 0.4), (1, 0.4), (2, 0.2)])
        engine = AIEngine(StubModel(probs))

        # 固定随机种子，避免偶发失败；仍要求覆盖两个并列候选
        random.seed(20260501)
        seen = set()
        for _ in range(200):
            seen.add(engine.get_move(board, 1, strategy="argmax").get_position())

        self.assertIn((0, 0), seen)
        self.assertIn((0, 1), seen)
        # (0,2) 概率更低，不应被选中
        self.assertNotIn((0, 2), seen)

    def test_argmax_skips_occupied_max(self):
        """用例 3：概率最大的位置已落子时，返回空位中概率最大的。"""
        board = _empty_board()
        board[7][7] = 1  # 占住 idx 112
        # idx 112 概率最大，但已占用；idx 0 次大且为空
        probs = _make_probs([(112, 0.9), (0, 0.5), (1, 0.1)])
        engine = AIEngine(StubModel(probs))

        decision = engine.get_move(board, 2, strategy="argmax")
        x, y = decision.get_position()
        self.assertEqual((x, y), (0, 0))

    def test_argmax_all_zero_probs_falls_back_to_uniform_pick(self):
        """用例 4：所有空位概率都为 0 时仍应挑一个空位，不抛异常。"""
        board = _empty_board()
        # 模型把所有概率压在 (0,0)，且 (0,0) 已落子 → 空位全 0
        board[0][0] = 1
        probs = _make_probs([(0, 1.0)])
        engine = AIEngine(StubModel(probs))

        decision = engine.get_move(board, 2, strategy="argmax")
        x, y = decision.get_position()
        self.assertEqual(board[x][y], 0, "回退时必须选空位")
        self.assertNotEqual((x, y), (0, 0), "不能选已占用的位置")

    # ---------- 默认行为 ----------

    def test_default_strategy_is_argmax(self):
        """用例 5：不传 strategy 时，行为等价于 argmax。"""
        board = _empty_board()
        probs = _make_probs([(50, 0.8), (51, 0.2)])
        engine = AIEngine(StubModel(probs))

        # 多次调用都应稳定返回概率最大的位置
        for _ in range(20):
            self.assertEqual(engine.get_move(board, 1).get_position(), (50 // 15, 50 % 15))

    # ---------- sampling 分支（已由 ai.move_sampler 实现）----------
    # 见 TestGetMoveSamplingStrategy 的用例 17-22

    # ---------- 未知策略 ----------

    def test_unknown_strategy_raises_value_error(self):
        """用例 8：未知 strategy 值应抛 ValueError。"""
        board = _empty_board()
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
        board = _empty_board()
        probs_vec = _make_probs([(100, 0.7), (101, 0.3)])
        stub = StubModel([0.0] * 225)  # 模型自身的概率故意设为全 0
        engine = AIEngine(stub)

        decision = engine.get_move(board, 1, probs=probs_vec, strategy="argmax")
        x, y = decision.get_position()

        self.assertEqual((x, y), (100 // 15, 100 % 15),
                         "应使用传入的 probs，而非模型自身的 predict 输出")
        self.assertEqual(stub.predict_call_count, 0,
                         "传入预计算 probs 时不应再调用 model.predict")


class TestGetMoveSamplingStrategy(unittest.TestCase):
    """sampling 策略测试（对应设计文档 move_sampler_design.md 用例 17-22）。

    strategy="sampling" 不再抛 NotImplementedError，而是委托给
    ai.move_sampler.temperature_sample 完成温度采样。
    """

    def test_sampling_strategy_no_longer_raises_not_implemented(self):
        """用例 17：sampling 策略不再抛 NotImplementedError。"""
        board = _empty_board()
        probs = [1.0 / 225] * 225
        engine = AIEngine(StubModel(probs))

        # 不传 temperature → 使用默认值 1.0；不抛异常即视为通过
        decision = engine.get_move(board, 1, strategy="sampling")
        x, y = decision.get_position()
        self.assertTrue(0 <= x < 15 and 0 <= y < 15)

    def test_sampling_strategy_returns_valid_position(self):
        """用例 18：sampling 返回的位置必须在合法空位集合中。"""
        board = _empty_board()
        # 占住若干位置，确保 sampling 跳过
        board[5][5] = 1
        board[6][6] = 2
        # 构造非均匀分布，测试覆盖
        probs = _make_probs([(0, 0.5), (1, 0.3), (2, 0.2)])
        engine = AIEngine(StubModel(probs))

        rng = random.Random(2026)
        for _ in range(100):
            decision = engine.get_move(
                board, 1, strategy="sampling", temperature=1.0, rng=rng
            )
            x, y = decision.get_position()
            self.assertEqual(board[x][y], 0, "sampling 必须返回空位")

    def test_sampling_strategy_small_T_equals_argmax(self):
        """用例 19：strategy=sampling, T 极小 → 与 argmax 等价。"""
        board = _empty_board()
        probs = _make_probs([(50, 0.8), (51, 0.15), (52, 0.05)])
        engine = AIEngine(StubModel(probs))

        rng = random.Random(7)
        expected = (50 // 15, 50 % 15)
        for _ in range(50):
            pos = engine.get_move(
                board, 1, strategy="sampling", temperature=0.01, rng=rng
            ).get_position()
            self.assertEqual(pos, expected)

    def test_sampling_strategy_respects_rng_seed(self):
        """用例 20：传入同 seed 的 rng → 两次调用序列一致。"""
        board = _empty_board()
        probs = _make_probs(
            [(0, 0.25), (1, 0.20), (2, 0.20), (3, 0.20), (4, 0.15)]
        )
        engine = AIEngine(StubModel(probs))

        rng1 = random.Random(2026)
        rng2 = random.Random(2026)
        seq1 = [
            engine.get_move(board, 1, strategy="sampling",
                            temperature=1.2, rng=rng1).get_position()
            for _ in range(30)
        ]
        seq2 = [
            engine.get_move(board, 1, strategy="sampling",
                            temperature=1.2, rng=rng2).get_position()
            for _ in range(30)
        ]
        self.assertEqual(seq1, seq2)

    def test_sampling_strategy_invalid_temperature_raises(self):
        """用例 21：T <= 0 → ValueError。"""
        board = _empty_board()
        probs = [1.0 / 225] * 225
        engine = AIEngine(StubModel(probs))

        with self.assertRaises(ValueError):
            engine.get_move(board, 1, strategy="sampling", temperature=0.0)
        with self.assertRaises(ValueError):
            engine.get_move(board, 1, strategy="sampling", temperature=-1.0)

    def test_argmax_path_unchanged_by_new_params(self):
        """用例 22：strategy=argmax 默认值下，新参数 temperature/top_k/rng 不影响行为。"""
        board = _empty_board()
        probs = _make_probs([(100, 0.9), (101, 0.1)])
        engine = AIEngine(StubModel(probs))

        # 既不传 strategy 也不传新参数
        pos1 = engine.get_move(board, 1).get_position()
        # 传默认 strategy=argmax
        pos2 = engine.get_move(board, 1, strategy="argmax").get_position()
        self.assertEqual(pos1, (100 // 15, 100 % 15))
        self.assertEqual(pos2, (100 // 15, 100 % 15))


if __name__ == "__main__":
    unittest.main()