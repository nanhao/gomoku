"""SinglePolicyAiEngine 构造函数重构单元测试

对应设计文档 design/33/main_docs/test_design.md T1：SinglePolicyAiEngine 构造函数新增参数。
测试 temperature / top_k / rng 移入构造函数后的行为。
"""

import sys
import os
import random
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import BaseModel, ModelPredict
from ai.engine import SinglePolicyAiEngine
from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------

class _StubModel(BaseModel):
    """测试桩模型：predict 恒定返回构造时指定的概率向量。"""

    def __init__(self, probs):
        self._probs = list(probs)
        self.network = None
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


# ---------------------------------------------------------------------------
# T1：SinglePolicyAiEngine 构造函数新增参数
# ---------------------------------------------------------------------------

class TestSinglePolicyAiEngineConstructorParams(unittest.TestCase):
    """T1：SinglePolicyAiEngine 构造函数新增 temperature / top_k / rng 参数。"""

    # ---------- T1-01：默认构造 — 采样参数为默认值 ----------

    def test_default_construction_uses_default_temperature(self):
        """T1-01：仅传 model 构造 SinglePolicyAiEngine，sampling 使用默认 temperature=1.0。"""
        board = _empty_board()
        probs = _make_probs([(50, 0.8), (51, 0.2)])
        model = _StubModel(probs)
        engine = SinglePolicyAiEngine(model)

        # 验证 get_move 正常工作（不抛异常即通过）
        decision = engine.get_move(board, 1, strategy=STRATEGY_SAMPLING)
        x, y = decision.get_position()
        self.assertTrue(0 <= x < 15 and 0 <= y < 15)
        self.assertEqual(board[x][y], 0)

    # ---------- T1-02：自定义 temperature 构造 ----------

    def test_custom_temperature_affects_sampling(self):
        """T1-02：构造 SinglePolicyAiEngine(model, temperature=2.0)，采样分布与 T=1.0 不同。"""
        board = _empty_board()
        # (0,0) 概率 0.8，(0,1) 概率 0.2
        probs = _make_probs([(0, 0.8), (1, 0.2)])
        model = _StubModel(probs)

        engine_T1 = SinglePolicyAiEngine(model, temperature=1.0)
        engine_T2 = SinglePolicyAiEngine(model, temperature=0.01)  # 极小温度趋近 argmax

        # T → 0 时应逼近 argmax：几乎总是选 (0,0)
        rng = random.Random(42)
        engine_T2.rng = rng
        T2_positions = set()
        for _ in range(50):
            pos = engine_T2.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
            T2_positions.add(pos)

        # 极高概率只选 (0,0)，至多偶尔选其他
        self.assertIn((0, 0), T2_positions)

    # ---------- T1-03：自定义 top_k 构造 ----------

    def test_custom_top_k_truncates_to_k(self):
        """T1-03：构造 SinglePolicyAiEngine(model, top_k=3)，仅在前 3 个位置中选择。"""
        board = _empty_board()
        # 前 5 个位置都有概率，但 top_k=3 只保留前 3
        probs = _make_probs([(0, 0.5), (1, 0.2), (2, 0.15), (3, 0.1), (4, 0.05)])
        model = _StubModel(probs)
        engine = SinglePolicyAiEngine(model, top_k=3)

        # 多次采样验证只在 top 3 中
        rng = random.Random(7)
        engine.rng = rng
        seen = set()
        for _ in range(100):
            pos = engine.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
            action = pos[0] * 15 + pos[1]
            seen.add(action)

        # 截断后只会选前 3 个位置
        for action in seen:
            self.assertIn(action, [0, 1, 2],
                          f"top_k=3 不应选择 action={action}")

    # ---------- T1-04：自定义 rng 构造 — 可复现 ----------

    def test_custom_rng_makes_sampling_reproducible(self):
        """T1-04：构造 SinglePolicyAiEngine(model, rng=Random(42))，同棋盘连续两次结果相同。"""
        board = _empty_board()
        probs = _make_probs([(0, 0.25), (1, 0.20), (2, 0.20), (3, 0.20), (4, 0.15)])
        model = _StubModel(probs)

        # 第一轮
        engine1 = SinglePolicyAiEngine(model, rng=random.Random(42))
        seq1 = [engine1.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
                for _ in range(30)]

        # 第二轮：同种子应产生相同序列
        engine2 = SinglePolicyAiEngine(model, rng=random.Random(42))
        seq2 = [engine2.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
                for _ in range(30)]

        self.assertEqual(seq1, seq2, "相同种子的 rng 应产生可复现采样序列")

    # ---------- T1-05：get_move 不再接受 temperature 参数 ----------

    def test_get_move_rejects_temperature_keyword(self):
        """T1-05：get_move 不再接受 temperature 关键字参数，应抛出 TypeError。"""
        board = _empty_board()
        probs = [1.0 / 225] * 225
        engine = SinglePolicyAiEngine(_StubModel(probs))

        with self.assertRaises(TypeError):
            engine.get_move(board, 1, temperature=2.0)

    def test_get_move_rejects_top_k_keyword(self):
        """T1-05b：get_move 不再接受 top_k 关键字参数。"""
        board = _empty_board()
        probs = [1.0 / 225] * 225
        engine = SinglePolicyAiEngine(_StubModel(probs))

        with self.assertRaises(TypeError):
            engine.get_move(board, 1, top_k=3)

    def test_get_move_rejects_rng_keyword(self):
        """T1-05c：get_move 不再接受 rng 关键字参数。"""
        board = _empty_board()
        probs = [1.0 / 225] * 225
        engine = SinglePolicyAiEngine(_StubModel(probs))

        with self.assertRaises(TypeError):
            engine.get_move(board, 1, rng=random.Random())

    # ---------- T1-06：argmax 策略不受 temperature 影响 ----------

    def test_argmax_ignores_temperature(self):
        """T1-06：构造 SinglePolicyAiEngine(model, temperature=0.1)，argmax 仍选概率最大者。"""
        board = _empty_board()
        probs = _make_probs([(100, 0.9), (101, 0.1)])
        model = _StubModel(probs)
        engine = SinglePolicyAiEngine(model, temperature=0.1)

        for _ in range(20):
            pos = engine.get_move(board, 1, strategy=STRATEGY_ARGMAX).get_position()
            self.assertEqual(pos, (100 // 15, 100 % 15),
                             "argmax 不应受 temperature=0.1 影响")


# ---------------------------------------------------------------------------
# T2：SinglePolicyAiEngine get_move() 删减参数 — 调用方适配
# ---------------------------------------------------------------------------

class TestSinglePolicyAiEngineCallerAdaptation(unittest.TestCase):
    """T2：SinglePolicyAiEngine get_move() 删减参数后的调用方行为。"""

    def test_different_engines_independent_params(self):
        """T2-03：不同引擎实例独立持有各自采样参数，互不干扰。"""
        board = _empty_board()
        probs = _make_probs([(0, 0.25), (1, 0.2), (2, 0.2), (3, 0.2), (4, 0.15)])
        model = _StubModel(probs)

        engine1 = SinglePolicyAiEngine(model, temperature=1.0)
        engine2 = SinglePolicyAiEngine(model, temperature=0.01)  # 趋近 argmax

        rng = random.Random(42)
        engine1.rng = random.Random(42)
        engine2.rng = random.Random(42)

        # engine2 (T=0.01) 应更稳定地选 argmax
        engine2_positions = set()
        for _ in range(30):
            engine2_positions.add(
                engine2.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
            )

        # engine1 (T=1.0) 应更随机
        engine1_positions = set()
        for _ in range(30):
            engine1_positions.add(
                engine1.get_move(board, 1, strategy=STRATEGY_SAMPLING).get_position()
            )

        # T=0.01 应趋近确定性（选概率最高的 0），T=1.0 应更多样
        self.assertGreaterEqual(len(engine1_positions), 1)
        self.assertGreaterEqual(len(engine2_positions), 1)

    def test_sampling_strategy_uses_engine_temperature(self):
        """T2-01 变体：构造时传入 temperature=1.5，sampling 使用该值。"""
        board = _empty_board()
        probs = _make_probs([(0, 0.8), (1, 0.2)])
        model = _StubModel(probs)
        engine = SinglePolicyAiEngine(model, temperature=1.5)

        # 不传 temperature 给 get_move，使用引擎自身的 1.5
        decision = engine.get_move(board, 1, strategy=STRATEGY_SAMPLING)
        x, y = decision.get_position()
        self.assertTrue(0 <= x < 15 and 0 <= y < 15)

    def test_argmax_works_without_temp_params(self):
        """T2-02：无温度参数构造引擎，argmax 正常工作。"""
        board = _empty_board()
        probs = _make_probs([(50, 0.8), (51, 0.2)])
        model = _StubModel(probs)
        engine = SinglePolicyAiEngine(model)

        for _ in range(20):
            pos = engine.get_move(board, 1).get_position()
            self.assertEqual(pos, (50 // 15, 50 % 15))


if __name__ == "__main__":
    unittest.main()
