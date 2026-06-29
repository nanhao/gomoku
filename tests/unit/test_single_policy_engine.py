"""SinglePolicyAiEngine 单元测试（意图 34：T34-01 ~ T34-06）

测试 AIEngine 抽象性、SinglePolicyAiEngine 继承关系、
get_move 行为、after_move 空操作。
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import BaseModel, ModelPredict


# ---------- T34-01：AIEngine 为抽象类，不可直接实例化 ----------

class TestAIEngineAbstract(unittest.TestCase):
    """T34-01：AIEngine 抽象基类不可直接实例化"""

    def test_cannot_instantiate_ai_engine(self):
        """直接实例化 AIEngine 抛出 TypeError"""
        from ai.engine import AIEngine
        with self.assertRaises(TypeError):
            AIEngine()  # type: ignore[abstract]


# ---------- T34-02：SinglePolicyAiEngine 是 AIEngine 的子类 ----------

class TestSinglePolicyInheritance(unittest.TestCase):
    """T34-02：SinglePolicyAiEngine 继承 AIEngine"""

    def test_is_subclass_of_ai_engine(self):
        """SinglePolicyAiEngine 是 AIEngine 的子类"""
        from ai.engine import AIEngine, SinglePolicyAiEngine
        self.assertTrue(issubclass(SinglePolicyAiEngine, AIEngine))

    def test_is_instance_of_ai_engine(self):
        """SinglePolicyAiEngine 实例也是 AIEngine 实例"""
        from ai.engine import AIEngine, SinglePolicyAiEngine
        from ai.models.random.model import RandomModel

        engine = SinglePolicyAiEngine(RandomModel())
        self.assertIsInstance(engine, AIEngine)


# ---------- T34-03：get_move 行为（argmax 策略） ----------

class StubModel(BaseModel):
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
        raise NotImplementedError

    def load(self, model_path=None):
        raise NotImplementedError


class TestSinglePolicyGetMove(unittest.TestCase):
    """T34-03 ~ T34-05：get_move 行为测试"""

    def setUp(self):
        from ai.engine import SinglePolicyAiEngine

        # 构造唯一概率最大在 (7,7) = idx 112
        probs = [0.0] * 225
        probs[112] = 0.9
        probs[0] = 0.05
        probs[1] = 0.05
        self.engine = SinglePolicyAiEngine(StubModel(probs))
        self.board = [[0] * 15 for _ in range(15)]

    def test_t34_03_get_move_returns_move_decision(self):
        """T34-03：get_move 返回 MoveDecision 实例"""
        from ai.engine import MoveDecision
        md = self.engine.get_move(self.board, 1)
        self.assertIsInstance(md, MoveDecision)

    def test_t34_04_get_move_argmax_picks_max_prob(self):
        """T34-04：argmax 策略返回概率最大的位置"""
        md = self.engine.get_move(self.board, 1, strategy="argmax")
        self.assertEqual(md.get_position(), (7, 7))

    def test_t34_05_get_move_returns_valid_empty_position(self):
        """T34-05：返回的位置一定是空位"""
        from ai.engine import SinglePolicyAiEngine

        # 填满除 (0,0) 外的所有位置
        board = [[1] * 15 for _ in range(15)]
        board[0][0] = 0

        probs = [0.0] * 225
        probs[0] = 0.5
        engine = SinglePolicyAiEngine(StubModel(probs))

        md = engine.get_move(board, 2)
        self.assertEqual(md.get_position(), (0, 0))


# ---------- T34-06：after_move 为空操作 ----------

class TestSinglePolicyAfterMove(unittest.TestCase):
    """T34-06：after_move 为空操作"""

    def setUp(self):
        from ai.engine import SinglePolicyAiEngine
        from ai.models.random.model import RandomModel
        self.engine = SinglePolicyAiEngine(RandomModel())

    def test_t34_06_after_move_is_noop(self):
        """T34-06：after_move 不抛异常，不改变引擎状态"""
        # 多次调用不应抛异常（使用 (x, y) 坐标）
        for x, y in [(0, 0), (7, 7), (14, 14)]:
            self.engine.after_move(x, y)

        # 调用 after_move 后 get_move 仍正常工作
        board = [[0] * 15 for _ in range(15)]
        md = self.engine.get_move(board, 1)
        x, y = md.get_position()
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)


if __name__ == "__main__":
    unittest.main()
