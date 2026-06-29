"""after_move(x, y) 签名变更单元测试（意图 35：T35-01 ~ T35-04, T35-18）

测试 after_move 接受 (x, y) 坐标参数：
- T35-01: SinglePolicyAiEngine.after_move(x, y) 正常调用
- T35-02: PolicyValueAiEngine.after_move(x, y) 成功推进
- T35-03: PolicyValueAiEngine.after_move 无效坐标抛 RuntimeError（含消息验证）
- T35-04: PolicyValueAiEngine.after_move 终局节点不抛异常
- T35-18: SinglePolicyAiEngine.after_move 无副作用验证
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from ai.base_model import BaseModel, ModelPredict


# ---------- 辅助桩 ----------

class StubModel(BaseModel):
    """测试桩模型：固定概率输出。"""

    def __init__(self, probs=None):
        self._probs = list(probs) if probs else [1.0 / 225] * 225
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


# ---------- T35-01：SinglePolicyAiEngine.after_move(x, y) 正常调用 ----------

class TestT35_01_SinglePolicyAfterMoveXY(unittest.TestCase):
    """T35-01：SinglePolicyAiEngine.after_move(x, y) 接受坐标参数"""

    def setUp(self):
        from ai.engine import SinglePolicyAiEngine
        from ai.models.random.model import RandomModel
        self.engine = SinglePolicyAiEngine(RandomModel())

    def test_after_move_with_xy_coordinates(self):
        """依次调用 after_move(0,0)、after_move(7,7)、after_move(14,14) 不抛异常"""
        # 必须使用 (x, y) 坐标，不再接受扁平 action
        self.engine.after_move(0, 0)
        self.engine.after_move(7, 7)
        self.engine.after_move(14, 14)
        # 之后 get_move 仍正常工作
        board = [[0] * 15 for _ in range(15)]
        md = self.engine.get_move(board, 1)
        x, y = md.get_position()
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)


# ---------- T35-02：PolicyValueAiEngine.after_move(x, y) 成功推进 ----------

class TestT35_02_PolicyValueAfterMoveSuccess(unittest.TestCase):
    """T35-02：PolicyValueAiEngine.after_move(x, y) 成功推进搜索树"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_after_move_with_xy_advances_root(self):
        """对空棋盘搜索后，用 get_move 返回的 (x,y) 调用 after_move 成功"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher)

        # 空棋盘搜索
        board_2d = [[0] * 15 for _ in range(15)]
        np.random.seed(42)
        md = engine.get_move(board_2d, 1, strategy="argmax")
        x, y = md.get_position()

        # 推进不应抛异常
        engine.after_move(x, y)

        # 步数应递增
        self.assertEqual(engine._step_count, 1)


# ---------- T35-03：PolicyValueAiEngine.after_move 无效坐标抛异常（含消息验证）----------

class TestT35_03_PolicyValueAfterMoveInvalid(unittest.TestCase):
    """T35-03：PolicyValueAiEngine.after_move 无效坐标抛 RuntimeError（含消息验证）"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_invalid_coordinate_raises_with_message(self):
        """after_move(99, 99) 抛出 RuntimeError，消息同时包含 (99, 99) 和 1584"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=10,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher)

        # 空棋盘搜索
        board_2d = [[0] * 15 for _ in range(15)]
        np.random.seed(42)
        engine.get_move(board_2d, 1, strategy="argmax")

        # 坐标 (99, 99) → action = 99 * 15 + 99 = 1584
        with self.assertRaises(RuntimeError) as ctx:
            engine.after_move(99, 99)

        msg = str(ctx.exception)
        self.assertIn("(99, 99)", msg,
                      "异常消息应包含坐标 (99, 99)")
        self.assertIn("1584", msg,
                      "异常消息应包含 action 1584")


# ---------- T35-04：PolicyValueAiEngine.after_move 终局节点不抛异常 ----------

class TestT35_04_PolicyValueAfterMoveTerminal(unittest.TestCase):
    """T35-04：PolicyValueAiEngine.after_move 终局节点不抛异常"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_terminal_node_after_move_does_not_raise(self):
        """必胜棋局：黑方 (7,0)-(7,3) 四子，after_move(7,4) 不抛异常"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel

        # 构造必胜棋局：黑方 (7,0)-(7,3) 四子，下一步 (7,4) 获胜
        board_flat = [0] * 225
        for y in range(4):
            board_flat[7 * 15 + y] = 1

        win_action = 7 * 15 + 4

        # 偏向必胜落子的策略
        policy = [0.001] * 225
        policy[win_action] = 10.0
        total = sum(policy)
        policy = [p / total for p in policy]

        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher, temperature=0.0)  # 贪婪

        board_2d = [[board_flat[i * 15 + j] for j in range(15)] for i in range(15)]

        np.random.seed(42)
        md = engine.get_move(board_2d, 1, strategy="argmax")
        x, y = md.get_position()

        # 终局节点推进不应抛异常
        try:
            engine.after_move(x, y)
        except RuntimeError as e:
            self.fail(f"after_move({x}, {y}) on terminal node should not raise: {e}")


# ---------- T35-18：SinglePolicyAiEngine.after_move 无副作用验证 ----------

class TestT35_18_SinglePolicyAfterMoveNoSideEffect(unittest.TestCase):
    """T35-18：SinglePolicyAiEngine.after_move 无副作用验证"""

    def setUp(self):
        from ai.engine import SinglePolicyAiEngine
        from ai.models.random.model import RandomModel
        self.engine = SinglePolicyAiEngine(RandomModel())

    def test_after_move_does_not_change_engine_state(self):
        """调用 after_move(0,0) 后，两次 get_move 结果一致（无副作用）"""
        # 先调用一次 after_move
        self.engine.after_move(0, 0)

        # 两次 get_move 应在同一棋局产生相同结果
        board = [[0] * 15 for _ in range(15)]
        # 用 argmax 策略确保确定性（random 模型空棋盘所有位置概率相同，
        # 但 argmax 会在全部空位中随机打破平局，无法保证两次一致。
        # 改用桩模型确保确定性）
        pass  # 此测试针对随机模型不适用——random 的 argmax 本身就是随机的

    def test_after_move_does_not_change_stub_engine_state(self):
        """使用确定性桩模型，after_move 不改变 get_move 结果"""
        from ai.engine import SinglePolicyAiEngine

        # 唯一最大值位于 (7,7)
        probs = [0.0] * 225
        probs[7 * 15 + 7] = 0.9
        probs[0] = 0.05
        probs[1] = 0.05
        engine = SinglePolicyAiEngine(StubModel(probs))

        board = [[0] * 15 for _ in range(15)]

        # 第一次 get_move（argmax 确定性返回 (7,7)）
        md1 = engine.get_move(board, 1, strategy="argmax")
        self.assertEqual(md1.get_position(), (7, 7))

        # 调用 after_move
        engine.after_move(0, 0)

        # 第二次 get_move 结果应与第一次一致
        md2 = engine.get_move(board, 1, strategy="argmax")
        self.assertEqual(md2.get_position(), (7, 7),
                         "after_move 不应改变引擎内部状态")


if __name__ == "__main__":
    unittest.main()
