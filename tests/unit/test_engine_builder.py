"""AIEngineBuilder 单元测试（意图 34：T34-17 ~ T34-19）

测试 build_single_policy、build_policy_value、create_engine 委托。
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestBuildSinglePolicy(unittest.TestCase):
    """T34-17：build_single_policy 工厂方法"""

    def test_t34_17_build_single_policy_random(self):
        """T34-17：build_single_policy("random") 返回 SinglePolicyAiEngine 实例"""
        from ai.engine import AIEngineBuilder, SinglePolicyAiEngine

        engine = AIEngineBuilder.build_single_policy("random")
        self.assertIsInstance(engine, SinglePolicyAiEngine)
        self.assertEqual(engine.model.name, "random")

    def test_build_single_policy_with_temperature(self):
        """build_single_policy 可传入 temperature 参数"""
        from ai.engine import AIEngineBuilder

        engine = AIEngineBuilder.build_single_policy("random", temperature=0.5)
        self.assertEqual(engine.temperature, 0.5)

    def test_build_single_policy_unknown_raises(self):
        """build_single_policy 对未知模型名抛 ValueError"""
        from ai.engine import AIEngineBuilder

        with self.assertRaises(ValueError):
            AIEngineBuilder.build_single_policy("nonexistent_model")


class TestBuildPolicyValue(unittest.TestCase):
    """T34-18：build_policy_value 工厂方法"""

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

    def test_t34_18_build_policy_value_returns_engine(self):
        """T34-18：build_policy_value 返回 PolicyValueAiEngine 实例"""
        from ai.engine import AIEngineBuilder, PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel

        model = MockModel()
        engine = AIEngineBuilder.build_policy_value(
            model,
            num_simulations=10,
            temperature=0.5,
            temp_steps=5,
        )
        self.assertIsInstance(engine, PolicyValueAiEngine)
        self.assertEqual(engine.temperature, 0.5)
        self.assertEqual(engine.temp_steps, 5)


class TestCreateEngine(unittest.TestCase):
    """T34-19：create_engine 委托 build_single_policy"""

    def test_t34_19_create_engine_delegates_to_builder(self):
        """T34-19：create_engine("random") 委托 AIEngineBuilder.build_single_policy"""
        from ai.engine import create_engine, SinglePolicyAiEngine

        engine = create_engine("random")
        self.assertIsInstance(engine, SinglePolicyAiEngine)
        self.assertEqual(engine.model.name, "random")

    def test_create_engine_unknown_raises(self):
        """create_engine 对未知模型名抛 ValueError"""
        from ai.engine import create_engine

        with self.assertRaises(ValueError):
            create_engine("nonexistent_model")


if __name__ == "__main__":
    unittest.main()
