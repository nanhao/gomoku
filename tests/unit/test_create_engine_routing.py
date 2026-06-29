"""create_engine 智能路由单元测试（意图 35：T35-05 ~ T35-10, T35-17）

测试 create_engine 根据模型类型自动创建对应引擎：
- T35-05: 单头模型创建 SinglePolicyAiEngine
- T35-06: 双头模型创建 PolicyValueAiEngine（需权重文件存在）
- T35-07: 未知模型名抛 ValueError
- T35-08: create_engine 返回的引擎可正常 get_move
- T35-09: create_engine 对双头模型的 GPU 迁移（CUDA 可用时）
- T35-10: ai/engine/__init__.py 正确导出 create_engine
- T35-17: 测试预测失败抛 ValueError
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import ModelPredict


# ---------- T35-05：单头模型创建 SinglePolicyAiEngine ----------

class TestT35_05_SinglePolicyFromCreateEngine(unittest.TestCase):
    """T35-05：create_engine("random") 返回 SinglePolicyAiEngine"""

    def test_create_engine_random_returns_single_policy(self):
        """create_engine("random") 返回 SinglePolicyAiEngine 实例"""
        from ai.engine import create_engine, SinglePolicyAiEngine

        engine = create_engine("random")
        self.assertIsInstance(engine, SinglePolicyAiEngine,
                              "create_engine('random') 应返回 SinglePolicyAiEngine")
        self.assertEqual(engine.model.name, "random")


# ---------- T35-06：双头模型创建 PolicyValueAiEngine ----------

class TestT35_06_PolicyValueFromCreateEngine(unittest.TestCase):
    """T35-06：create_engine("cnn_v7.1") 返回 PolicyValueAiEngine（需 gen-1 权重存在）"""

    @classmethod
    def setUpClass(cls):
        # 检查 cnn_v7 gen-1 的权重文件是否存在
        import os
        cnn_v7_models_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "ai", "models", "cnn_v7"
        )
        gen1_path = os.path.join(cnn_v7_models_dir, "gen1", "cnn_v7.pth")
        cls._gen1_exists = os.path.exists(gen1_path)
        if not cls._gen1_exists:
            # 尝试从 ai_data 加载
            pass  # 如果没有 gen1 权重，此测试 skip

    def setUp(self):
        if not self._gen1_exists:
            self.skipTest("cnn_v7 gen-1 权重不存在，跳过双头模型测试")

    def test_create_engine_cnn_v7_dot_1_returns_policy_value(self):
        """create_engine("cnn_v7.1") 返回 PolicyValueAiEngine 实例"""
        from ai.engine import create_engine, PolicyValueAiEngine

        engine = create_engine("cnn_v7.1")
        self.assertIsInstance(engine, PolicyValueAiEngine,
                              "create_engine('cnn_v7.1') 应返回 PolicyValueAiEngine")
        self.assertIsNotNone(engine.searcher,
                             "PolicyValueAiEngine 的 searcher 不应为 None")


# ---------- T35-07：未知模型名抛 ValueError ----------

class TestT35_07_CreateEngineUnknownModel(unittest.TestCase):
    """T35-07：create_engine("nonexistent_ai") 抛出 ValueError"""

    def test_create_engine_unknown_raises_value_error(self):
        """未知模型名应抛出 ValueError"""
        from ai.engine import create_engine

        with self.assertRaises(ValueError):
            create_engine("nonexistent_ai")


# ---------- T35-08：create_engine 返回的引擎可正常 get_move ----------

class TestT35_08_CreateEngineGetMove(unittest.TestCase):
    """T35-08：create_engine("random") 返回的引擎可正常 get_move"""

    def test_get_move_returns_valid_position(self):
        """空棋盘调用 get_move，返回 MoveDecision，位置在 0-14 范围内"""
        from ai.engine import create_engine, MoveDecision

        engine = create_engine("random")
        board = [[0] * 15 for _ in range(15)]
        md = engine.get_move(board, 1)

        self.assertIsInstance(md, MoveDecision,
                              "get_move 应返回 MoveDecision 实例")
        x, y = md.get_position()
        self.assertTrue(0 <= x < 15, f"x={x} 不在 0-14 范围内")
        self.assertTrue(0 <= y < 15, f"y={y} 不在 0-14 范围内")


# ---------- T35-09：create_engine 对双头模型的 GPU 迁移 ----------

class TestT35_09_CreateEngineGpuMigration(unittest.TestCase):
    """T35-09：create_engine 对双头模型的 GPU 迁移（CUDA 可用时）"""

    @classmethod
    def setUpClass(cls):
        import os
        cnn_v7_models_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "ai", "models", "cnn_v7"
        )
        gen1_path = os.path.join(cnn_v7_models_dir, "gen1", "cnn_v7.pth")
        cls._gen1_exists = os.path.exists(gen1_path)

    def setUp(self):
        if not self._gen1_exists:
            self.skipTest("cnn_v7 gen-1 权重不存在，跳过 GPU 迁移测试")

    def test_create_engine_cnn_v7_gpu_migration(self):
        """create_engine("cnn_v7.1") 创建的引擎中模型已在 GPU 上（CUDA 可用时）"""
        import torch
        from ai.engine import create_engine

        if not torch.cuda.is_available():
            self.skipTest("CUDA 不可用，跳过 GPU 迁移测试")

        engine = create_engine("cnn_v7.1")

        # 检查 searcher 中模型 network 所在设备
        device = next(engine.searcher.model.network.parameters()).device
        self.assertEqual(device.type, "cuda",
                         "CUDA 可用时，network 应位于 GPU 上")

        # get_move 应正常返回，无设备错误
        board = [[0] * 15 for _ in range(15)]
        md = engine.get_move(board, 1)
        x, y = md.get_position()
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)


# ---------- T35-10：ai/engine/__init__.py 正确导出 create_engine ----------

class TestT35_10_CreateEngineReExport(unittest.TestCase):
    """T35-10：ai/engine/__init__.py 正确导出 create_engine"""

    def test_create_engine_importable_from_ai_engine(self):
        """from ai.engine import create_engine 成功且 callable"""
        from ai.engine import create_engine
        self.assertTrue(callable(create_engine),
                        "create_engine 应可调用")

    def test_create_engine_importable_from_ai(self):
        """from ai import create_engine 成功且 callable"""
        from ai import create_engine
        self.assertTrue(callable(create_engine),
                        "create_engine 从 ai 包导出应可调用")


# ---------- T35-17：测试预测失败抛 ValueError ----------

class TestT35_17_CreateEnginePredictFailure(unittest.TestCase):
    """T35-17：create_engine 测试预测失败抛 ValueError"""

    def test_predict_failure_raises_value_error_mocked(self):
        """使用 mock 模拟 predict 失败场景，create_engine 应抛出 ValueError"""
        from unittest.mock import patch, MagicMock
        from ai.engine import create_engine

        # 构造一个会失败的 mock 模型
        mock_model = MagicMock()
        mock_model.network = None
        mock_model.predict.side_effect = RuntimeError("模拟预测失败")
        mock_model.move_to_device = MagicMock()

        # Patch registry.get_model 返回 mock 模型
        with patch('ai.registry.get_model', return_value=mock_model):
            with self.assertRaises(ValueError) as ctx:
                create_engine("failing_model")

            # 验证异常消息包含原始错误信息
            msg = str(ctx.exception)
            self.assertIn("预测", msg,
                          "异常消息应提及'预测'失败")

    def test_predict_model_load_failure_raises_value_error(self):
        """get_model 失败时 create_engine 应透传 ValueError"""
        from ai.engine import create_engine

        # unknown model triggers get_model ValueError
        with self.assertRaises(ValueError):
            create_engine("nonexistent_model_xyz")


if __name__ == "__main__":
    unittest.main()
