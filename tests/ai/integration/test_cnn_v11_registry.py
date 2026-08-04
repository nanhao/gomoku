"""CNN v11 注册表集成测试

验证 registry 正确路由 cnn_v11 模型和训练器。

测试覆盖 TC4.1 ~ TC4.5（设计文档 test_cases.md 四）。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV11Registry(unittest.TestCase):
    """Registry 集成测试"""

    @classmethod
    def setUpClass(cls):
        """创建 gen1 权重文件供测试使用"""
        from ai.registry import MODEL_REGISTRY

        # 检查注册表是否包含 cnn_v11
        assert "cnn_v11" in MODEL_REGISTRY, "cnn_v11 应在 MODEL_REGISTRY 中"

        # 创建临时目录和 gen1 权重
        import tempfile
        cls._tmp_dir = tempfile.mkdtemp()

        from ai.models.cnn_v11.model import CNNV11Model
        # 需要创建在 src/ai/models/cnn_v11/gen1/ 下
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        _model_dir = os.path.join(_root, "src", "ai", "models", "cnn_v11")
        gen1_dir = os.path.join(_model_dir, "gen1")
        os.makedirs(gen1_dir, exist_ok=True)
        pth_path = os.path.join(gen1_dir, "cnn_v11.pth")
        if not os.path.exists(pth_path):
            tmp_model = CNNV11Model(fresh=True)
            tmp_model.save(pth_path)
        cls._saved_gen1_dir = gen1_dir

    @classmethod
    def tearDownClass(cls):
        """测试结束后清理 gen1 目录"""
        if hasattr(cls, '_saved_gen1_dir'):
            import shutil
            shutil.rmtree(cls._saved_gen1_dir, ignore_errors=True)
        if hasattr(cls, '_tmp_dir'):
            import shutil
            shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    # ---------- TC4.1 get_model — 加载 cnn_v11 模型 ----------

    def test_get_model(self):
        """get_model('cnn_v11.1') 返回 CNNV11Model"""
        from ai.registry import get_model

        model = get_model("cnn_v11.1")
        self.assertEqual(type(model).__name__, "CNNV11Model")
        self.assertEqual(model.name, "cnn_v11")
        self.assertEqual(model.gen, 1)

    # ---------- TC4.2 get_model — 未指定代数报错 ----------

    def test_get_model_no_gen_raises(self):
        """多代模型未指定代数抛出 ValueError"""
        from ai.registry import get_model

        with self.assertRaises(ValueError):
            get_model("cnn_v11")

    # ---------- TC4.3 get_trainer — 加载训练器 ----------

    def test_get_trainer(self):
        """get_trainer('cnn_v11.1') 返回 CNNV11Trainer"""
        from ai.registry import get_trainer

        trainer = get_trainer(
            "cnn_v11.1", device=torch.device("cpu"))
        self.assertEqual(type(trainer).__name__, "CNNV11Trainer")
        self.assertEqual(trainer.model_name, "cnn_v11")

    # ---------- TC4.4 list_models_with_full_name — 包含 cnn_v11 ----------

    def test_list_models_includes_cnn_v11(self):
        """list_models_with_full_name 包含 cnn_v11.1"""
        from ai.registry import list_models_with_full_name

        models = list_models_with_full_name()
        self.assertIn("cnn_v11.1", models,
                      f"'cnn_v11.1' 不在 {models} 中")

    # ---------- TC4.5 create_engine — 路由到 PolicyValueAiEngine ----------

    def test_create_engine(self):
        """create_engine('cnn_v11.1') 返回 PolicyValueAiEngine"""
        from ai.engine.builder import create_engine

        engine = create_engine("cnn_v11.1")
        type_name = type(engine).__name__
        self.assertIn(
            type_name, ["PolicyValueAiEngine", "AiEngine"],
            f"引擎类型应为 PolicyValueAiEngine，实际 {type_name}")


if __name__ == '__main__':
    unittest.main()
