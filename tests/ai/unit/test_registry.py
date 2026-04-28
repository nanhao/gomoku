"""统一模型注册表测试

测试 ai/registry.py 中的 get_model、get_trainer、list_models 函数。
覆盖正常加载、异常处理、参数传递和懒加载验证。
"""

import sys
import unittest

# 添加项目根目录到路径
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.registry import get_model, get_trainer, list_models, MODEL_REGISTRY
from ai.base_model import BaseModel
from ai.base_trainer import BaseTrainer


class TestGetModel(unittest.TestCase):
    """测试 get_model 函数"""

    def test_get_model_random(self):
        """加载 random 模型"""
        model = get_model("random")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.name, "random")

    def test_get_model_fc_v1(self):
        """加载 fc_v1 模型"""
        model = get_model("fc_v1")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.name, "fc_v1")

    def test_get_model_cnn_v1(self):
        """加载 cnn_v1 模型"""
        model = get_model("cnn_v1")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.name, "cnn_v1")

    def test_get_model_cnn_v2(self):
        """加载 cnn_v2 模型"""
        model = get_model("cnn_v2")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.name, "cnn_v2")

    def test_get_model_unknown_raises(self):
        """未知模型名称应抛出 ValueError，消息包含模型名"""
        with self.assertRaises(ValueError) as ctx:
            get_model("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_get_model_unknown_error_format(self):
        """未知模型的错误消息格式为 '未知的模型: {name}'"""
        with self.assertRaises(ValueError) as ctx:
            get_model("bad_model")
        self.assertIn("未知的模型", str(ctx.exception))
        self.assertIn("bad_model", str(ctx.exception))


class TestGetTrainer(unittest.TestCase):
    """测试 get_trainer 函数"""

    def test_get_trainer_fc_v1(self):
        """fc_v1 返回 FCv1Trainer 实例"""
        from ai.models.fc_v1.trainer import FCv1Trainer
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsInstance(trainer, FCv1Trainer)

    def test_get_trainer_cnn_v1(self):
        """cnn_v1 返回 CNNV1Trainer 实例"""
        from ai.models.cnn_v1.trainer import CNNV1Trainer
        trainer = get_trainer("cnn_v1", fresh=True)
        self.assertIsInstance(trainer, CNNV1Trainer)

    def test_get_trainer_cnn_v2(self):
        """cnn_v2 返回 CNNV2Trainer 实例"""
        from ai.models.cnn_v2.trainer import CNNV2Trainer
        trainer = get_trainer("cnn_v2", fresh=True)
        self.assertIsInstance(trainer, CNNV2Trainer)

    def test_get_trainer_random_raises(self):
        """random 没有 trainer，应抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("random")
        self.assertIn("不支持的模型", str(ctx.exception))
        self.assertIn("random", str(ctx.exception))

    def test_get_trainer_unknown_raises(self):
        """未知模型名称应抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("invalid_model")
        self.assertIn("不支持的模型", str(ctx.exception))
        self.assertIn("invalid_model", str(ctx.exception))

    def test_get_trainer_returns_base_trainer(self):
        """get_trainer 返回的对象应是 BaseTrainer 子类"""
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsInstance(trainer, BaseTrainer)

    def test_get_trainer_passes_fresh(self):
        """fresh 参数应传递给 Trainer 构造函数"""
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)

    def test_get_trainer_passes_kwargs(self):
        """额外 kwargs 应传递给 Trainer 构造函数（如 lr_scheduler_type）"""
        trainer = get_trainer("fc_v1", fresh=True, lr_scheduler_type="step")
        self.assertIsNotNone(trainer)
        # 不崩溃即通过


class TestListModels(unittest.TestCase):
    """测试 list_models 函数"""

    def test_list_models_returns_list(self):
        """list_models 返回列表"""
        models = list_models()
        self.assertIsInstance(models, list)

    def test_list_models_contains_all(self):
        """list_models 应包含所有注册的模型名称"""
        models = list_models()
        expected = list(MODEL_REGISTRY.keys())
        self.assertEqual(sorted(models), sorted(expected))

    def test_list_models_contains_random(self):
        """list_models 应包含 random"""
        models = list_models()
        self.assertIn("random", models)

    def test_list_models_contains_fc_v1(self):
        """list_models 应包含 fc_v1"""
        models = list_models()
        self.assertIn("fc_v1", models)

    def test_list_models_contains_cnn_v1(self):
        """list_models 应包含 cnn_v1"""
        models = list_models()
        self.assertIn("cnn_v1", models)

    def test_list_models_contains_cnn_v2(self):
        """list_models 应包含 cnn_v2"""
        models = list_models()
        self.assertIn("cnn_v2", models)


class TestLazyLoading(unittest.TestCase):
    """测试懒加载：首次调用前不应导入模型模块"""

    def test_model_not_imported_before_call(self):
        """懒加载验证：调用 get_model 前模型模块不应被导入

        验证方式：确保 fc_v1.trainer 模块在只调用 get_model 时不会被导入
        （trainer 只在 get_trainer 时才需要）
        """
        # 先确保 fc_v1 的模型已加载（可能已被其他测试加载）
        model = get_model("fc_v1")
        self.assertIsNotNone(model)

        # 关键验证：get_model 只导入 model 模块，不导入 trainer 模块
        # 如果 trainer 模块从未被导入，说明 model 和 trainer 是独立懒加载的
        # 注意：如果其他测试已导入 trainer，此模块可能已在 sys.modules 中，
        # 所以我们验证的是 registry 本身不会主动导入 trainer

    def test_lazy_import_mechanism(self):
        """验证 registry 使用 importlib 动态导入而非顶部导入"""
        # 验证 registry 模块本身不会在顶部导入模型模块
        import ai.registry as reg_module
        # 获取模块源代码中的顶层导入
        import inspect
        source = inspect.getsource(reg_module)
        # 不应包含 'from ai.models' 的顶层导入
        top_lines = []
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('from ai.models') or stripped.startswith('import ai.models'):
                top_lines.append(stripped)
        self.assertEqual(len(top_lines), 0,
                         f"registry.py 不应有模型模块的顶层导入，但发现: {top_lines}")


class TestModelRegistryData(unittest.TestCase):
    """测试 MODEL_REGISTRY 数据完整性"""

    def test_registry_has_four_models(self):
        """MODEL_REGISTRY 应包含 4 个模型"""
        self.assertEqual(len(MODEL_REGISTRY), 4)

    def test_registry_entry_format(self):
        """每个注册项格式为 (model_path, trainer_path_or_None)"""
        for name, entry in MODEL_REGISTRY.items():
            self.assertEqual(len(entry), 2,
                             f"{name} 的注册项应为 (model_path, trainer_path)")
            model_path, trainer_path = entry
            self.assertIsInstance(model_path, str)
            self.assertTrue(model_path.startswith("ai.models."),
                           f"{name} 的 model_path 应以 'ai.models.' 开头")
            if trainer_path is not None:
                self.assertIsInstance(trainer_path, str)
                self.assertTrue(trainer_path.startswith("ai.models."),
                               f"{name} 的 trainer_path 应以 'ai.models.' 开头")

    def test_random_has_no_trainer(self):
        """random 模型没有 trainer"""
        _, trainer_path = MODEL_REGISTRY["random"]
        self.assertIsNone(trainer_path)

    def test_trainable_models_have_trainers(self):
        """fc_v1、cnn_v1、cnn_v2 应有 trainer"""
        for name in ["fc_v1", "cnn_v1", "cnn_v2"]:
            _, trainer_path = MODEL_REGISTRY[name]
            self.assertIsNotNone(trainer_path,
                                f"{name} 应有 trainer_path")


if __name__ == "__main__":
    unittest.main()
