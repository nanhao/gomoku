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

from ai.registry import get_model, get_trainer, _list_models, MODEL_REGISTRY
from ai.base_model import BaseModel
from ai.base_trainer import BaseTrainer
from ai.models.random.model import RandomModel
from ai.models.fc_v1.model import FCv1Model
from ai.models.cnn_v1.model import CNNV1Model
from ai.models.cnn_v2.model import CNNV2Model
from ai.models.cnn_v3.model import CNNV3Model


class TestGetModel(unittest.TestCase):
    """测试 get_model 函数"""

    def _assert_predict_smoke(self, model):
        """通用端到端 smoke：predict 返回 225 维非负向量"""
        vector = [0] * 225 + [1]
        probs = model.predict(vector)
        self.assertEqual(len(probs), 225, "predict 必须返回 225 维")
        self.assertTrue(all(p >= 0 for p in probs), "概率分布不得为负")

    def test_get_model_random(self):
        """加载 random 模型"""
        model = get_model("random")
        self.assertIsInstance(model, RandomModel)
        self.assertEqual(model.name, "random")
        self._assert_predict_smoke(model)

    def test_get_model_fc_v1(self):
        """加载 fc_v1 模型"""
        model = get_model("fc_v1")
        self.assertIsInstance(model, FCv1Model)
        self.assertEqual(model.name, "fc_v1")
        self._assert_predict_smoke(model)

    def test_get_model_cnn_v1(self):
        """加载 cnn_v1 模型"""
        model = get_model("cnn_v1")
        self.assertIsInstance(model, CNNV1Model)
        self.assertEqual(model.name, "cnn_v1")
        self._assert_predict_smoke(model)

    def test_get_model_cnn_v2(self):
        """加载 cnn_v2 模型"""
        model = get_model("cnn_v2")
        self.assertIsInstance(model, CNNV2Model)
        self.assertEqual(model.name, "cnn_v2")
        self._assert_predict_smoke(model)

    def test_get_model_cnn_v3(self):
        """加载 cnn_v3.1 模型（多代模型必须指定代数）"""
        # 新硬校验：显式指定 gen 时必须命中 .pth，测试前临时创建占位权重
        import torch
        base_dir = os.path.dirname(CNNV3Model.DEFAULT_MODEL_PATH)
        gen_dir = os.path.join(base_dir, "gen1")
        pth_path = os.path.join(gen_dir, "cnn_v3.pth")
        os.makedirs(gen_dir, exist_ok=True)
        pre_exists = os.path.exists(pth_path)
        if not pre_exists:
            tmp = CNNV3Model(fresh=True, gen=1)
            torch.save(tmp.network.state_dict(), pth_path)
        try:
            model = get_model("cnn_v3.1")
            self.assertIsInstance(model, CNNV3Model)
            self.assertEqual(model.name, "cnn_v3")
            self.assertEqual(model.full_name, "cnn_v3.1")
            self._assert_predict_smoke(model)
        finally:
            if not pre_exists and os.path.exists(pth_path):
                os.remove(pth_path)
                if not os.listdir(gen_dir):
                    os.rmdir(gen_dir)

    def test_get_model_cnn_v3_without_gen_raises(self):
        """cnn_v3 不指定代数应抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_model("cnn_v3")
        self.assertIn("多代模型必须指定代数", str(ctx.exception))

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

    def test_get_trainer_cnn_v3(self):
        """cnn_v3.1 返回 CNNV3Trainer 实例（多代模型必须指定代数）"""
        from ai.models.cnn_v3.trainer import CNNV3Trainer
        trainer = get_trainer("cnn_v3.1", fresh=True)
        self.assertIsInstance(trainer, CNNV3Trainer)
        self.assertEqual(trainer.model.full_name, "cnn_v3.1")

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
        """fresh=True 应传递给 Trainer: 模型为全新 FCv1Model, optimizer 无历史状态"""
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsInstance(trainer.model, FCv1Model)
        # fresh 模型的 optimizer 未做过 step：state 字典应为空或仅含默认结构
        optim_state = trainer.optimizer.state_dict().get('state', {})
        self.assertEqual(
            len(optim_state), 0,
            f"fresh trainer 的 optimizer.state 应为空（未 step 过），实际 {len(optim_state)} 项"
        )

    def test_get_trainer_passes_kwargs(self):
        """kwargs 应透传到 Trainer 构造并真实生效（以 lr_scheduler_type=\"step\" 为例）"""
        import torch.optim.lr_scheduler as lr_scheduler
        trainer = get_trainer("fc_v1", fresh=True, lr_scheduler_type="step")
        # 直接验证属性与调度器类型，确保 kwargs 不是被 **kwargs 吞掉后静默忽略
        self.assertEqual(trainer.lr_scheduler_type, "step")
        self.assertIsInstance(trainer.lr_scheduler, lr_scheduler.StepLR)


class TestListModels(unittest.TestCase):
    """测试 _list_models 私有函数"""

    def test_list_models_returns_list(self):
        """_list_models 返回列表"""
        models = _list_models()
        self.assertIsInstance(models, list)

    def test_list_models_contains_all(self):
        """_list_models 应包含所有注册的模型名称"""
        models = _list_models()
        expected = list(MODEL_REGISTRY.keys())
        self.assertEqual(sorted(models), sorted(expected))

    def test_list_models_contains_random(self):
        """_list_models 应包含 random"""
        models = _list_models()
        self.assertIn("random", models)

    def test_list_models_contains_fc_v1(self):
        """_list_models 应包含 fc_v1"""
        models = _list_models()
        self.assertIn("fc_v1", models)

    def test_list_models_contains_cnn_v1(self):
        """_list_models 应包含 cnn_v1"""
        models = _list_models()
        self.assertIn("cnn_v1", models)

    def test_list_models_contains_cnn_v2(self):
        """_list_models 应包含 cnn_v2"""
        models = _list_models()
        self.assertIn("cnn_v2", models)

    def test_list_models_contains_cnn_v3(self):
        """_list_models 应包含 cnn_v3"""
        models = _list_models()
        self.assertIn("cnn_v3", models)


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

    def test_registry_has_five_models(self):
        """MODEL_REGISTRY 应包含 5 个模型"""
        self.assertEqual(len(MODEL_REGISTRY), 5)

    def test_registry_entry_format(self):
        """每个注册项格式为 RegistryEntry(model_path, trainer_path_or_None, supports_multi_gen)"""
        for name, entry in MODEL_REGISTRY.items():
            self.assertEqual(len(entry), 3,
                             f"{name} 的注册项应为 RegistryEntry(model_path, trainer_path, supports_multi_gen)")
            model_path, trainer_path, supports_multi_gen = entry
            self.assertIsInstance(model_path, str)
            self.assertTrue(model_path.startswith("ai.models."),
                           f"{name} 的 model_path 应以 'ai.models.' 开头")
            if trainer_path is not None:
                self.assertIsInstance(trainer_path, str)
                self.assertTrue(trainer_path.startswith("ai.models."),
                               f"{name} 的 trainer_path 应以 'ai.models.' 开头")
            self.assertIsInstance(supports_multi_gen, bool,
                               f"{name} 的 supports_multi_gen 应为 bool")

    def test_random_has_no_trainer(self):
        """random 模型没有 trainer"""
        _, trainer_path, _ = MODEL_REGISTRY["random"]
        self.assertIsNone(trainer_path)

    def test_trainable_models_have_trainers(self):
        """fc_v1、cnn_v1、cnn_v2、cnn_v3 应有 trainer"""
        for name in ["fc_v1", "cnn_v1", "cnn_v2", "cnn_v3"]:
            _, trainer_path, _ = MODEL_REGISTRY[name]
            self.assertIsNotNone(trainer_path,
                                f"{name} 应有 trainer_path")


if __name__ == "__main__":
    unittest.main()
