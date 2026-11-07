"""CNN v12 注册表集成测试

验证 registry 正确路由 cnn_v12 模型和训练器。
测试覆盖 TC4.1 ~ TC4.8（设计 61 test_cases.md 四）。

前置条件：用例只读。
- 依赖真实权重的用例在 gen1/cnn_v12.pth 不存在时抛 unittest.SkipTest
  （cnn_v12 采用「手工预置权重」冷启动，仓库内当前无权重属正常状态）
- 纯注册表逻辑的用例不依赖权重，始终执行
禁止为跑通测试而创建真实权重文件。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV12Registry(unittest.TestCase):
    """Registry 集成测试（只读：不创建/删除真实模型文件）"""

    @classmethod
    def setUpClass(cls):
        """导入被测模块并定位 cnn_v12 模型目录"""
        from ai.models.cnn_v12.model import CNNV12Model
        cls.ModelCls = CNNV12Model
        cls.ModelDir = os.path.dirname(CNNV12Model.DEFAULT_MODEL_PATH)

    @classmethod
    def _gen_dir(cls, gen: int) -> str:
        """返回 gen{N} 目录的绝对路径"""
        return os.path.join(cls.ModelDir, f"gen{gen}")

    @classmethod
    def _has_any_pth(cls, gen: int) -> bool:
        """gen{N} 目录下是否至少存在一个 .pth 文件（与 registry 判定一致）"""
        gen_dir = cls._gen_dir(gen)
        if not os.path.isdir(gen_dir):
            return False
        return any(f.endswith(".pth") for f in os.listdir(gen_dir))

    @classmethod
    def _valid_gens(cls) -> list:
        """扫描出所有有效代数（gen{N} 目录下存在 .pth）"""
        gens = []
        if not os.path.isdir(cls.ModelDir):
            return gens
        for item in os.listdir(cls.ModelDir):
            if not item.startswith("gen"):
                continue
            gen_str = item[3:]
            if not gen_str:
                continue
            try:
                gen = int(gen_str)
            except ValueError:
                continue
            if gen >= 1 and cls._has_any_pth(gen):
                gens.append(gen)
        return sorted(gens)

    @classmethod
    def _require_gen1_weights(cls) -> None:
        """真实权重依赖用例的前置检查

        缺失时抛 SkipTest（禁止为跑通测试而创建真实权重文件）。

        Raises:
            unittest.SkipTest: gen1/cnn_v12.pth 不存在
        """
        gen1_pth = os.path.join(cls._gen_dir(1), "cnn_v12.pth")
        if not os.path.isfile(gen1_pth):
            raise unittest.SkipTest(
                f"cnn_v12.1 (gen1) 权重不存在: {gen1_pth}，跳过")

    # ---------- TC4.1 注册表包含条目 ----------

    def test_registry_contains_entry(self):
        """MODEL_REGISTRY 含 cnn_v12 且声明为多代模型"""
        from ai.registry import MODEL_REGISTRY

        self.assertIn("cnn_v12", MODEL_REGISTRY)
        entry = MODEL_REGISTRY["cnn_v12"]
        self.assertIs(entry.supports_multi_gen, True)
        self.assertEqual(entry.model_class_path,
                         "ai.models.cnn_v12.model.CNNV12Model")
        self.assertEqual(entry.trainer_class_path,
                         "ai.models.cnn_v12.trainer.CNNV12Trainer")

    # ---------- TC4.2 未指定代数报错 ----------

    def test_get_model_without_gen_raises(self):
        """多代模型未指定代数抛 ValueError"""
        from ai.registry import get_model

        with self.assertRaises(ValueError):
            get_model("cnn_v12")

    # ---------- TC4.3 按代数加载 ----------

    def test_get_model_by_gen(self):
        """get_model('cnn_v12.1') 返回 CNNV12Model，gen=1"""
        self._require_gen1_weights()
        from ai.registry import get_model

        model = get_model("cnn_v12.1")
        self.assertEqual(type(model).__name__, "CNNV12Model")
        self.assertEqual(model.name, "cnn_v12")
        self.assertEqual(model.gen, 1)

    # ---------- TC4.4 训练器加载 ----------

    def test_get_trainer_by_gen(self):
        """get_trainer('cnn_v12.1') 返回 CNNV12Trainer"""
        self._require_gen1_weights()
        from ai.registry import get_trainer

        trainer = get_trainer("cnn_v12.1", device=torch.device("cpu"))
        self.assertEqual(type(trainer).__name__, "CNNV12Trainer")
        self.assertEqual(trainer.model_name, "cnn_v12")

    # ---------- TC4.5 模型清单可见性 ----------

    def test_list_models_visibility(self):
        """list_models_with_full_name 仅在 gen 目录有 .pth 时展示 cnn_v12.N"""
        from ai.registry import list_models_with_full_name

        listed = {name for name in list_models_with_full_name()
                  if name.startswith("cnn_v12")}
        expected = {f"cnn_v12.{gen}" for gen in self._valid_gens()}
        self.assertEqual(listed, expected,
                         f"cnn_v12 可见代数应为 {sorted(expected)}，"
                         f"实际 {sorted(listed)}")

    # ---------- TC4.6 最高代数 ----------

    def test_get_highest_gen(self):
        """get_highest_gen('cnn_v12') 与文件系统有效代数一致"""
        from ai.registry import get_highest_gen

        valid_gens = self._valid_gens()
        if not valid_gens:
            # 无任何有效代数 → 抛 ValueError（cnn_v12 冷启动前的正常状态）
            with self.assertRaises(ValueError):
                get_highest_gen("cnn_v12")
            return

        self.assertEqual(get_highest_gen("cnn_v12"), max(valid_gens))

    # ---------- TC4.7 引擎路由 ----------

    def test_create_engine(self):
        """create_engine('cnn_v12.1') 返回 PolicyValueAiEngine（Python MCTS 回退）"""
        self._require_gen1_weights()
        from ai.engine.builder import create_engine

        engine = create_engine("cnn_v12.1")
        self.assertEqual(type(engine).__name__, "PolicyValueAiEngine")

    # ---------- TC4.8 未知模型不误命中 ----------

    def test_unknown_model_not_matched(self):
        """get_model('cnn_v13.1') 抛 ValueError"""
        from ai.registry import get_model

        with self.assertRaises(ValueError):
            get_model("cnn_v13.1")


if __name__ == '__main__':
    unittest.main()
