"""多代模型端到端集成测试

覆盖从 spec 字符串到模型实例、训练器、引擎、app 层异常转换的完整流程，
以及 save/load 代数隔离等场景。

测试要点：
- 不修改真实数据文件和模型文件
- 有输出文件的测试使用临时目录
- 所有断言验证端到端一致性
"""

import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from ai.registry import get_model, get_trainer, list_models_with_full_name
from ai.engine import create_engine
from ai.models.cnn_v3.model import CNNV3Model
from app.exceptions import ModelLoadError
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 测试辅助：在真实 ai/models/cnn_v3/ 下创建/清理 gen{i}/cnn_v3.pth 占位权重
#
# 背景：NeuralNetworkModel 对显式指定 gen 的多代模型实例化时，要求对应 .pth
# 必须存在，否则抛 ValueError。测试这里只验证属性/流程，不关心权重数值，
# 因此用 fresh=True 构造的随机权重 state_dict 保存为占位文件即可。
# ---------------------------------------------------------------------------
def _setup_fake_gen_weights(gens):
    """为指定代数列表在真实模型目录下创建占位 .pth 文件

    Args:
        gens: 要创建的代数列表，如 [1, 2]

    Returns:
        已创建的 .pth 绝对路径列表（tearDown 时按此清理）
    """
    base_dir = os.path.dirname(CNNV3Model.DEFAULT_MODEL_PATH)
    filename = os.path.basename(CNNV3Model.DEFAULT_MODEL_PATH)
    created = []
    for g in gens:
        gen_dir = os.path.join(base_dir, f"gen{g}")
        os.makedirs(gen_dir, exist_ok=True)
        pth_path = os.path.join(gen_dir, filename)
        if not os.path.exists(pth_path):
            tmp_model = CNNV3Model(fresh=True, gen=g)
            torch.save(tmp_model.network.state_dict(), pth_path)
            created.append(pth_path)
    return created


def _cleanup_fake_gen_weights(paths):
    """清理 _setup_fake_gen_weights 创建的占位文件及空 gen 目录"""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
        gen_dir = os.path.dirname(p)
        if os.path.isdir(gen_dir) and not os.listdir(gen_dir):
            os.rmdir(gen_dir)


# ---------------------------------------------------------------------------
# 1. 从 spec 到模型实例的完整流程
# ---------------------------------------------------------------------------
class TestSpecToModelInstance(unittest.TestCase):
    """验证 get_model("cnn_v3.2") 端到端流程"""

    def setUp(self):
        self._fake_pths = _setup_fake_gen_weights([1, 2])

    def tearDown(self):
        _cleanup_fake_gen_weights(self._fake_pths)

    def test_gen_attribute(self):
        """get_model("cnn_v3.2") → model.gen == 2"""
        model = get_model("cnn_v3.2")
        self.assertEqual(model.gen, 2)

    def test_full_name(self):
        """get_model("cnn_v3.2") → model.full_name == "cnn_v3.2" """
        model = get_model("cnn_v3.2")
        self.assertEqual(model.full_name, "cnn_v3.2")

    def test_default_model_path_contains_gen(self):
        """get_model("cnn_v3.2") → DEFAULT_MODEL_PATH 包含 gen2 子目录"""
        model = get_model("cnn_v3.2")
        self.assertIn(os.path.join("gen2", "cnn_v3.pth"), model.DEFAULT_MODEL_PATH)

    def test_gen_1_spec(self):
        """get_model("cnn_v3.1") → gen=1, full_name="cnn_v3.1" """
        model = get_model("cnn_v3.1")
        self.assertEqual(model.gen, 1)
        self.assertEqual(model.full_name, "cnn_v3.1")

    def test_non_multi_gen_model(self):
        """get_model("random") → full_name="random", 无 gen 属性 """
        model = get_model("random")
        # RandomModel 继承 BaseModel（非 NeuralNetworkModel），无 gen 属性
        self.assertFalse(hasattr(model, 'gen'))
        self.assertEqual(model.full_name, "random")


# ---------------------------------------------------------------------------
# 2. 从 spec 到训练器的完整流程
# ---------------------------------------------------------------------------
class TestSpecToTrainer(unittest.TestCase):
    """验证 get_trainer("cnn_v3.2", fresh=True) 端到端流程"""

    def test_trainer_model_gen(self):
        """训练器模型的 gen 属性应为 2"""
        trainer = get_trainer("cnn_v3.2", fresh=True)
        self.assertEqual(trainer.model.gen, 2)

    def test_trainer_model_full_name(self):
        """训练器模型的 full_name 应为 "cnn_v3.2" """
        trainer = get_trainer("cnn_v3.2", fresh=True)
        self.assertEqual(trainer.model.full_name, "cnn_v3.2")

    def test_trainer_model_full_name_property(self):
        """训练器的 model_full_name 属性应为 "cnn_v3.2" """
        trainer = get_trainer("cnn_v3.2", fresh=True)
        self.assertEqual(trainer.model_full_name, "cnn_v3.2")

    def test_trainer_gen_attribute(self):
        """训练器自身的 gen 属性应为 2"""
        trainer = get_trainer("cnn_v3.2", fresh=True)
        self.assertEqual(trainer.gen, 2)


# ---------------------------------------------------------------------------
# 3. list_models_with_full_name 的实际文件系统扫描
# ---------------------------------------------------------------------------
class TestListModelsWithFullNameFileSystem(unittest.TestCase):
    """验证 list_models_with_full_name 使用真实文件系统扫描逻辑"""

    def test_with_gen1_and_gen2_dirs(self):
        """使用临时目录模拟 gen1/gen2 子目录 → 返回 "cnn_v3.1", "cnn_v3.2" """
        with tempfile.TemporaryDirectory() as tmp:
            # 为非多代模型创建空目录
            for name in ["random", "fc_v1", "cnn_v1", "cnn_v2"]:
                os.makedirs(os.path.join(tmp, name), exist_ok=True)
            # 为 cnn_v3 创建 gen1/gen2，并在每个 gen 目录下放一个占位 .pth
            # （list_models_with_full_name 要求 gen 目录内至少有一个 .pth 才
            # 识别为有效代数，避免空目录被误展示）
            for g in [1, 2]:
                gen_dir = os.path.join(tmp, "cnn_v3", f"gen{g}")
                os.makedirs(gen_dir, exist_ok=True)
                open(os.path.join(gen_dir, "cnn_v3.pth"), "w").close()

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            self.assertIn("cnn_v3.1", result)
            self.assertIn("cnn_v3.2", result)
            # 裸名不应出现
            self.assertNotIn("cnn_v3", result)

    def test_no_gen_dirs_hides_multi_gen_model(self):
        """cnn_v3 目录存在但无 gen 子目录 → 不展示"""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["random", "fc_v1", "cnn_v1", "cnn_v2"]:
                os.makedirs(os.path.join(tmp, name), exist_ok=True)
            os.makedirs(os.path.join(tmp, "cnn_v3"), exist_ok=True)

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            self.assertTrue(all("cnn_v3" not in r for r in result))


# ---------------------------------------------------------------------------
# 4. create_engine 多代模型流程
# ---------------------------------------------------------------------------
class TestCreateEngineMultiGen(unittest.TestCase):
    """验证 create_engine 多代模型端到端流程"""

    def setUp(self):
        self._fake_pths = _setup_fake_gen_weights([1, 2])

    def tearDown(self):
        _cleanup_fake_gen_weights(self._fake_pths)

    def test_create_engine_with_valid_spec(self):
        """create_engine("cnn_v3.2") → 引擎模型的 full_name 为 "cnn_v3.2" """
        engine = create_engine("cnn_v3.2")
        self.assertEqual(engine.model.full_name, "cnn_v3.2")

    def test_create_engine_with_gen1(self):
        """create_engine("cnn_v3.1") → 引擎模型的 gen 为 1"""
        engine = create_engine("cnn_v3.1")
        self.assertEqual(engine.model.gen, 1)
        self.assertEqual(engine.model.full_name, "cnn_v3.1")


# ---------------------------------------------------------------------------
# 5. create_engine 错误场景
# ---------------------------------------------------------------------------
class TestCreateEngineError(unittest.TestCase):
    """验证 create_engine 多代未指定代数时抛出 ValueError"""

    def test_multi_gen_without_gen_raises_value_error(self):
        """create_engine("cnn_v3") → ValueError（多代未指定代数）"""
        with self.assertRaises(ValueError) as ctx:
            create_engine("cnn_v3")
        self.assertIn("多代模型必须指定代数", str(ctx.exception))


# ---------------------------------------------------------------------------
# 6. engine → app 层异常转换流程
# ---------------------------------------------------------------------------
class TestEngineToAppExceptionConversion(unittest.TestCase):
    """验证 create_engine ValueError → app 层 ModelLoadError 转换"""

    def test_game_session_converts_value_error(self):
        """GameSession 在 create_engine ValueError 时转换为 ModelLoadError"""
        from game import GomokuController
        from game.save import GomokuSave
        from app import GameSession

        temp_dir = tempfile.mkdtemp(prefix="test_integ_mle_")
        save_dir = os.path.join(temp_dir, "saves")
        try:
            controller = GomokuController()
            controller.save = GomokuSave(save_dir)

            with self.assertRaises(ModelLoadError) as ctx:
                GameSession(
                    mode="pve", controller=controller,
                    ai_name="cnn_v3", ai_player=2,
                )
            # 错误信息应包含原始 ValueError 的提示
            self.assertIn("多代模型必须指定代数", str(ctx.exception))
            # __cause__ 应为 ValueError
            self.assertIsInstance(ctx.exception.__cause__, ValueError)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_battle_session_converts_value_error(self):
        """BattleSession 在 create_engine ValueError 时转换为 ModelLoadError"""
        from app import BattleSession

        temp_dir = tempfile.mkdtemp(prefix="test_integ_battle_")
        ai_data_dir = os.path.join(temp_dir, "ai_data")
        train_data_dir = os.path.join(temp_dir, "train_data")
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        try:
            session = BattleSession(
                ai1_name="cnn_v3",
                ai2_name="random",
                num_games=1,
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
            )
            with self.assertRaises(ModelLoadError) as ctx:
                session.run()
            self.assertIn("模型加载失败", str(ctx.exception))
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. save + load 的代数隔离
# ---------------------------------------------------------------------------
class TestSaveLoadGenIsolation(unittest.TestCase):
    """验证不同代数模型的 save/load 互相隔离"""

    def test_gen1_gen2_save_load_isolated(self):
        """CNNV3Model(gen=1) 和 CNNV3Model(gen=2) 分别保存到不同目录，互不影响"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 gen=1 模型并保存
            model1 = CNNV3Model(fresh=True, gen=1)
            save_path1 = os.path.join(tmpdir, "gen1", "cnn_v3.pth")
            model1.save(save_path1)

            # 创建 gen=2 模型并保存
            model2 = CNNV3Model(fresh=True, gen=2)
            save_path2 = os.path.join(tmpdir, "gen2", "cnn_v3.pth")
            model2.save(save_path2)

            # 验证两个文件分别存在
            self.assertTrue(os.path.isfile(save_path1))
            self.assertTrue(os.path.isfile(save_path2))

            # 验证目录结构隔离
            self.assertNotEqual(
                os.path.dirname(save_path1),
                os.path.dirname(save_path2),
            )

    def test_load_does_not_cross_contaminate(self):
        """加载 gen=1 的权重不影响 gen=2 的模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存 gen=1 模型
            model1 = CNNV3Model(fresh=True, gen=1)
            save_path1 = os.path.join(tmpdir, "gen1", "cnn_v3.pth")
            model1.save(save_path1)

            # 创建并保存 gen=2 模型
            model2 = CNNV3Model(fresh=True, gen=2)
            save_path2 = os.path.join(tmpdir, "gen2", "cnn_v3.pth")
            model2.save(save_path2)

            # 创建新模型，加载 gen=1 的权重
            loader1 = CNNV3Model(fresh=True, gen=1)
            loader1.load(save_path1)

            # 创建新模型，加载 gen=2 的权重
            loader2 = CNNV3Model(fresh=True, gen=2)
            loader2.load(save_path2)

            # 验证加载后 full_name 各自正确
            self.assertEqual(loader1.full_name, "cnn_v3.1")
            self.assertEqual(loader2.full_name, "cnn_v3.2")

            # 验证加载后权重恢复一致：loader1 的预测应与 model1 一致
            board_vector = [0] * 225 + [1]
            out_original1 = model1.predict(board_vector)
            out_loader1 = loader1.predict(board_vector)
            for a, b in zip(out_original1, out_loader1):
                self.assertAlmostEqual(a, b, places=5)

            # 同理 loader2 的预测应与 model2 一致
            out_original2 = model2.predict(board_vector)
            out_loader2 = loader2.predict(board_vector)
            for a, b in zip(out_original2, out_loader2):
                self.assertAlmostEqual(a, b, places=5)

    def test_gen1_save_does_not_create_gen2_dir(self):
        """保存 gen=1 不会在 gen2 目录下创建文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = CNNV3Model(fresh=True, gen=1)
            save_path1 = os.path.join(tmpdir, "gen1", "cnn_v3.pth")
            model1.save(save_path1)

            gen2_dir = os.path.join(tmpdir, "gen2")
            self.assertFalse(
                os.path.exists(gen2_dir),
                "保存 gen=1 模型不应创建 gen2 目录",
            )


if __name__ == "__main__":
    unittest.main()
