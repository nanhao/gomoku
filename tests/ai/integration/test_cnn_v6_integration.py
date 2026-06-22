"""CNN v6 集成测试

验证 cnn_v6 在模型注册表中的集成正确性：
- get_model / get_trainer 能正确返回 CNNV6Model / CNNV6Trainer 实例
- predict 通过注册表调用正常工作
- list_models_with_full_name 包含 cnn_v6 条目

测试使用临时目录隔离，不修改任何真实模型文件。
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from ai.registry import get_model, get_trainer, list_models_with_full_name, _MODELS_DIR
from ai.models.cnn_v6.model import CNNV6Model, CNNV6Network
from ai.models.cnn_v6.trainer import CNNV6Trainer


class TestCNNV6Integration(unittest.TestCase):
    """CNN v6 注册表集成测试

    在临时目录中创建 gen 子目录和虚拟 .pth 文件，
    验证 get_model/get_trainer 集成链路。
    """

    def setUp(self):
        """创建临时目录和虚拟模型文件"""
        self.temp_dir = tempfile.mkdtemp()

        # 模拟 cnn_v6 模型目录结构：temp_dir/cnn_v6/gen1/cnn_v6.pth
        self.model_dir = os.path.join(self.temp_dir, "cnn_v6")
        self.gen_dir = os.path.join(self.model_dir, "gen1")
        os.makedirs(self.gen_dir, exist_ok=True)

        # 创建一个干净的 CNNV6Model 并保存到 gen1 目录
        self.model_pth_path = os.path.join(self.gen_dir, "cnn_v6.pth")
        model = CNNV6Model(fresh=True)
        model.save(self.model_pth_path)
        self.assertTrue(
            os.path.exists(self.model_pth_path),
            f"临时模型文件未创建: {self.model_pth_path}"
        )

        # Mock CNNV6Model.DEFAULT_MODEL_PATH，指向临时目录
        # 这样 CNNV6Model(gen=1) 会计算出 temp_dir/cnn_v6/gen1/cnn_v6.pth
        self._model_path_patcher = mock.patch.object(
            CNNV6Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.model_dir, "cnn_v6.pth")
        )
        self._model_path_patcher.start()

        # Mock registry._MODELS_DIR，让 list_models_with_full_name
        # 扫描临时目录而不是真实的 ai/models/ 目录
        self._models_dir_patcher = mock.patch(
            'ai.registry._MODELS_DIR', self.temp_dir
        )
        self._models_dir_patcher.start()

    def tearDown(self):
        """清理临时目录和 mock"""
        self._model_path_patcher.stop()
        self._models_dir_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- UT-INT-1：get_model("cnn_v6.1") 返回 CNNV6Model 实例 ---

    def test_get_model_returns_cnn_v6_model(self):
        """[UT-INT-1] get_model('cnn_v6.1') 返回 CNNV6Model 实例"""
        model = get_model("cnn_v6.1")
        self.assertIsInstance(
            model, CNNV6Model,
            "get_model('cnn_v6.1') 应返回 CNNV6Model 实例"
        )
        self.assertEqual(model.gen, 1, "gen 属性应为 1")
        self.assertEqual(model.full_name, "cnn_v6.1",
                         "full_name 应为 'cnn_v6.1'")

    # --- UT-INT-2：get_trainer("cnn_v6.1") 返回 CNNV6Trainer 实例 ---

    def test_get_trainer_returns_cnn_v6_trainer(self):
        """[UT-INT-2] get_trainer('cnn_v6.1') 返回 CNNV6Trainer 实例"""
        trainer = get_trainer("cnn_v6.1")
        self.assertIsInstance(
            trainer, CNNV6Trainer,
            "get_trainer('cnn_v6.1') 应返回 CNNV6Trainer 实例"
        )
        self.assertEqual(trainer.model_name, "cnn_v6",
                         "trainer.model_name 应为 'cnn_v6'")

    # --- UT-INT-3：通过注册表获取模型后 predict 正常 ---

    def test_get_model_predict_works(self):
        """[UT-INT-3] get_model('cnn_v6.1').predict(board) 返回 (列表, 价值) 元组"""
        model = get_model("cnn_v6.1")
        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)
        self.assertIsInstance(result, tuple,
                              "predict 应返回元组")
        self.assertEqual(len(result), 2,
                         "predict 应返回 (策略, 价值) 元组")
        policy, value = result
        self.assertEqual(len(policy), 225,
                         "策略应为 225 维列表")
        self.assertIsInstance(policy[0], float,
                              "策略元素应为 float")
        self.assertIsInstance(value, float,
                              "价值应为 float")
        self.assertGreaterEqual(value, -1.0,
                                 "价值应 >= -1")
        self.assertLessEqual(value, 1.0,
                              "价值应 <= 1")

    # --- UT-INT-4：list_models_with_full_name 包含 cnn_v6 条目 ---

    def test_list_models_includes_cnn_v6(self):
        """[UT-INT-4] list_models_with_full_name() 包含 cnn_v6.1"""
        models = list_models_with_full_name()
        self.assertIn(
            "cnn_v6.1", models,
            f"list_models_with_full_name() 应包含 'cnn_v6.1'，实际: {models}"
        )

    # --- 边界条件：get_model 无代数时抛出异常 ---

    def test_get_model_without_gen_raises_error(self):
        """多代模型 cnn_v6 不指定代数时抛出 ValueError"""
        with self.assertRaises(ValueError,
                               msg="多代模型必须指定代数"):
            get_model("cnn_v6")

    # --- 边界条件：get_trainer 无代数时抛出异常 ---

    def test_get_trainer_without_gen_raises_error(self):
        """多代模型 cnn_v6 不指定代数时抛出 ValueError"""
        with self.assertRaises(ValueError,
                               msg="多代模型必须指定代数"):
            get_trainer("cnn_v6")

    # --- 异常情况：权重文件不存在时 get_model 抛出异常 ---

    def test_get_model_missing_weights_raises_error(self):
        """gen 目录存在但无权重文件时抛出 ValueError"""
        # gen2 目录不存在，CNNV6Model(gen=2) 找不到权重文件应报错
        with self.assertRaises(ValueError,
                               msg="模型权重文件不存在时报错"):
            get_model("cnn_v6.2")


if __name__ == '__main__':
    unittest.main()
