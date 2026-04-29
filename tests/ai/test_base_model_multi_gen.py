"""多代模型基类改造测试

测试范围：
- NeuralNetworkModel 新增 gen 参数和 full_name property
- BaseModel 新增 full_name 接口（默认实现返回 name）
- save() 自动创建 gen 子目录
- RandomModel 不接受 gen 参数

TDD 流程：先写测试，再改实现。
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.base_model import BaseModel, NeuralNetworkModel
from ai.models.fc_v1.model import FCv1Model
from ai.models.cnn_v3.model import CNNV3Model
from ai.models.random.model import RandomModel


class TestFullNameProperty(unittest.TestCase):
    """§5.6 full_name property 测试用例"""

    def test_fcv1_full_name_without_gen(self):
        """FCv1Model() 的 full_name 应为 'fc_v1'（非多代，与 name 相同）"""
        model = FCv1Model(fresh=True)
        self.assertEqual(model.full_name, "fc_v1")
        self.assertEqual(model.full_name, model.name)

    def test_cnnv3_full_name_without_gen(self):
        """CNNV3Model() 的 full_name 应为 'cnn_v3'（多代但 gen=None，与 name 相同）"""
        model = CNNV3Model(fresh=True)
        self.assertEqual(model.full_name, "cnn_v3")
        self.assertEqual(model.full_name, model.name)

    def test_cnnv3_full_name_with_gen_1(self):
        """CNNV3Model(gen=1) 的 full_name 应为 'cnn_v3.1'"""
        model = CNNV3Model(fresh=True, gen=1)
        self.assertEqual(model.full_name, "cnn_v3.1")
        self.assertEqual(model.name, "cnn_v3")

    def test_cnnv3_full_name_with_gen_2(self):
        """CNNV3Model(gen=2) 的 full_name 应为 'cnn_v3.2'"""
        model = CNNV3Model(fresh=True, gen=2)
        self.assertEqual(model.full_name, "cnn_v3.2")
        self.assertEqual(model.name, "cnn_v3")

    def test_random_full_name(self):
        """RandomModel() 的 full_name 应为 'random'（非多代，与 name 相同）"""
        model = RandomModel()
        self.assertEqual(model.full_name, "random")
        self.assertEqual(model.full_name, model.name)


class TestGenParameterPassed(unittest.TestCase):
    """gen 参数传递测试"""

    def test_fcv1_gen_attribute_set(self):
        """FCv1Model(fresh=True, gen=3) 应设置 gen 属性为 3"""
        model = FCv1Model(fresh=True, gen=3)
        self.assertEqual(model.gen, 3)

    def test_cnnv3_gen_attribute_set(self):
        """CNNV3Model(fresh=True, gen=5) 应设置 gen 属性为 5"""
        model = CNNV3Model(fresh=True, gen=5)
        self.assertEqual(model.gen, 5)

    def test_gen_none_by_default(self):
        """不传入 gen 时，gen 属性应为 None"""
        model = CNNV3Model(fresh=True)
        self.assertIsNone(model.gen)

    def test_gen_parameter_via_registry(self):
        """通过 registry.get_model 传入 gen 参数"""
        from ai.registry import get_model
        model = get_model("cnn_v3.2")
        self.assertEqual(model.gen, 2)
        self.assertEqual(model.full_name, "cnn_v3.2")


class TestRandomModelRejectsGen(unittest.TestCase):
    """RandomModel 不接受 gen 参数的测试"""

    def test_random_model_rejects_gen(self):
        """RandomModel(gen=1) 应抛出 TypeError"""
        with self.assertRaises(TypeError):
            RandomModel(gen=1)


class TestSavePathWithGen(unittest.TestCase):
    """§5.7 model.save() 路径测试用例"""

    def test_fcv1_save_path_without_gen(self):
        """FCv1Model() 保存路径应为 ai/models/fc_v1/fc_v1.pth（非多代不变）"""
        model = FCv1Model(fresh=True)
        expected = os.path.join(os.path.dirname(__file__), "..", "..", "ai", "models", "fc_v1", "fc_v1.pth")
        expected = os.path.normpath(os.path.abspath(expected))
        self.assertEqual(os.path.normpath(model.DEFAULT_MODEL_PATH), expected)

    def test_cnnv3_save_path_with_gen(self):
        """CNNV3Model(gen=2) 保存路径应为 ai/models/cnn_v3/gen2/cnn_v3.pth"""
        model = CNNV3Model(fresh=True, gen=2)
        expected = os.path.join(
            os.path.dirname(__file__), "..", "..", "ai", "models", "cnn_v3", "gen2", "cnn_v3.pth"
        )
        expected = os.path.normpath(os.path.abspath(expected))
        self.assertEqual(os.path.normpath(model.DEFAULT_MODEL_PATH), expected)

    def test_save_creates_gen_directory(self):
        """gen 子目录不存在时，save() 应自动创建目录后保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = CNNV3Model(fresh=True, gen=2)
            # 使用临时路径覆盖默认路径
            save_path = os.path.join(tmpdir, "gen2", "cnn_v3.pth")
            # 目录不应预先存在
            self.assertFalse(os.path.exists(os.path.dirname(save_path)))
            # 保存
            result = model.save(save_path)
            # 应返回保存路径
            self.assertEqual(result, save_path)
            # 目录应被自动创建
            self.assertTrue(os.path.isdir(os.path.dirname(save_path)))
            # 文件应存在
            self.assertTrue(os.path.isfile(save_path))

    def test_save_default_path_creates_gen_directory(self):
        """使用 DEFAULT_MODEL_PATH 且 gen 不为 None 时，save() 自动创建 gen 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = CNNV3Model(fresh=True, gen=2)
            # 临时覆盖 DEFAULT_MODEL_PATH 到临时目录下
            gen_dir = os.path.join(tmpdir, "gen2")
            save_path = os.path.join(gen_dir, "cnn_v3.pth")
            model.DEFAULT_MODEL_PATH = save_path
            # 目录不应预先存在
            self.assertFalse(os.path.exists(gen_dir))
            # 保存
            result = model.save()
            # 应自动创建目录并保存
            self.assertTrue(os.path.isdir(gen_dir))
            self.assertTrue(os.path.isfile(save_path))
            self.assertEqual(result, save_path)

    def test_save_without_gen_no_extra_directory(self):
        """非多代模型 save() 不应创建额外的 gen 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FCv1Model(fresh=True)
            save_path = os.path.join(tmpdir, "fc_v1.pth")
            model.DEFAULT_MODEL_PATH = save_path
            result = model.save()
            self.assertTrue(os.path.isfile(save_path))
            self.assertEqual(result, save_path)


class TestBaseModelFullNameInterface(unittest.TestCase):
    """BaseModel full_name 接口完整性测试"""

    def test_base_model_has_full_name(self):
        """BaseModel 应声明 full_name property"""
        self.assertTrue(hasattr(BaseModel, 'full_name'))

    def test_neural_network_model_has_full_name(self):
        """NeuralNetworkModel 应覆盖 full_name property"""
        self.assertTrue(hasattr(NeuralNetworkModel, 'full_name'))


class TestLoadWithGenPath(unittest.TestCase):
    """load 与 gen 路径的交互测试"""

    def test_load_from_gen_path(self):
        """从 gen 路径保存后再加载应成功"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = CNNV3Model(fresh=True, gen=2)
            save_path = os.path.join(tmpdir, "gen2", "cnn_v3.pth")
            model.save(save_path)
            # 重新创建模型并加载
            model2 = CNNV3Model(fresh=True, gen=2)
            model2.load(save_path)
            # 加载后网络参数应一致（通过前向传播验证）
            board_vector = [0] * 225 + [1]
            out1 = model.predict(board_vector)
            out2 = model2.predict(board_vector)
            for a, b in zip(out1, out2):
                self.assertAlmostEqual(a, b, places=5)


if __name__ == '__main__':
    unittest.main()
