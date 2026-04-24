"""CNN v2 模型测试

测试 CNN v2 轻量化卷积神经网络模型的功能。
"""

import os
import sys
import unittest
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from ai.models.cnn_v2.model import CNNV2Model, CNNV2Network


class TestCNNv2Network(unittest.TestCase):
    """测试 CNN v2 神经网络结构"""

    def test_network_structure(self):
        """测试网络结构是否正确"""
        network = CNNV2Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        self.assertEqual(output.shape, (batch_size, 225))

    def test_network_softmax_output_range(self):
        """[primary] Softmax 输出值域应在 (0, 1) 范围内"""
        network = CNNV2Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        self.assertTrue(torch.all(output > 0), "Softmax 输出应大于 0")
        self.assertTrue(torch.all(output < 1), "Softmax 输出应小于 1")

    def test_network_softmax_row_sum(self):
        """[primary] Softmax 每行概率和应为 1"""
        network = CNNV2Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        row_sums = output.sum(dim=1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-5),
                       "Softmax 每行概率和应接近 1")

    def test_network_forward_single(self):
        """测试单样本前向传播"""
        network = CNNV2Network()

        input_tensor = torch.randn(1, 3, 15, 15)
        output = network(input_tensor)

        self.assertEqual(output.shape, (1, 225))

    def test_network_forward_batch(self):
        """测试批量前向传播"""
        network = CNNV2Network()

        batch_sizes = [2, 8, 16]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 15, 15)
            output = network(input_tensor)
            self.assertEqual(output.shape, (batch_size, 225))

    def test_network_param_count(self):
        """CNN v2 参数量应约为 101 万（3层64通道）"""
        network = CNNV2Network()
        total = sum(p.numel() for p in network.parameters())

        # 3层64通道轻量化设计：参数量从 197 万降至约 101 万
        self.assertGreaterEqual(total, 1_000_000,
                               f"参数量 {total} 应 >= 100 万")
        self.assertLessEqual(total, 1_050_000,
                            f"参数量 {total} 应 <= 105 万")


class TestCNNv2Model(unittest.TestCase):
    """测试 CNN v2 模型类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")

        self.original_default_path = CNNV2Model.DEFAULT_MODEL_PATH
        CNNV2Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "cnn_v2.pth")

    def tearDown(self):
        """测试后清理"""
        CNNV2Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_name(self):
        """测试模型名称"""
        model = CNNV2Model()
        self.assertEqual(model.name, "cnn_v2")

    def test_model_predict_shape(self):
        """测试 predict 方法返回形状"""
        model = CNNV2Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        self.assertEqual(len(result), 225)

    def test_model_predict_probability_range(self):
        """[primary] 测试 predict 方法返回的概率范围"""
        model = CNNV2Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        for i in range(225):
            prob = result[i]
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] 测试 predict 方法返回的概率总和为 1"""
        model = CNNV2Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        prob_sum = sum(result)
        self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="概率总和应为 1")

    def test_model_save_and_load(self):
        """测试模型保存和加载"""
        model1 = CNNV2Model()
        model1.save(self.model_path)

        self.assertTrue(os.path.exists(self.model_path))

        model2 = CNNV2Model(self.model_path)
        self.assertEqual(model2.name, "cnn_v2")

    def test_model_load_nonexistent_file(self):
        """测试加载不存在的模型文件"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pth")

        model = CNNV2Model(nonexistent_path)
        self.assertEqual(model.name, "cnn_v2")

    def test_vector_to_tensor_conversion(self):
        """测试向量到张量的转换"""
        model = CNNV2Model()

        board_vector = [0] * 225 + [1]
        board_vector[0] = 1
        board_vector[1 * 15 + 1] = 2

        tensor = model._vector_to_tensor(board_vector)

        self.assertEqual(tensor.shape, (1, 3, 15, 15))
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(tensor[0, 1, 1, 1].item(), 1.0)
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1.0)


class TestCNNv2ModelIntegration(unittest.TestCase):
    """CNN v2 模型集成测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "cnn_v2.pth")

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """测试完整工作流程"""
        model1 = CNNV2Model()

        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)

        model1.save(self.model_path)

        model2 = CNNV2Model(self.model_path)
        result2 = model2.predict(board_vector)

        self.assertEqual(result1, result2)


class TestCNNv2ModelDefaultLoad(unittest.TestCase):
    """测试 CNN v2 模型默认加载行为"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_default_path = CNNV2Model.DEFAULT_MODEL_PATH
        CNNV2Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "cnn_v2.pth")

    def tearDown(self):
        """测试后清理"""
        CNNV2Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_load_when_model_exists(self):
        """测试当默认模型文件存在时自动加载"""
        model1 = CNNV2Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()

        model2 = CNNV2Model()
        result2 = model2.predict(board_vector)

        self.assertEqual(result1, result2)

    def test_fresh_parameter_skips_load(self):
        """测试 fresh=True 时跳过加载"""
        model1 = CNNV2Model(fresh=True)
        model1.save()

        model2 = CNNV2Model(fresh=True)
        self.assertEqual(model2.name, "cnn_v2")


if __name__ == '__main__':
    unittest.main()
