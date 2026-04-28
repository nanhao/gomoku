"""CNN v3 极简模型测试

测试 CNN v3 极简卷积神经网络模型的功能。
结构：3层卷积(32通道, 无BN) + FC(7200→225) + Softmax
"""

import os
import sys
import unittest
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from ai.models.cnn_v3.model import CNNV3Model, CNNV3Network
from ai.base_model import board_vector_to_cnn_tensor


class TestCNNv3Network(unittest.TestCase):
    """测试 CNN v3 神经网络结构"""

    def test_network_structure(self):
        """测试网络结构：前向传播输出形状为 (batch, 225)"""
        network = CNNV3Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        self.assertEqual(output.shape, (batch_size, 225))

    def test_network_softmax_output_range(self):
        """[primary] Softmax 输出值域应在 (0, 1) 范围内"""
        network = CNNV3Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        self.assertTrue(torch.all(output > 0), "Softmax 输出应大于 0")
        self.assertTrue(torch.all(output < 1), "Softmax 输出应小于 1")

    def test_network_softmax_row_sum(self):
        """[primary] Softmax 每行概率和应为 1"""
        network = CNNV3Network()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        row_sums = output.sum(dim=1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-5),
                       "Softmax 每行概率和应接近 1")

    def test_network_forward_single(self):
        """测试单样本前向传播"""
        network = CNNV3Network()

        input_tensor = torch.randn(1, 3, 15, 15)
        output = network(input_tensor)

        self.assertEqual(output.shape, (1, 225))

    def test_network_forward_batch(self):
        """测试批量前向传播"""
        network = CNNV3Network()

        batch_sizes = [2, 8, 16]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 15, 15)
            output = network(input_tensor)
            self.assertEqual(output.shape, (batch_size, 225))

    def test_network_no_bn(self):
        """网络中不包含 BatchNorm 层"""
        network = CNNV3Network()

        # 检查网络中不应有任何 BatchNorm 层
        for module in network.modules():
            self.assertNotIsInstance(module, (nn.BatchNorm1d, nn.BatchNorm2d),
                                   f"网络中不应包含 BatchNorm 层，但发现 {type(module).__name__}")

    def test_network_no_dropout(self):
        """网络中不包含 Dropout 层"""
        network = CNNV3Network()

        for module in network.modules():
            self.assertNotIsInstance(module, nn.Dropout,
                                   f"网络中不应包含 Dropout 层，但发现 {type(module).__name__}")

    def test_network_param_count(self):
        """CNN v3 参数量应约为 49.5 万（3层32通道，无BN，64维瓶颈FC）"""
        network = CNNV3Network()
        total = sum(p.numel() for p in network.parameters())

        # 3层32通道 + FC1(7200→64) + FC_output(64→225): 参数量约 49.5 万
        self.assertGreaterEqual(total, 490_000,
                               f"参数量 {total} 应 >= 49 万")
        self.assertLessEqual(total, 500_000,
                            f"参数量 {total} 应 <= 50 万")

    def test_network_conv_channels(self):
        """验证卷积层通道数为 32"""
        network = CNNV3Network()

        # Conv1: 3→32
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)

        # Conv2: 32→32
        self.assertEqual(network.conv2.in_channels, 32)
        self.assertEqual(network.conv2.out_channels, 32)

        # Conv3: 32→32
        self.assertEqual(network.conv3.in_channels, 32)
        self.assertEqual(network.conv3.out_channels, 32)

    def test_network_fc_direct_output(self):
        """验证 FC 层为 64 维瓶颈结构：7200→64→225"""
        network = CNNV3Network()

        # FC1 层输入应为 32*15*15=7200，输出为 64（瓶颈）
        self.assertEqual(network.fc1.in_features, 32 * 15 * 15)
        self.assertEqual(network.fc1.out_features, 64)

        # FC_output 层输入为 64，输出为 225
        self.assertEqual(network.fc_output.in_features, 64)
        self.assertEqual(network.fc_output.out_features, 225)


class TestCNNv3Model(unittest.TestCase):
    """测试 CNN v3 模型类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")

        self.original_default_path = CNNV3Model.DEFAULT_MODEL_PATH
        CNNV3Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "cnn_v3.pth")

    def tearDown(self):
        """测试后清理"""
        CNNV3Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_name(self):
        """测试模型名称"""
        model = CNNV3Model()
        self.assertEqual(model.name, "cnn_v3")

    def test_model_predict_shape(self):
        """测试 predict 方法返回形状"""
        model = CNNV3Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        self.assertEqual(len(result), 225)

    def test_model_predict_probability_range(self):
        """[primary] 测试 predict 方法返回的概率范围"""
        model = CNNV3Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        for i in range(225):
            prob = result[i]
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] 测试 predict 方法返回的概率总和为 1"""
        model = CNNV3Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        prob_sum = sum(result)
        self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="概率总和应为 1")

    def test_model_save_and_load(self):
        """测试模型保存和加载"""
        model1 = CNNV3Model()
        model1.save(self.model_path)

        self.assertTrue(os.path.exists(self.model_path))

        model2 = CNNV3Model(self.model_path)
        self.assertEqual(model2.name, "cnn_v3")

    def test_model_load_nonexistent_file(self):
        """测试加载不存在的模型文件"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pth")

        model = CNNV3Model(nonexistent_path)
        self.assertEqual(model.name, "cnn_v3")

    def test_vector_to_tensor_conversion(self):
        """测试向量到张量的转换"""
        board_vector = [0] * 225 + [1]
        board_vector[0] = 1
        board_vector[1 * 15 + 1] = 2

        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))

        self.assertEqual(tensor.shape, (1, 3, 15, 15))
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(tensor[0, 1, 1, 1].item(), 1.0)
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1.0)


class TestCNNv3ModelIntegration(unittest.TestCase):
    """CNN v3 模型集成测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "cnn_v3.pth")

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """测试完整工作流程：创建→预测→保存→加载→再预测"""
        model1 = CNNV3Model()

        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)

        model1.save(self.model_path)

        model2 = CNNV3Model(self.model_path)
        result2 = model2.predict(board_vector)

        self.assertEqual(result1, result2)


class TestCNNv3ModelDefaultLoad(unittest.TestCase):
    """测试 CNN v3 模型默认加载行为"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_default_path = CNNV3Model.DEFAULT_MODEL_PATH
        CNNV3Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "cnn_v3.pth")

    def tearDown(self):
        """测试后清理"""
        CNNV3Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_load_when_model_exists(self):
        """测试当默认模型文件存在时自动加载"""
        model1 = CNNV3Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()

        model2 = CNNV3Model()
        result2 = model2.predict(board_vector)

        self.assertEqual(result1, result2)

    def test_fresh_parameter_skips_load(self):
        """测试 fresh=True 时跳过加载"""
        model1 = CNNV3Model(fresh=True)
        model1.save()

        model2 = CNNV3Model(fresh=True)
        self.assertEqual(model2.name, "cnn_v3")


if __name__ == '__main__':
    unittest.main()
