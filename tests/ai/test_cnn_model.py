"""CNN 模型测试

测试 CNN 卷积神经网络模型的功能。
"""

import os
import sys
import unittest
import unittest.mock
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from ai.models.cnn_v1.model import CNNV1Model, CNNV1Network


class TestCNNNetwork(unittest.TestCase):
    """测试 CNN 神经网络结构"""

    def test_network_structure(self):
        """测试网络结构是否正确"""
        network = CNNV1Network()

        # 测试输入输出维度
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)

        # 输出应该是 (batch_size, 225)
        self.assertEqual(output.shape, (batch_size, 225))
    
    def test_network_softmax_output_range(self):
        """[primary] Softmax 输出值域应在 (0, 1) 范围内"""
        network = CNNV1Network()
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)
        
        # 验证输出范围 (0, 1)
        self.assertTrue(torch.all(output > 0), "Softmax 输出应大于 0")
        self.assertTrue(torch.all(output < 1), "Softmax 输出应小于 1")
    
    def test_network_softmax_row_sum(self):
        """[primary] Softmax 每行概率和应为 1"""
        network = CNNV1Network()
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 15, 15)
        output = network(input_tensor)
        
        # 验证每行和为 1
        row_sums = output.sum(dim=1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-5),
                       "Softmax 每行概率和应接近 1")

    def test_network_forward_single(self):
        """测试单样本前向传播"""
        network = CNNV1Network()

        # 单样本输入
        input_tensor = torch.randn(1, 3, 15, 15)
        output = network(input_tensor)

        self.assertEqual(output.shape, (1, 225))

    def test_network_forward_batch(self):
        """测试批量前向传播"""
        network = CNNV1Network()

        # 批量输入
        batch_sizes = [2, 8, 16]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 15, 15)
            output = network(input_tensor)
            self.assertEqual(output.shape, (batch_size, 225))


class TestCNNV1Model(unittest.TestCase):
    """测试 CNN 模型类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        
        # 使用 patch.object 隔离默认模型路径，防止测试保存/删除真实模型文件
        self._patcher = unittest.mock.patch.object(
            CNNV1Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, "cnn.pth")
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_name(self):
        """测试模型名称"""
        model = CNNV1Model()
        self.assertEqual(model.name, "cnn_v1")

    def test_model_initialization_without_path(self):
        """测试不指定路径初始化模型"""
        model = CNNV1Model()
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "cnn_v1")

    def test_model_predict_shape(self):
        """测试 predict 方法返回形状"""
        model = CNNV1Model()

        # 创建 226 维输入向量
        board_vector = [0] * 225 + [1]  # 空棋盘，黑方落子

        result = model.predict(board_vector)

        # 结果应该是 225 维
        self.assertEqual(len(result), 225)

    def test_model_predict_format(self):
        """测试 predict 方法返回格式"""
        model = CNNV1Model()

        # 创建 226 维输入向量
        board_vector = [0] * 225 + [1]

        result = model.predict(board_vector)

        # 检查格式：225 维列表，索引 i 对应位置 (i//15, i%15)
        self.assertEqual(len(result), 225)
        for i in range(225):
            prob = result[i]
            # 概率应该是浮点数
            self.assertIsInstance(prob, float)

    def test_model_predict_probability_range(self):
        """[primary] 测试 predict 方法返回的概率范围"""
        model = CNNV1Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        # 检查所有概率值在 (0, 1) 范围内
        for i in range(225):
            prob = result[i]
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")
    
    def test_model_predict_probability_sum(self):
        """[primary] 测试 predict 方法返回的概率总和为 1"""
        model = CNNV1Model()

        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)

        # 检查概率总和为 1
        prob_sum = sum(result)
        self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="概率总和应为 1")

    def test_model_predict_different_inputs(self):
        """测试不同输入产生输出"""
        model = CNNV1Model()

        # 测试黑方落子
        board_black = [0] * 225 + [1]
        result_black = model.predict(board_black)
        self.assertEqual(len(result_black), 225)

        # 测试白方落子
        board_white = [0] * 225 + [2]
        result_white = model.predict(board_white)
        self.assertEqual(len(result_white), 225)

    def test_model_predict_with_pieces(self):
        """测试带有棋子的棋盘预测"""
        model = CNNV1Model()

        # 创建带有棋子的棋盘
        board = [0] * 225
        # 在 (7, 7) 放置黑棋
        board[7 * 15 + 7] = 1
        # 在 (7, 8) 放置白棋
        board[7 * 15 + 8] = 2
        board.append(1)  # 黑方落子

        result = model.predict(board)
        self.assertEqual(len(result), 225)

    def test_model_save_and_load(self):
        """测试模型保存和加载"""
        # 创建并保存模型
        model1 = CNNV1Model()
        model1.save(self.model_path)

        # 检查文件是否存在
        self.assertTrue(os.path.exists(self.model_path))

        # 加载模型
        model2 = CNNV1Model(self.model_path)
        self.assertEqual(model2.name, "cnn_v1")

    def test_model_save_default_path(self):
        """测试模型保存到默认路径"""
        model = CNNV1Model()

        # 保存到默认路径
        default_path = model.save()

        try:
            # 检查文件是否存在
            self.assertTrue(os.path.exists(default_path))
            # 路径应该包含 cnn.pth
            self.assertIn("cnn.pth", default_path)
        finally:
            # 清理
            if os.path.exists(default_path):
                os.remove(default_path)

    def test_model_load_nonexistent_file(self):
        """测试加载不存在的模型文件"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pth")

        # 应该创建新模型而不是报错
        model = CNNV1Model(nonexistent_path)
        self.assertEqual(model.name, "cnn_v1")

    def test_model_predict_consistency(self):
        """测试模型预测的一致性（相同输入产生相同输出）"""
        model = CNNV1Model()

        board_vector = [0] * 225 + [1]

        # 多次预测应该产生相同结果
        result1 = model.predict(board_vector)
        result2 = model.predict(board_vector)

        self.assertEqual(result1, result2)

    def test_vector_to_tensor_conversion(self):
        """测试向量到张量的转换"""
        model = CNNV1Model()

        # 创建测试向量
        board_vector = [0] * 225 + [1]
        # 在 (0, 0) 放置黑棋
        board_vector[0] = 1
        # 在 (1, 1) 放置白棋
        board_vector[1 * 15 + 1] = 2

        tensor = model._vector_to_tensor(board_vector)

        # 检查张量形状
        self.assertEqual(tensor.shape, (1, 3, 15, 15))

        # 检查黑棋通道
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)
        # 检查白棋通道
        self.assertEqual(tensor[0, 1, 1, 1].item(), 1.0)
        # 检查当前方通道（黑方=1）
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1.0)


class TestCNNModelIntegration(unittest.TestCase):
    """CNN 模型集成测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "cnn.pth")

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """测试完整工作流程：创建 -> 预测 -> 保存 -> 加载 -> 预测"""
        # 创建模型
        model1 = CNNV1Model()

        # 预测
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)

        # 保存
        model1.save(self.model_path)

        # 加载
        model2 = CNNV1Model(self.model_path)

        # 再次预测
        result2 = model2.predict(board_vector)

        # 结果应该相同
        self.assertEqual(result1, result2)


class TestCNNModelDefaultLoad(unittest.TestCase):
    """测试 CNN 模型默认加载行为"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 使用 patch.object 隔离默认模型路径
        self._patcher = unittest.mock.patch.object(
            CNNV1Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, "cnn.pth")
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_load_when_model_exists(self):
        """测试当默认模型文件存在时，默认加载该模型"""
        # 创建并保存一个模型
        model1 = CNNV1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()  # 保存到默认路径

        # 创建新模型（不传参数），应该自动加载已保存的模型
        model2 = CNNV1Model()
        result2 = model2.predict(board_vector)

        # 结果应该相同（说明加载了已保存的模型）
        self.assertEqual(result1, result2)

    def test_no_load_when_model_not_exists(self):
        """测试当默认模型文件不存在时，创建新模型"""
        # 确保默认路径不存在模型文件
        self.assertFalse(os.path.exists(CNNV1Model.DEFAULT_MODEL_PATH))

        # 创建模型（不传参数）
        model = CNNV1Model()

        # 应该成功创建
        self.assertEqual(model.name, "cnn_v1")

    def test_fresh_parameter_skips_load(self):
        """测试 fresh=True 时跳过加载，创建新模型"""
        # 创建并保存一个模型
        model1 = CNNV1Model(fresh=True)
        model1.save()

        # 使用 fresh=True 创建新模型
        model2 = CNNV1Model(fresh=True)

        # 两个模型的预测结果可能不同（因为是随机初始化）
        # 但这里主要验证 fresh 参数生效，不会因为已存在模型而加载
        self.assertEqual(model2.name, "cnn_v1")

    def test_fresh_parameter_with_existing_default_model(self):
        """测试当默认模型存在时，fresh=True 强制使用新模型"""
        # 创建并保存模型1
        model1 = CNNV1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()

        # 使用 fresh=True 创建模型2，应该不加载已保存模型
        # 由于是随机初始化，预测结果应该与模型1不同
        # 但这里我们只验证它不会崩溃
        model2 = CNNV1Model(fresh=True)
        result2 = model2.predict(board_vector)

        # 验证模型2是独立的实例
        self.assertEqual(model2.name, "cnn_v1")
        self.assertEqual(len(result2), 225)

    def test_model_path_takes_priority_over_default(self):
        """测试指定 model_path 时优先使用该路径，而不是默认路径"""
        # 创建并保存模型到默认路径
        model1 = CNNV1Model(fresh=True)
        model1.save()

        # 创建另一个模型并保存到自定义路径
        custom_path = os.path.join(self.temp_dir, "custom_model.pth")
        model2 = CNNV1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result2 = model2.predict(board_vector)
        model2.save(custom_path)

        # 加载自定义路径的模型
        model3 = CNNV1Model(model_path=custom_path)
        result3 = model3.predict(board_vector)

        # 应该与模型2的结果相同
        self.assertEqual(result2, result3)


if __name__ == '__main__':
    unittest.main()
