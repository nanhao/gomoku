"""CNN 模型参数化测试（v1/v2/v3 统一）

按 task_reports/2026-05-01/test_suite_review.md Top #4 执行：
合并原三个文件（test_cnn_model.py / test_cnn_v2_model.py / test_cnn_v3_model.py）的
大面积重复用例（~60% 代码逐行相同）为 Mixin + 三子类结构。

共性行为统一放在 Mixin 里，三代模型各自一个子类继承 Mixin + unittest.TestCase。
版本特有的架构断言（参数量、无 BN、无 Dropout、通道数、1×1 输出）保留在各自子类。
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
import torch.nn as nn
from ai.models.cnn_v1.model import CNNV1Model, CNNV1Network
from ai.models.cnn_v2.model import CNNV2Model, CNNV2Network
from ai.models.cnn_v3.model import CNNV3Model, CNNV3Network
from ai.base_model import board_vector_to_cnn_tensor


# ---------------------------------------------------------------------------
# Mixin：所有 CNN 模型都应通过的共性行为
# ---------------------------------------------------------------------------

class CNNNetworkSharedMixin:
    """所有 CNN Network 类共有的行为测试。

    子类需设置类属性：NETWORK_CLS
    """

    NETWORK_CLS = None  # 子类覆盖为具体 Network 类

    def test_network_structure(self):
        """前向输出形状为 (batch, 225)"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertEqual(output.shape, (4, 225))

    def test_network_softmax_output_range(self):
        """[primary] Softmax 输出严格在 (0, 1) 范围内"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertTrue(torch.all(output > 0), "Softmax 输出应大于 0")
        self.assertTrue(torch.all(output < 1), "Softmax 输出应小于 1")

    def test_network_softmax_row_sum(self):
        """[primary] Softmax 每行概率和为 1"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        row_sums = output.sum(dim=1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones(4), atol=1e-5),
            "Softmax 每行概率和应接近 1"
        )

    def test_network_forward_single(self):
        """单样本前向"""
        network = self.NETWORK_CLS()
        output = network(torch.randn(1, 3, 15, 15))
        self.assertEqual(output.shape, (1, 225))

    def test_network_forward_batch(self):
        """不同 batch size 前向"""
        network = self.NETWORK_CLS()
        for batch_size in (2, 8, 16):
            with self.subTest(batch_size=batch_size):
                output = network(torch.randn(batch_size, 3, 15, 15))
                self.assertEqual(output.shape, (batch_size, 225))


class CNNModelSharedMixin:
    """所有 CNN Model 类共有的行为测试。

    子类需设置类属性：MODEL_CLS, MODEL_NAME, DEFAULT_FILENAME
    """

    MODEL_CLS = None         # 子类覆盖
    MODEL_NAME = None        # 子类覆盖（如 "cnn_v1"）
    DEFAULT_FILENAME = None  # 子类覆盖（如 "cnn.pth"）

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        # 隔离默认模型路径，防止测试保存/删除真实模型文件
        self._patcher = unittest.mock.patch.object(
            self.MODEL_CLS, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, self.DEFAULT_FILENAME)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- 基本属性 ---

    def test_model_name(self):
        """模型 name 属性"""
        model = self.MODEL_CLS()
        self.assertEqual(model.name, self.MODEL_NAME)

    # --- predict 行为 ---

    def test_model_predict_shape(self):
        """predict 返回长度 225"""
        model = self.MODEL_CLS()
        result = model.predict([0] * 225 + [1])
        self.assertEqual(len(result), 225)

    def test_model_predict_format(self):
        """predict 返回元素均为 float"""
        model = self.MODEL_CLS()
        result = model.predict([0] * 225 + [1])
        for prob in result:
            self.assertIsInstance(prob, float)

    def test_model_predict_probability_range(self):
        """[primary] predict 返回概率均在 (0, 1) 内"""
        model = self.MODEL_CLS()
        result = model.predict([0] * 225 + [1])
        for prob in result:
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] predict 返回概率总和为 1"""
        model = self.MODEL_CLS()
        result = model.predict([0] * 225 + [1])
        self.assertAlmostEqual(sum(result), 1.0, places=5, msg="概率总和应为 1")

    def test_model_predict_different_inputs(self):
        """黑方/白方输入均能正常输出"""
        model = self.MODEL_CLS()
        result_black = model.predict([0] * 225 + [1])
        result_white = model.predict([0] * 225 + [2])
        self.assertEqual(len(result_black), 225)
        self.assertEqual(len(result_white), 225)

    def test_model_predict_with_pieces(self):
        """带有棋子的棋盘输入"""
        model = self.MODEL_CLS()
        board = [0] * 225
        board[7 * 15 + 7] = 1  # 黑棋
        board[7 * 15 + 8] = 2  # 白棋
        board.append(1)
        result = model.predict(board)
        self.assertEqual(len(result), 225)

    def test_model_predict_consistency(self):
        """相同输入产生相同输出"""
        model = self.MODEL_CLS()
        board_vector = [0] * 225 + [1]
        self.assertEqual(model.predict(board_vector), model.predict(board_vector))

    # --- 保存 / 加载 ---

    def test_model_save_and_load(self):
        """save 到指定路径 → 加载后 name 一致"""
        model1 = self.MODEL_CLS()
        model1.save(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        model2 = self.MODEL_CLS(self.model_path)
        self.assertEqual(model2.name, self.MODEL_NAME)

    def test_model_save_default_path(self):
        """save() 无参 → 保存到 DEFAULT_MODEL_PATH"""
        model = self.MODEL_CLS()
        default_path = model.save()
        self.assertTrue(os.path.exists(default_path))
        self.assertIn(self.DEFAULT_FILENAME, default_path)

    def test_model_load_nonexistent_file(self):
        """加载不存在的文件 → 创建新模型而不报错"""
        model = self.MODEL_CLS(os.path.join(self.temp_dir, "nonexistent.pth"))
        self.assertEqual(model.name, self.MODEL_NAME)

    # --- 默认加载行为 ---

    def test_default_load_when_model_exists(self):
        """默认路径存在时自动加载（相同输入给相同输出）"""
        model1 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()  # 保存到默认路径

        model2 = self.MODEL_CLS()  # 不传参，应自动加载
        result2 = model2.predict(board_vector)
        self.assertEqual(result1, result2)

    def test_no_load_when_model_not_exists(self):
        """默认路径不存在时创建新模型"""
        self.assertFalse(os.path.exists(self.MODEL_CLS.DEFAULT_MODEL_PATH))
        model = self.MODEL_CLS()
        self.assertEqual(model.name, self.MODEL_NAME)

    def test_fresh_parameter_skips_load(self):
        """fresh=True 跳过加载"""
        model1 = self.MODEL_CLS(fresh=True)
        model1.save()
        model2 = self.MODEL_CLS(fresh=True)
        self.assertEqual(model2.name, self.MODEL_NAME)

    def test_model_path_takes_priority_over_default(self):
        """显式传入 model_path 时优先使用该路径"""
        model1 = self.MODEL_CLS(fresh=True)
        model1.save()  # 保存到默认路径

        custom_path = os.path.join(self.temp_dir, "custom_model.pth")
        model2 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result2 = model2.predict(board_vector)
        model2.save(custom_path)

        model3 = self.MODEL_CLS(model_path=custom_path)
        self.assertEqual(result2, model3.predict(board_vector))

    # --- 向量 → 张量转换（三代统一走 base_model.board_vector_to_cnn_tensor）---

    def test_vector_to_tensor_conversion(self):
        """base_model.board_vector_to_cnn_tensor 通道布局正确"""
        board_vector = [0] * 225 + [1]
        board_vector[0] = 1
        board_vector[1 * 15 + 1] = 2
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))

        self.assertEqual(tensor.shape, (1, 3, 15, 15))
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)  # 黑棋通道
        self.assertEqual(tensor[0, 1, 1, 1].item(), 1.0)  # 白棋通道
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1.0)  # 当前方通道（黑=1）

    # --- 完整工作流 ---

    def test_full_workflow(self):
        """创建 → 预测 → 保存 → 加载 → 再预测，结果一致"""
        model1 = self.MODEL_CLS()
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)

        model1.save(self.model_path)
        model2 = self.MODEL_CLS(self.model_path)
        self.assertEqual(result1, model2.predict(board_vector))


# ---------------------------------------------------------------------------
# 三代具体子类
# ---------------------------------------------------------------------------

class TestCNNV1Network(CNNNetworkSharedMixin, unittest.TestCase):
    NETWORK_CLS = CNNV1Network


class TestCNNV2Network(CNNNetworkSharedMixin, unittest.TestCase):
    NETWORK_CLS = CNNV2Network

    def test_network_param_count(self):
        """CNN v2 参数量约 101 万（3层64通道轻量化）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(total, 1_000_000, f"参数量 {total} 应 >= 100 万")
        self.assertLessEqual(total, 1_050_000, f"参数量 {total} 应 <= 105 万")


class TestCNNV3Network(CNNNetworkSharedMixin, unittest.TestCase):
    NETWORK_CLS = CNNV3Network

    def test_network_no_bn(self):
        """网络中不包含 BatchNorm 层"""
        for module in self.NETWORK_CLS().modules():
            self.assertNotIsInstance(
                module, (nn.BatchNorm1d, nn.BatchNorm2d),
                f"不应包含 BatchNorm 层，但发现 {type(module).__name__}"
            )

    def test_network_no_dropout(self):
        """网络中不包含 Dropout 层"""
        for module in self.NETWORK_CLS().modules():
            self.assertNotIsInstance(
                module, nn.Dropout,
                f"不应包含 Dropout 层，但发现 {type(module).__name__}"
            )

    def test_network_param_count(self):
        """CNN v3 参数量约 1.9 万（纯卷积，1×1 Conv 替代 FC）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(total, 19_000, f"参数量 {total} 应 >= 1.9 万")
        self.assertLessEqual(total, 20_000, f"参数量 {total} 应 <= 2 万")

    def test_network_conv_channels(self):
        """卷积层通道数为 32"""
        network = self.NETWORK_CLS()
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)
        self.assertEqual(network.conv2.in_channels, 32)
        self.assertEqual(network.conv2.out_channels, 32)
        self.assertEqual(network.conv3.in_channels, 32)
        self.assertEqual(network.conv3.out_channels, 32)

    def test_network_fc_direct_output(self):
        """无 FC 层，输出层为 1×1 卷积(32→1)"""
        network = self.NETWORK_CLS()
        for module in network.modules():
            self.assertNotIsInstance(
                module, nn.Linear,
                f"纯卷积网络不应包含 Linear 层，但发现 {type(module).__name__}"
            )
        self.assertEqual(network.conv_out.in_channels, 32)
        self.assertEqual(network.conv_out.out_channels, 1)
        self.assertEqual(network.conv_out.kernel_size, (1, 1))


class TestCNNV1Model(CNNModelSharedMixin, unittest.TestCase):
    MODEL_CLS = CNNV1Model
    MODEL_NAME = "cnn_v1"
    DEFAULT_FILENAME = "cnn.pth"


class TestCNNV2Model(CNNModelSharedMixin, unittest.TestCase):
    MODEL_CLS = CNNV2Model
    MODEL_NAME = "cnn_v2"
    DEFAULT_FILENAME = "cnn_v2.pth"


class TestCNNV3Model(CNNModelSharedMixin, unittest.TestCase):
    MODEL_CLS = CNNV3Model
    MODEL_NAME = "cnn_v3"
    DEFAULT_FILENAME = "cnn_v3.pth"


if __name__ == '__main__':
    unittest.main()
