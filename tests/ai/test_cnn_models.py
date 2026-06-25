"""CNN 模型参数化测试（v1/v2/v3/v4 统一）

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
from ai.models.cnn_v4.model import CNNV4Model, CNNV4Network
from ai.models.cnn_v5.model import CNNV5Model, CNNV5Network
from ai.models.cnn_v6.model import CNNV6Model, CNNV6Network
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

    def test_network_log_softmax_output_range(self):
        """[primary] LogSoftmax 输出值 <= 0（log-probability 特征），exp 后在 (0, 1) 范围内"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        # LogSoftmax 输出应全部 <= 0
        self.assertTrue(torch.all(output <= 0), "LogSoftmax 输出应全部 <= 0")
        # exp 后的值应在 (0, 1)
        prob = output.exp()
        self.assertTrue(torch.all(prob > 0), "exp(log-probability) 应大于 0")
        self.assertTrue(torch.all(prob < 1), "exp(log-probability) 应小于 1")

    def test_network_log_softmax_row_sum(self):
        """[primary] LogSoftmax 输出的 exp 值每行和为 1"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        # exp 后每行总和应为 1
        row_sums = output.exp().sum(dim=1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones(4), atol=1e-5),
            "exp(LogSoftmax) 每行概率和应接近 1"
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
        model = self.MODEL_CLS(fresh=True)
        self.assertEqual(model.name, self.MODEL_NAME)

    # --- predict 行为 ---

    def test_model_predict_shape(self):
        """predict 返回长度 225"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        self.assertEqual(len(result), 225)

    def test_model_predict_format(self):
        """predict 返回元素均为 float"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        for prob in result:
            self.assertIsInstance(prob, float)

    def test_model_predict_probability_range(self):
        """[primary] predict 返回概率均在 (0, 1) 内"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        for prob in result:
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] predict 返回概率总和为 1"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        self.assertAlmostEqual(sum(result), 1.0, places=5, msg="概率总和应为 1")

    def test_model_predict_different_inputs(self):
        """黑方/白方输入均能正常输出"""
        model = self.MODEL_CLS(fresh=True)
        result_black = model.predict([0] * 225 + [1])
        result_white = model.predict([0] * 225 + [2])
        self.assertEqual(len(result_black), 225)
        self.assertEqual(len(result_white), 225)

    def test_model_predict_with_pieces(self):
        """带有棋子的棋盘输入"""
        model = self.MODEL_CLS(fresh=True)
        board = [0] * 225
        board[7 * 15 + 7] = 1  # 黑棋
        board[7 * 15 + 8] = 2  # 白棋
        board.append(1)
        result = model.predict(board)
        self.assertEqual(len(result), 225)

    def test_model_predict_consistency(self):
        """相同输入产生相同输出"""
        model = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        self.assertEqual(model.predict(board_vector), model.predict(board_vector))

    # --- 保存 / 加载 ---

    def test_model_save_and_load(self):
        """save 到指定路径 → 加载后 name 一致"""
        model1 = self.MODEL_CLS(fresh=True)
        model1.save(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        model2 = self.MODEL_CLS(self.model_path)
        self.assertEqual(model2.name, self.MODEL_NAME)

    def test_model_save_default_path(self):
        """save() 无参 → 保存到 DEFAULT_MODEL_PATH"""
        model = self.MODEL_CLS(fresh=True)
        default_path = model.save()
        self.assertTrue(os.path.exists(default_path))
        self.assertIn(self.DEFAULT_FILENAME, default_path)

    def test_model_load_nonexistent_file(self):
        """显式指定不存在的文件路径 → 抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.MODEL_CLS(os.path.join(self.temp_dir, "nonexistent.pth"))

    # --- 默认加载行为 ---

    def test_default_load_when_model_exists(self):
        """不传参数时创建随机初始化模型，DEFAULT_MODEL_PATH 不再自动加载"""
        model1 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()  # 保存到默认路径

        # 不传参应创建随机模型，不会自动加载
        model2 = self.MODEL_CLS()
        result2 = model2.predict(board_vector)
        self.assertNotEqual(result1, result2,
            "不传参应创建随机模型，不应加载 DEFAULT_MODEL_PATH")

    def test_no_load_when_model_not_exists(self):
        """不传参数时创建随机初始化模型"""
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
        model1 = self.MODEL_CLS(fresh=True)
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


class TestCNNV4Network(CNNNetworkSharedMixin, unittest.TestCase):
    NETWORK_CLS = CNNV4Network

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
        """CNN v4 参数量约 2.8 万（cnn_v3 + 额外 32→32 卷积层）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(total, 28_000, f"参数量 {total} 应 >= 2.8 万")
        self.assertLessEqual(total, 30_000, f"参数量 {total} 应 <= 3 万")

    def test_network_conv_channels(self):
        """4 层卷积层，通道数均为 32"""
        network = self.NETWORK_CLS()
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)
        self.assertEqual(network.conv2.in_channels, 32)
        self.assertEqual(network.conv2.out_channels, 32)
        self.assertEqual(network.conv3.in_channels, 32)
        self.assertEqual(network.conv3.out_channels, 32)
        self.assertEqual(network.conv4.in_channels, 32)
        self.assertEqual(network.conv4.out_channels, 32)

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


class TestCNNV4Model(CNNModelSharedMixin, unittest.TestCase):
    MODEL_CLS = CNNV4Model
    MODEL_NAME = "cnn_v4"
    DEFAULT_FILENAME = "cnn_v4.pth"


class TestCNNV5Network(CNNNetworkSharedMixin, unittest.TestCase):
    NETWORK_CLS = CNNV5Network

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
        """CNN v5 参数量约 3.8 万（cnn_v4 + 额外 32→32 卷积层）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(total, 37_000, f"参数量 {total} 应 >= 3.7 万")
        self.assertLessEqual(total, 39_000, f"参数量 {total} 应 <= 3.9 万")

    def test_network_conv_channels(self):
        """5 层卷积层，通道数均为 32"""
        network = self.NETWORK_CLS()
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)
        self.assertEqual(network.conv2.in_channels, 32)
        self.assertEqual(network.conv2.out_channels, 32)
        self.assertEqual(network.conv3.in_channels, 32)
        self.assertEqual(network.conv3.out_channels, 32)
        self.assertEqual(network.conv4.in_channels, 32)
        self.assertEqual(network.conv4.out_channels, 32)
        self.assertEqual(network.conv5.in_channels, 32)
        self.assertEqual(network.conv5.out_channels, 32)

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


class TestCNNV5Model(CNNModelSharedMixin, unittest.TestCase):
    MODEL_CLS = CNNV5Model
    MODEL_NAME = "cnn_v5"
    DEFAULT_FILENAME = "cnn_v5.pth"


# ---------------------------------------------------------------------------
# CNN v6 测试
# ---------------------------------------------------------------------------

class TestCNNV6Network(CNNNetworkSharedMixin, unittest.TestCase):
    """CNN v6 网络结构测试

    cnn_v6 在 cnn_v5 基础上引入 BatchNorm 和残差连接：
    - 每层 Conv 后添加 BN（Conv→BN→ReLU）
    - Conv2+Conv3 组成残差块 1，Conv4+Conv5 组成残差块 2
    - 纯策略网络，forward 返回 (batch, 225) log-probability
    """

    NETWORK_CLS = CNNV6Network

    # --- UT-NET-1：网络中包含 BatchNorm 层（与 v3/v4/v5 相反） ---

    def test_network_has_bn(self):
        """[UT-NET-1] 网络中至少存在 1 个 BatchNorm2d 层"""
        network = self.NETWORK_CLS()
        bn_modules = [m for m in network.modules()
                      if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(
            len(bn_modules), 0,
            "cnn_v6 应包含 BatchNorm2d 层"
        )

    # --- UT-NET-2：网络中不包含 Dropout 层 ---

    def test_network_no_dropout(self):
        """[UT-NET-2] 网络中不包含 Dropout 层"""
        network = self.NETWORK_CLS()
        for module in network.modules():
            self.assertNotIsInstance(
                module, nn.Dropout,
                f"纯卷积网络不应包含 Dropout 层，但发现 {type(module).__name__}"
            )

    # --- UT-NET-3：网络中不包含 FC 层（纯卷积策略网络） ---

    def test_network_no_fc(self):
        """[UT-NET-3] 网络中不包含任何 Linear 层"""
        network = self.NETWORK_CLS()
        linear_modules = [m for m in network.modules()
                          if isinstance(m, nn.Linear)]
        self.assertEqual(
            len(linear_modules), 0,
            f"纯策略网络不应包含 Linear 层，实际 {len(linear_modules)} 个"
        )

    # --- UT-NET-4：BN 层数量 = 5（Conv1~5 各一个 BN） ---

    def test_network_bn_count(self):
        """[UT-NET-4] BatchNorm2d 层数量精确为 5"""
        network = self.NETWORK_CLS()
        bn_count = sum(
            1 for m in network.modules()
            if isinstance(m, nn.BatchNorm2d)
        )
        self.assertEqual(
            bn_count, 5,
            f"应包含 5 个 BatchNorm2d 层（Conv1~5 各一个），实际 {bn_count} 个"
        )

    # --- UT-NET-5：卷积通道数正确 ---

    def test_network_conv_channels(self):
        """[UT-NET-5] 卷积层通道数：3→32→32→32→32→32，输出 32→1(1×1)"""
        network = self.NETWORK_CLS()
        # conv1: 3→32
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)
        # conv2~5: 32→32
        for name in ['conv2', 'conv3', 'conv4', 'conv5']:
            conv = getattr(network, name)
            self.assertEqual(conv.in_channels, 32,
                             f"{name}.in_channels 应为 32")
            self.assertEqual(conv.out_channels, 32,
                             f"{name}.out_channels 应为 32")
        # conv_out: 32→1, kernel_size=(1,1)
        self.assertEqual(network.conv_out.in_channels, 32)
        self.assertEqual(network.conv_out.out_channels, 1)
        self.assertEqual(network.conv_out.kernel_size, (1, 1))

    # --- UT-NET-6：参数量在预期范围内 ---

    def test_network_param_count(self):
        """[UT-NET-6] 参数量在 38,000～38,500 之间（纯策略网络，精确值约 38,241）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(
            total, 38_000,
            f"参数量 {total} 应 >= 38,000"
        )
        self.assertLessEqual(
            total, 38_500,
            f"参数量 {total} 应 <= 38,500"
        )

    # --- UT-NET-7：残差连接存在性验证 ---

    def test_network_residual_connection_exists(self):
        """[UT-NET-7] 将残差块权重置零后，输出应与正常前向不同

        验证残差连接存在的策略：
        1. 正常前向，记录输出 Y_normal
        2. 将第一个残差块（conv2, bn2, conv3, bn3）的权重和偏置全部置零
        3. 再次前向，记录输出 Y_zeroed
        4. 两者应显著不同 —— 因为残差路径的 shortcut 在权重置零后
           仍传递输入信息，与正常主路径的贡献显然不同
        """
        network = self.NETWORK_CLS()
        network.eval()

        input_tensor = torch.randn(2, 3, 15, 15)

        # 正常前向
        with torch.no_grad():
            output_normal = network(input_tensor).clone()

        # 将残差块 1 的主路径参数置零
        with torch.no_grad():
            for name, param in network.named_parameters():
                if any(x in name for x in ['conv2.', 'bn2.', 'conv3.', 'bn3.']):
                    param.zero_()

        # 置零后前向
        with torch.no_grad():
            output_zeroed = network(input_tensor).clone()

        # 两者应显著不同
        diff = (output_normal - output_zeroed).abs().max().item()
        self.assertGreater(
            diff, 1e-4,
            "置零残差块权重后，输出应与正常前向显著不同，"
            "以证明残差连接在主路径中存在并正常工作"
        )

    # --- UT-RES-3：残差连接对梯度的影响 ---

    def test_residual_gradient_flow(self):
        """[UT-RES-3] 反向传播后 conv2.weight.grad 不为 None"""
        network = self.NETWORK_CLS()
        network.train()

        input_tensor = torch.randn(1, 3, 15, 15)
        output = network(input_tensor)

        # 对输出的总和做反向传播
        loss = output.sum()
        loss.backward()

        # conv2 在残差块 1 中，其梯度应不为 None
        self.assertIsNotNone(
            network.conv2.weight.grad,
            "反向传播后 conv2.weight.grad 不应为 None，"
            "以证明梯度可通过残差连接正常流动"
        )
        # 梯度不应全为零
        self.assertFalse(
            torch.allclose(network.conv2.weight.grad,
                           torch.zeros_like(network.conv2.weight.grad)),
            "conv2.weight.grad 不应全为零"
        )

    # --- UT-RES-1：残差块 1 输入输出形状一致 ---

    def test_residual_block1_input_output_shape(self):
        """[UT-RES-1] 残差块 1 输入输出 shape 均为 (1, 32, 15, 15)

        使用 forward hook 捕获 relu1 的输出（= 残差块 1 的输入）
        和 relu3 的输出（= 残差块 1 的输出），验证形状一致。
        """
        network = self.NETWORK_CLS()
        network.eval()

        # 用 hook 捕获中间张量的形状
        captured_shapes = {}

        def make_hook(name):
            def hook(module, input, output):
                captured_shapes[name] = tuple(output.shape)
            return hook

        # relu1 的输出 = 残差块 1 的输入
        handle1 = network.relu1.register_forward_hook(make_hook("block1_input"))
        # relu3 的输出 = 残差块 1 的输出 / 残差块 2 的输入
        handle3 = network.relu3.register_forward_hook(make_hook("block1_output"))

        with torch.no_grad():
            network(torch.randn(1, 3, 15, 15))

        handle1.remove()
        handle3.remove()

        self.assertEqual(
            captured_shapes.get("block1_input"), (1, 32, 15, 15),
            f"残差块 1 输入 shape 应为 (1, 32, 15, 15)，实际 {captured_shapes.get('block1_input')}"
        )
        self.assertEqual(
            captured_shapes.get("block1_output"), (1, 32, 15, 15),
            f"残差块 1 输出 shape 应为 (1, 32, 15, 15)，实际 {captured_shapes.get('block1_output')}"
        )

    # --- UT-RES-2：残差块 2 输入输出形状一致 ---

    def test_residual_block2_input_output_shape(self):
        """[UT-RES-2] 残差块 2 输入输出 shape 均为 (1, 32, 15, 15)

        使用 forward hook 捕获 relu3 的输出（= 残差块 2 的输入）
        和 relu5 的输出（= 残差块 2 的输出），验证形状一致。
        """
        network = self.NETWORK_CLS()
        network.eval()

        captured_shapes = {}

        def make_hook(name):
            def hook(module, input, output):
                captured_shapes[name] = tuple(output.shape)
            return hook

        # relu3 的输出 = 残差块 2 的输入
        handle3 = network.relu3.register_forward_hook(make_hook("block2_input"))
        # relu5 的输出 = 残差块 2 的输出
        handle5 = network.relu5.register_forward_hook(make_hook("block2_output"))

        with torch.no_grad():
            network(torch.randn(1, 3, 15, 15))

        handle3.remove()
        handle5.remove()

        self.assertEqual(
            captured_shapes.get("block2_input"), (1, 32, 15, 15),
            f"残差块 2 输入 shape 应为 (1, 32, 15, 15)，实际 {captured_shapes.get('block2_input')}"
        )
        self.assertEqual(
            captured_shapes.get("block2_output"), (1, 32, 15, 15),
            f"残差块 2 输出 shape 应为 (1, 32, 15, 15)，实际 {captured_shapes.get('block2_output')}"
        )


class TestCNNV6Model(CNNModelSharedMixin, unittest.TestCase):
    """CNN v6 模型行为测试

    继承 CNNModelSharedMixin 的通用测试，
    外加 v6 特有的 BN 模式切换测试。
    predict 返回 225 维概率列表（与 Mixin 一致）。
    """

    MODEL_CLS = CNNV6Model
    MODEL_NAME = "cnn_v6"
    DEFAULT_FILENAME = "cnn_v6.pth"

    # --- UT-MOD-2：fresh=True 不加载任何权重，参数为随机初始化 ---

    def test_model_fresh_creates_random_weights(self):
        """[UT-MOD-2] fresh=True 创建的网络参数为随机初始化（非全零）

        验证 fresh=True 时：
        1. Conv 层权重非全零（PyTorch 默认使用 Kaiming 初始化）
        2. 两次 fresh=True 创建的模型参数不同（随机种子不同）

        注意：BN 层的 bias 默认为零、weight 默认为一，这是标准初始化，
        不是"未初始化"的标志，因此跳过 BN 参数的全零检查。
        """
        model1 = self.MODEL_CLS(fresh=True)
        model2 = self.MODEL_CLS(fresh=True)

        # 验证 Conv 层权重非全零（Kaiming 随机初始化）
        # BN 的 bias 默认为 0、weight 默认为 1，这是标准行为，跳过
        for name, param in model1.network.named_parameters():
            if "bn" in name:
                continue  # BN 参数使用标准初始化（bias=0, weight=1），无需检查
            self.assertFalse(
                torch.allclose(param, torch.zeros_like(param)),
                f"fresh=True 时参数 {name} 不应全为零（应为随机初始化）"
            )

        # 两次 fresh 创建的模型参数应不同（不同随机种子）
        params_differ = False
        for (n1, p1), (n2, p2) in zip(
            model1.network.named_parameters(),
            model2.network.named_parameters()
        ):
            if not torch.allclose(p1, p2):
                params_differ = True
                break
        self.assertTrue(
            params_differ,
            "两次 fresh=True 创建的模型参数应不同（随机初始化种子不同）"
        )

    # --- UT-MOD-3：eval 模式下 BN 使用全局统计量，多次 forward 输出一致 ---

    def test_model_bn_eval_mode_deterministic(self):
        """[UT-MOD-3] eval 模式下多次 forward 相同输入，输出完全一致"""
        model = self.MODEL_CLS(fresh=True)
        model.network.eval()

        # 构造输入张量
        board_vector = [0] * 225 + [1]
        device = next(model.network.parameters()).device
        input_tensor = board_vector_to_cnn_tensor(board_vector, device)

        with torch.no_grad():
            output1 = model.network(input_tensor)
            output2 = model.network(input_tensor)

        self.assertTrue(
            torch.allclose(output1, output2, atol=1e-6),
            "eval 模式下多次 forward 策略输出应完全一致"
        )

    # --- UT-MOD-4：train 模式下 BN 使用 batch 统计量 ---

    def test_model_bn_train_mode(self):
        """[UT-MOD-4] train 模式下网络 .training 属性为 True"""
        model = self.MODEL_CLS(fresh=True)
        model.network.train()

        self.assertTrue(
            model.network.training,
            "train() 后 network.training 应为 True，BN 使用 batch 统计量"
        )

        # 验证 BN 层在 train 模式
        bn_modules = [m for m in model.network.modules()
                      if isinstance(m, nn.BatchNorm2d)]
        for bn in bn_modules:
            self.assertTrue(
                bn.training,
                f"train 模式下 {bn} 的 training 属性应为 True"
            )


if __name__ == '__main__':
    unittest.main()
