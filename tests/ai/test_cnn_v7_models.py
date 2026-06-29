"""CNN v7 双头模型测试

验证 cnn_v7 的网络结构、双头输出、模型行为。
cnn_v7 在 cnn_v6 残差网络基础上增加价值头，形成策略+价值双头输出。

Mixin 从 test_cnn_models.py 复制，避免跨文件导入依赖性。
"""

import os
import sys
import unittest
import unittest.mock
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from ai.models.cnn_v7.model import CNNV7Model, CNNV7Network
from ai.base_model import board_vector_to_cnn_tensor


# ---------------------------------------------------------------------------
# Mixin：从 test_cnn_models.py 复制，保持测试行为一致
# ---------------------------------------------------------------------------

class CNNNetworkSharedMixin:
    """所有 CNN Network 类共有的行为测试。"""

    NETWORK_CLS = None  # 子类覆盖

    def test_network_structure(self):
        """前向输出形状为 (batch, 225)"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertEqual(output.shape, (4, 225))

    def test_network_log_softmax_output_range(self):
        """[primary] LogSoftmax 输出值 <= 0，exp 后在 (0, 1) 范围内"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertTrue(torch.all(output <= 0), "LogSoftmax 输出应全部 <= 0")
        prob = output.exp()
        self.assertTrue(torch.all(prob > 0), "exp(log-probability) 应大于 0")
        self.assertTrue(torch.all(prob < 1), "exp(log-probability) 应小于 1")

    def test_network_log_softmax_row_sum(self):
        """[primary] LogSoftmax 输出的 exp 值每行和为 1"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
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
    """所有 CNN Model 类共有的行为测试。"""

    MODEL_CLS = None  # 子类覆盖
    MODEL_NAME = None
    DEFAULT_FILENAME = None

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        self._patcher = unittest.mock.patch.object(
            self.MODEL_CLS, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, self.DEFAULT_FILENAME)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_name(self):
        """模型 name 属性"""
        model = self.MODEL_CLS(fresh=True)
        self.assertEqual(model.name, self.MODEL_NAME)

    def test_model_predict_shape(self):
        """predict 返回 ModelPredict 对象，policy 长度 225"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        self.assertEqual(len(result.policy), 225)

    def test_model_predict_format(self):
        """predict 返回 policy 元素均为 float"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        for prob in result.policy:
            self.assertIsInstance(prob, float)

    def test_model_predict_probability_range(self):
        """[primary] predict 返回 policy 概率均在 (0, 1) 内"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        for prob in result.policy:
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] predict 返回 policy 概率总和为 1"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        self.assertAlmostEqual(sum(result.policy), 1.0, places=5, msg="概率总和应为 1")

    def test_model_predict_different_inputs(self):
        """黑方/白方输入均能正常输出"""
        model = self.MODEL_CLS(fresh=True)
        result_black = model.predict([0] * 225 + [1])
        result_white = model.predict([0] * 225 + [2])
        self.assertEqual(len(result_black.policy), 225)
        self.assertEqual(len(result_white.policy), 225)

    def test_model_predict_with_pieces(self):
        """带有棋子的棋盘输入"""
        model = self.MODEL_CLS(fresh=True)
        board = [0] * 225
        board[7 * 15 + 7] = 1
        board[7 * 15 + 8] = 2
        board.append(1)
        result = model.predict(board)
        self.assertEqual(len(result.policy), 225)

    def test_model_predict_consistency(self):
        """相同输入产生相同输出"""
        model = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        self.assertEqual(model.predict(board_vector).policy,
                         model.predict(board_vector).policy)

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

    def test_default_load_when_model_exists(self):
        """不传参数时创建随机初始化模型，DEFAULT_MODEL_PATH 不再自动加载"""
        model1 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()
        # 不传参应创建随机模型，不会自动加载
        model2 = self.MODEL_CLS()
        result2 = model2.predict(board_vector)
        self.assertNotEqual(result1.policy, result2.policy,
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
        model1.save()
        custom_path = os.path.join(self.temp_dir, "custom_model.pth")
        model2 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result2 = model2.predict(board_vector)
        model2.save(custom_path)
        model3 = self.MODEL_CLS(model_path=custom_path)
        self.assertEqual(result2.policy, model3.predict(board_vector).policy)

    def test_vector_to_tensor_conversion(self):
        """base_model.board_vector_to_cnn_tensor 通道布局正确"""
        board_vector = [0] * 225 + [1]
        board_vector[0] = 1
        board_vector[1 * 15 + 1] = 2
        tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
        self.assertEqual(tensor.shape, (1, 3, 15, 15))
        self.assertEqual(tensor[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(tensor[0, 1, 1, 1].item(), 1.0)
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1.0)

    def test_full_workflow(self):
        """创建 → 预测 → 保存 → 加载 → 再预测，结果一致"""
        model1 = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save(self.model_path)
        model2 = self.MODEL_CLS(self.model_path)
        self.assertEqual(result1.policy, model2.predict(board_vector).policy)


# ---------------------------------------------------------------------------
# CNN v7 测试
# ---------------------------------------------------------------------------

class TestCNNV7Network(CNNNetworkSharedMixin, unittest.TestCase):
    """CNN v7 网络结构测试

    cnn_v7 双头残差网络：在 cnn_v6 残差网络基础上增加价值头。
    """

    NETWORK_CLS = CNNV7Network

    # --- 覆盖 Mixin：forward 返回 (policy, value) 元组 ---

    def test_network_structure(self):
        """forward 返回 (policy, value) 元组，policy 形状 (batch, 225)"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertIsInstance(output, tuple, "forward 应返回元组")
        self.assertEqual(len(output), 2, "forward 应返回 (policy, value) 元组")
        policy, value = output
        self.assertEqual(policy.shape, (4, 225), "policy 形状应为 (batch, 225)")
        self.assertEqual(value.shape, (4, 1), "value 形状应为 (batch, 1)")

    def test_network_log_softmax_output_range(self):
        """[primary] policy 输出 LogSoftmax 值 <= 0，value 在 [-1, 1] 内"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        policy, value = network(input_tensor)
        self.assertTrue(torch.all(policy <= 0), "LogSoftmax 输出应全部 <= 0")
        prob = policy.exp()
        self.assertTrue(torch.all(prob > 0), "exp(log-probability) 应大于 0")
        self.assertTrue(torch.all(prob < 1), "exp(log-probability) 应小于 1")
        self.assertTrue(torch.all(value >= -1), "value 应 >= -1")
        self.assertTrue(torch.all(value <= 1), "value 应 <= 1")

    def test_network_log_softmax_row_sum(self):
        """[primary] policy 的 exp 值每行和为 1"""
        network = self.NETWORK_CLS()
        input_tensor = torch.randn(4, 3, 15, 15)
        policy, _ = network(input_tensor)
        row_sums = policy.exp().sum(dim=1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones(4), atol=1e-5),
            "exp(LogSoftmax) 每行概率和应接近 1"
        )

    def test_network_forward_single(self):
        """单样本前向"""
        network = self.NETWORK_CLS()
        policy, value = network(torch.randn(1, 3, 15, 15))
        self.assertEqual(policy.shape, (1, 225))
        self.assertEqual(value.shape, (1, 1))

    def test_network_forward_batch(self):
        """不同 batch size 前向"""
        network = self.NETWORK_CLS()
        for batch_size in (2, 8, 16):
            with self.subTest(batch_size=batch_size):
                policy, value = network(torch.randn(batch_size, 3, 15, 15))
                self.assertEqual(policy.shape, (batch_size, 225))
                self.assertEqual(value.shape, (batch_size, 1))

    # --- v7 特有测试 ---

    def test_network_has_bn(self):
        """网络中至少存在 1 个 BatchNorm2d 层"""
        network = self.NETWORK_CLS()
        bn_modules = [m for m in network.modules()
                      if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(len(bn_modules), 0, "cnn_v7 应包含 BatchNorm2d 层")

    def test_network_no_dropout(self):
        """网络中不包含 Dropout 层"""
        network = self.NETWORK_CLS()
        for module in network.modules():
            self.assertNotIsInstance(
                module, nn.Dropout,
                f"不应包含 Dropout 层，但发现 {type(module).__name__}"
            )

    def test_network_fc_only_value_head(self):
        """价值头包含 value_fc1(225→128) 和 value_fc2(128→1) 两个 Linear 层"""
        network = self.NETWORK_CLS()
        linear_modules = [m for m in network.modules()
                          if isinstance(m, nn.Linear)]
        self.assertEqual(len(linear_modules), 2,
                         f"价值头应有 value_fc1 和 value_fc2 两个 Linear 层，实际 {len(linear_modules)} 个")
        self.assertTrue(hasattr(network, 'value_fc1'), "网络应有 value_fc1 属性")
        self.assertTrue(hasattr(network, 'value_fc2'), "网络应有 value_fc2 属性")
        self.assertEqual(network.value_fc1.in_features, 225)
        self.assertEqual(network.value_fc1.out_features, 128)
        self.assertEqual(network.value_fc2.in_features, 128)
        self.assertEqual(network.value_fc2.out_features, 1)

    def test_network_bn_count(self):
        """BatchNorm2d 层数量精确为 5"""
        network = self.NETWORK_CLS()
        bn_count = sum(1 for m in network.modules()
                       if isinstance(m, nn.BatchNorm2d))
        self.assertEqual(bn_count, 5,
                         f"应包含 5 个 BatchNorm2d 层，实际 {bn_count} 个")

    def test_network_conv_channels(self):
        """卷积层通道数：3→32→32→32→32→32，输出 32→1(1×1)"""
        network = self.NETWORK_CLS()
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 32)
        for name in ['conv2', 'conv3', 'conv4', 'conv5']:
            conv = getattr(network, name)
            self.assertEqual(conv.in_channels, 32, f"{name}.in_channels 应为 32")
            self.assertEqual(conv.out_channels, 32, f"{name}.out_channels 应为 32")
        self.assertEqual(network.conv_out.in_channels, 32)
        self.assertEqual(network.conv_out.out_channels, 1)
        self.assertEqual(network.conv_out.kernel_size, (1, 1))

    def test_network_param_count(self):
        """参数量在 67,000～68,000 之间（含双 FC 价值头，精确值 67,298）"""
        total = sum(p.numel() for p in self.NETWORK_CLS().parameters())
        self.assertGreaterEqual(total, 67_000, f"参数量 {total} 应 >= 67,000")
        self.assertLessEqual(total, 68_000, f"参数量 {total} 应 <= 68,000")


class TestCNNV7Model(CNNModelSharedMixin, unittest.TestCase):
    """CNN v7 模型行为测试

    继承 CNNModelSharedMixin 的通用测试，外加 v7 特有的价值头测试。
    predict 返回 (List[float], float) 元组，覆盖 Mixin 中相应测试。
    """

    MODEL_CLS = CNNV7Model
    MODEL_NAME = "cnn_v7"
    DEFAULT_FILENAME = "cnn_v7.pth"

    # --- 覆盖 Mixin：predict 返回 ModelPredict(value=...) ---

    def test_model_predict_shape(self):
        """predict 返回 ModelPredict 对象，策略长度 225，包含 value"""
        model = self.MODEL_CLS(fresh=True)
        result = model.predict([0] * 225 + [1])
        self.assertEqual(result.type, "policy_value", "cnn_v7 应为 policy_value 类型")
        self.assertEqual(len(result.policy), 225, "策略长度应为 225")
        self.assertIsInstance(result.value, float, "价值应为 float")

    def test_model_predict_format(self):
        """predict 返回策略元素均为 float"""
        model = self.MODEL_CLS(fresh=True)
        policy, _ = model.predict([0] * 225 + [1])
        for prob in policy:
            self.assertIsInstance(prob, float)

    def test_model_predict_probability_range(self):
        """[primary] predict 返回策略概率均在 (0, 1) 内"""
        model = self.MODEL_CLS(fresh=True)
        policy, _ = model.predict([0] * 225 + [1])
        for prob in policy:
            self.assertGreater(prob, 0.0, "概率应大于 0")
            self.assertLess(prob, 1.0, "概率应小于 1")

    def test_model_predict_probability_sum(self):
        """[primary] predict 返回策略概率总和为 1"""
        model = self.MODEL_CLS(fresh=True)
        policy, _ = model.predict([0] * 225 + [1])
        self.assertAlmostEqual(sum(policy), 1.0, places=5, msg="概率总和应为 1")

    def test_model_predict_different_inputs(self):
        """黑方/白方输入均能正常输出"""
        model = self.MODEL_CLS(fresh=True)
        policy_black, value_black = model.predict([0] * 225 + [1])
        policy_white, value_white = model.predict([0] * 225 + [2])
        self.assertEqual(len(policy_black), 225)
        self.assertEqual(len(policy_white), 225)
        self.assertIsInstance(value_black, float)
        self.assertIsInstance(value_white, float)

    def test_model_predict_with_pieces(self):
        """带有棋子的棋盘输入"""
        model = self.MODEL_CLS(fresh=True)
        board = [0] * 225
        board[7 * 15 + 7] = 1
        board[7 * 15 + 8] = 2
        board.append(1)
        policy, value = model.predict(board)
        self.assertEqual(len(policy), 225)
        self.assertIsInstance(value, float)

    def test_model_predict_value_range(self):
        """predict 返回价值在 [-1, 1] 范围内"""
        model = self.MODEL_CLS(fresh=True)
        _, value = model.predict([0] * 225 + [1])
        self.assertGreaterEqual(value, -1.0, "价值应 >= -1")
        self.assertLessEqual(value, 1.0, "价值应 <= 1")

    def test_model_predict_consistency(self):
        """相同输入产生相同输出"""
        model = self.MODEL_CLS(fresh=True)
        board_vector = [0] * 225 + [1]
        policy1, value1 = model.predict(board_vector)
        policy2, value2 = model.predict(board_vector)
        self.assertEqual(policy1, policy2, "相同输入策略应一致")
        self.assertEqual(value1, value2, "相同输入价值应一致")


if __name__ == '__main__':
    unittest.main()
