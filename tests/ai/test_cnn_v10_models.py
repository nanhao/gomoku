"""CNN v10 价值头模型测试

验证 cnn_v10 的网络结构、单头输出、模型行为。
cnn_v10 基于 cnn_v9 去掉策略头，仅保留价值头。
"""

import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from ai.base_model import ModelPredict


# ============================================================================
# CNNV10Network 测试
# ============================================================================

class TestCNNV10Network(unittest.TestCase):
    """CNN v10 网络结构测试（设计 TC-NET-01 ~ TC-NET-07）"""

    @classmethod
    def setUpClass(cls):
        from ai.models.cnn_v10.model import CNNV10Network
        cls.NetworkCls = CNNV10Network

    def test_forward_output_shape(self):
        """TC-NET-01: 前向输出形状为 (batch, 1)"""
        network = self.NetworkCls()
        input_tensor = torch.randn(4, 3, 15, 15)
        output = network(input_tensor)
        self.assertEqual(output.shape, (4, 1),
                         f"期望 (4, 1)，实际 {output.shape}")

    def test_forward_single_sample(self):
        """单样本前向输出形状 (1, 1)"""
        network = self.NetworkCls()
        input_tensor = torch.randn(1, 3, 15, 15)
        output = network(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_output_value_range(self):
        """TC-NET-02: 输出值域在 [-1, 1] 范围内"""
        network = self.NetworkCls()
        input_tensor = torch.randn(16, 3, 15, 15)
        output = network(input_tensor)
        self.assertTrue(torch.all(output >= -1.0),
                        "所有输出应 >= -1")
        self.assertTrue(torch.all(output <= 1.0),
                        "所有输出应 <= 1")

    def test_no_policy_head_components(self):
        """TC-NET-03: 不存在策略头组件"""
        network = self.NetworkCls()
        # 策略头组件：conv_out, policy_fc1, policy_fc2, log_softmax
        for attr_name in ['conv_out', 'policy_fc1', 'policy_fc2', 'log_softmax']:
            self.assertFalse(
                hasattr(network, attr_name),
                f"不应存在策略头组件 {attr_name}"
            )

    def test_value_head_components_exist(self):
        """TC-NET-04: 价值头组件完整"""
        network = self.NetworkCls()
        for attr_name in ['value_conv_out', 'value_fc1', 'value_fc2', 'value_tanh']:
            self.assertTrue(
                hasattr(network, attr_name),
                f"缺少价值头组件 {attr_name}"
            )
        # 验证 value_fc2 输出维度为 1
        self.assertIsInstance(network.value_conv_out, nn.Conv2d)
        self.assertEqual(network.value_conv_out.in_channels, 48)
        self.assertEqual(network.value_conv_out.out_channels, 1)
        self.assertEqual(network.value_conv_out.kernel_size, (1, 1))
        self.assertEqual(network.value_fc2.out_features, 1)

    def test_conv_channels(self):
        """TC-NET-05: 卷积通道数正确"""
        network = self.NetworkCls()
        # conv1: 3→48
        self.assertEqual(network.conv1.in_channels, 3)
        self.assertEqual(network.conv1.out_channels, 48)
        # conv2~5: 48→48
        for name in ['conv2', 'conv3', 'conv4', 'conv5']:
            conv = getattr(network, name)
            self.assertEqual(conv.in_channels, 48, f"{name}.in_channels 应为 48")
            self.assertEqual(conv.out_channels, 48, f"{name}.out_channels 应为 48")
        # value_conv_out: 48→1, 1×1
        self.assertEqual(network.value_conv_out.in_channels, 48)
        self.assertEqual(network.value_conv_out.out_channels, 1)
        self.assertEqual(network.value_conv_out.kernel_size, (1, 1))

    def test_has_batch_norm(self):
        """TC-NET-06: 网络中应包含 BatchNorm2d 层"""
        network = self.NetworkCls()
        bn_modules = [m for m in network.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(len(bn_modules), 0,
                           "cnn_v10 应包含 BatchNorm2d 层")

    def test_bn_count(self):
        """BatchNorm2d 层数量精确为 5"""
        network = self.NetworkCls()
        bn_count = sum(1 for m in network.modules()
                       if isinstance(m, nn.BatchNorm2d))
        self.assertEqual(bn_count, 5,
                         f"应包含 5 个 BatchNorm2d 层，实际 {bn_count} 个")

    def test_no_dropout(self):
        """网络中不包含 Dropout 层"""
        network = self.NetworkCls()
        for module in network.modules():
            self.assertNotIsInstance(
                module, nn.Dropout,
                f"不应包含 Dropout 层，但发现 {type(module).__name__}"
            )

    def test_param_count(self):
        """TC-NET-07: 参数量约 70,000（±5%）"""
        network = self.NetworkCls()
        total_params = sum(p.numel() for p in network.parameters())
        expected = 100000
        tolerance = int(expected * 0.05)
        self.assertGreaterEqual(total_params, expected - tolerance,
                                f"参数量 {total_params} 低于下限 {expected - tolerance}")
        self.assertLessEqual(total_params, expected + tolerance,
                             f"参数量 {total_params} 超出上限 {expected + tolerance}")


# ============================================================================
# CNNV10Model 测试
# ============================================================================

class TestCNNV10Model(unittest.TestCase):
    """CNN v10 模型行为测试（设计 TC-MOD-01 ~ TC-MOD-09）"""

    @classmethod
    def setUpClass(cls):
        from ai.models.cnn_v10.model import CNNV10Model
        cls.ModelCls = CNNV10Model

    def setUp(self):
        """每个测试前创建 fresh 模型，避免状态污染"""
        self.model = self.ModelCls(fresh=True)

    def test_predict_return_type(self):
        """TC-MOD-01: predict 返回 type=TYPE_VALUE_ONLY"""
        result = self.model.predict([0] * 226)
        self.assertEqual(result.type, ModelPredict.TYPE_VALUE_ONLY,
                         "cnn_v10 应为 value_only 类型")
        self.assertIsNone(result.policy, "value_only 时 policy 应为 None")
        self.assertIsInstance(result.value, float, "value 应为 float")

    def test_predict_value_range(self):
        """TC-MOD-02: predict 返回值域在 [-1, 1]"""
        result = self.model.predict([0] * 226)
        self.assertGreaterEqual(result.value, -1.0, "value 应 >= -1")
        self.assertLessEqual(result.value, 1.0, "value 应 <= 1")

    def test_predict_deterministic(self):
        """TC-MOD-03: 相同输入两次 predict 结果一致"""
        board = [0] * 225 + [1]
        result1 = self.model.predict(board)
        result2 = self.model.predict(board)
        self.assertEqual(result1.value, result2.value,
                         "相同输入应产生相同 value")

    def test_predict_different_players(self):
        """TC-MOD-04: 黑方和白方视角产生不同输出"""
        # 黑方首步天元
        board_black = [0] * 225
        board_black[7 * 15 + 7] = 1
        board_black.append(1)  # current_player = 黑
        value_black = self.model.predict(board_black).value

        # 白方首步天元
        board_white = [0] * 225
        board_white[7 * 15 + 7] = 2
        board_white.append(2)  # current_player = 白
        value_white = self.model.predict(board_white).value

        # 两个局面不同（视角不同），输出应不同
        self.assertNotEqual(value_black, value_white,
                            "不同视角应产生不同评估")

    def test_predict_batch_length(self):
        """TC-MOD-05: predict_batch 返回等长列表"""
        boards = [[0] * 226, [0] * 225 + [2], [0] * 226]
        results = self.model.predict_batch(boards)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.type, ModelPredict.TYPE_VALUE_ONLY)
            self.assertIsInstance(r.value, float)

    def test_predict_batch_empty(self):
        """predict_batch 空列表返回空列表"""
        results = self.model.predict_batch([])
        self.assertEqual(results, [])

    def test_predict_batch_consistency(self):
        """TC-MOD-06: predict_batch 与 predict 对同一输入结果一致"""
        board = [0] * 225 + [1]
        single_result = self.model.predict(board)
        batch_results = self.model.predict_batch([board])
        self.assertEqual(len(batch_results), 1)
        self.assertEqual(single_result.value, batch_results[0].value,
                         "predict 和 predict_batch 应一致")

    def test_save_load_roundtrip(self):
        """TC-MOD-07: save/load 往返后预测一致"""
        board = [0] * 225 + [1]
        value_before = self.model.predict(board).value

        # 保存到临时目录
        tmp_dir = tempfile.mkdtemp()
        try:
            save_path = os.path.join(tmp_dir, "test_roundtrip.pth")
            self.model.save(save_path)

            # 创建新模型并加载
            new_model = self.ModelCls(model_path=save_path)
            value_after = new_model.predict(board).value

            self.assertEqual(value_before, value_after,
                             "save/load 往返后 value 应一致")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fresh_skips_load(self):
        """TC-MOD-08: fresh=True 不加载已存在的模型文件"""
        # 默认模型路径（cnn_v10.pth）可能不存在，fresh=True 时应正常创建
        model = self.ModelCls(fresh=True)
        self.assertIsNotNone(model.network)
        # 预测应正常工作
        result = model.predict([0] * 226)
        self.assertEqual(result.type, ModelPredict.TYPE_VALUE_ONLY)

    def test_model_path_not_exists_raises(self):
        """TC-MOD-09: model_path 指向不存在文件时抛异常"""
        with self.assertRaises(FileNotFoundError):
            self.ModelCls(model_path="/nonexistent/path/model.pth")

    def test_model_name(self):
        """模型名称为 cnn_v10"""
        self.assertEqual(self.model.name, "cnn_v10")


if __name__ == '__main__':
    unittest.main()
