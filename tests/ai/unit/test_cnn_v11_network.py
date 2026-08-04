"""CNN v11 Network 单元测试

验证 CNNV11Network 的结构、前向传播、参数量。
遵循 TDD 流程，先写测试再实现（当前为 RED 阶段）。

测试覆盖 TC1.1 ~ TC1.8（设计文档 test_cases.md 一）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV11Network(unittest.TestCase):
    """CNNV11Network 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入模块（RED 阶段会失败——cnn_v11 包尚不存在）"""
        from ai.models.cnn_v11.model import CNNV11Network
        cls.NetworkCls = CNNV11Network

    def setUp(self):
        self.network = self.NetworkCls()
        self.network.eval()

    # ---------- TC1.1 forward — 输出形状 ----------

    def test_forward_output_shapes(self):
        """forward 返回 (policy, value)，形状正确"""
        x = torch.randn(4, 3, 15, 15)
        policy, value = self.network(x)
        self.assertEqual(policy.shape, (4, 225),
                         f"policy shape 应为 (4, 225)，实际 {policy.shape}")
        self.assertEqual(value.shape, (4, 1),
                         f"value shape 应为 (4, 1)，实际 {value.shape}")

    def test_forward_single_sample(self):
        """单样本 (1, 3, 15, 15) 前向传播"""
        x = torch.randn(1, 3, 15, 15)
        policy, value = self.network(x)
        self.assertEqual(policy.shape, (1, 225))
        self.assertEqual(value.shape, (1, 1))

    # ---------- TC1.2 forward — 策略头值域 ----------

    def test_policy_value_range(self):
        """策略头输出 LogSoftmax → 所有值 ≤ 0"""
        x = torch.randn(4, 3, 15, 15)
        policy, _ = self.network(x)
        self.assertTrue((policy <= 0).all(),
                        "LogSoftmax 输出应全部 ≤ 0")

    def test_policy_sum_equals_one(self):
        """exp(policy) 每行之和 ≈ 1.0"""
        x = torch.randn(4, 3, 15, 15)
        policy, _ = self.network(x)
        row_sums = policy.exp().sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(4), atol=1e-4),
                        f"exp(policy) 行和应 ≈ 1.0，实际 {row_sums}")

    # ---------- TC1.3 forward — 价值头值域 ----------

    def test_value_value_range(self):
        """价值头输出 Tanh → 值域 [-1, 1]"""
        x = torch.randn(4, 3, 15, 15)
        _, value = self.network(x)
        self.assertTrue((value >= -1.0).all() and (value <= 1.0).all(),
                        f"value 值域应在 [-1, 1] 内，"
                        f"实际 [{value.min():.4f}, {value.max():.4f}]")

    # ---------- TC1.4 forward — 确定性输出 ----------

    def test_forward_deterministic(self):
        """相同输入产生相同输出（eval 模式）"""
        x = torch.randn(1, 3, 15, 15)
        policy1, value1 = self.network(x)
        policy2, value2 = self.network(x)
        self.assertTrue(torch.allclose(policy1, policy2, atol=1e-6))
        self.assertTrue(torch.allclose(value1, value2, atol=1e-6))

    # ---------- TC1.5 forward — 梯度可传播 ----------

    def test_gradient_flow(self):
        """双头梯度均可达卷积体"""
        x = torch.randn(2, 3, 15, 15, requires_grad=False)
        # 需要 network 处于 train 模式且 requires_grad=True
        self.network.train()
        x_grad = torch.randn(2, 3, 15, 15, requires_grad=True)
        policy, value = self.network(x_grad)
        loss = policy.sum() + value.sum()
        loss.backward()

        # 验证梯度非 None
        self.assertIsNotNone(x_grad.grad, "输入梯度应为非 None")
        # 验证关键参数有梯度
        self.assertIsNotNone(
            self.network.conv1.weight.grad,
            "conv1.weight.grad 应为非 None")
        self.assertIsNotNone(
            self.network.policy_fc.weight.grad,
            "policy_fc.weight.grad 应为非 None")
        self.assertIsNotNone(
            self.network.value_fc2.weight.grad,
            "value_fc2.weight.grad 应为非 None")

    # ---------- TC1.6 forward — eval 模式确定性 ----------

    def test_train_vs_eval_consistency(self):
        """eval 模式多次前向输出一致（无 dropout，确定性）"""
        x = torch.randn(1, 3, 15, 15)
        self.network.eval()
        with torch.no_grad():
            policy_1, value_1 = self.network(x)
            policy_2, value_2 = self.network(x)
            policy_3, value_3 = self.network(x)
        # eval 模式下多次输出完全一致
        self.assertTrue(torch.allclose(policy_1, policy_2, atol=1e-6))
        self.assertTrue(torch.allclose(value_1, value_2, atol=1e-6))
        self.assertTrue(torch.allclose(policy_1, policy_3, atol=1e-6))
        self.assertTrue(torch.allclose(value_1, value_3, atol=1e-6))

    # ---------- TC1.7 参数量范围 ----------

    def test_parameter_count(self):
        """参数量在 830 万 ~ 860 万之间"""
        total_params = sum(p.numel() for p in self.network.parameters())
        self.assertGreater(total_params, 8_300_000,
                           f"参数 {total_params:,} < 8,300,000")
        self.assertLess(total_params, 8_600_000,
                        f"参数 {total_params:,} > 8,600,000")

    # ---------- TC1.8 残差块数 ----------

    def test_residual_block_count(self):
        """网络应有 7 个残差块（可通过 Conv2d 层数验证）"""
        # 统计非 1×1 的 Conv2d 层数
        conv2d_layers = [
            m for m in self.network.modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        # conv1(3→256) + 14 conv(256→256) + conv_out(256→2, 1x1) + value_conv_out(256→1, 1x1) = 17
        kernel_3x3_count = sum(
            1 for m in conv2d_layers
            if m.kernel_size == (3, 3)
        )
        self.assertEqual(kernel_3x3_count, 15,
                         f"3×3 Conv2d 应为 15 层，实际 {kernel_3x3_count}")

    def test_module_list_structure(self):
        """残差块通过 nn.ModuleList 组织"""
        has_module_list = hasattr(self.network, 'res_blocks') and isinstance(
            self.network.res_blocks, torch.nn.ModuleList)
        self.assertTrue(has_module_list,
                        "res_blocks 应为 nn.ModuleList")
        self.assertEqual(len(self.network.res_blocks), 7,
                         f"res_blocks 长度应为 7，实际 {len(self.network.res_blocks)}")


if __name__ == '__main__':
    unittest.main()
