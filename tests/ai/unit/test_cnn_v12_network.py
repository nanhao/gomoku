"""CNN v12 Network 单元测试

验证 CNNV12Network 的结构、前向传播、参数量、TorchScript 可导出性。
测试覆盖 TC1.1 ~ TC1.11（设计 61 test_cases.md 一）。

约束：所有用例纯内存执行，不写任何文件。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV12Network(unittest.TestCase):
    """CNNV12Network 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入被测与对照模块"""
        from ai.models.cnn_v11.model import CNNV11Network
        from ai.models.cnn_v12.model import CNNV12Network
        cls.NetworkCls = CNNV12Network
        cls.V11NetworkCls = CNNV11Network

    def setUp(self):
        """每个用例创建独立网络（eval 模式，避免 BN 统计量干扰）"""
        self.network = self.NetworkCls()
        self.network.eval()

    # ---------- TC1.1 批量前向输出形状 ----------

    def test_forward_output_shapes(self):
        """forward 返回 (policy, value)，形状正确"""
        x = torch.randn(4, 3, 15, 15)
        policy, value = self.network(x)
        self.assertEqual(policy.shape, (4, 225),
                         f"policy shape 应为 (4, 225)，实际 {policy.shape}")
        self.assertEqual(value.shape, (4, 1),
                         f"value shape 应为 (4, 1)，实际 {value.shape}")

    # ---------- TC1.2 单样本前向 ----------

    def test_forward_single_sample(self):
        """单样本 (1, 3, 15, 15) 前向传播"""
        x = torch.randn(1, 3, 15, 15)
        policy, value = self.network(x)
        self.assertEqual(policy.shape, (1, 225))
        self.assertEqual(value.shape, (1, 1))

    # ---------- TC1.3 策略头值域（LogSoftmax） ----------

    def test_policy_value_range(self):
        """策略头输出 LogSoftmax → 所有值 ≤ 0"""
        policy, _ = self.network(torch.randn(4, 3, 15, 15))
        self.assertTrue((policy <= 0).all(),
                        "LogSoftmax 输出应全部 ≤ 0")

    # ---------- TC1.4 策略头归一化 ----------

    def test_policy_sum_equals_one(self):
        """exp(policy) 每行之和 ≈ 1.0"""
        policy, _ = self.network(torch.randn(4, 3, 15, 15))
        row_sums = policy.exp().sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(4), atol=1e-5),
                        f"exp(policy) 行和应 ≈ 1.0，实际 {row_sums}")

    # ---------- TC1.5 价值头值域（Tanh） ----------

    def test_value_range(self):
        """价值头输出 Tanh → 值域 [-1, 1]"""
        _, value = self.network(torch.randn(4, 3, 15, 15))
        self.assertTrue((value >= -1.0).all() and (value <= 1.0).all(),
                        f"value 值域应在 [-1, 1] 内，"
                        f"实际 [{value.min():.4f}, {value.max():.4f}]")

    # ---------- TC1.6 参数量精确值 ----------

    def test_parameter_count_exact(self):
        """参数量恰好 1,346,541（与设计 61 §3.5 明细一致）"""
        total_params = sum(p.numel() for p in self.network.parameters())
        self.assertEqual(total_params, 1_346_541,
                         f"参数量应为 1,346,541，实际 {total_params:,}")

    # ---------- TC1.7 架构常量 ----------

    def test_architecture_constants(self):
        """CHANNELS=128、NUM_RES_BLOCKS=4、res_blocks 长度 4、块内通道 128"""
        self.assertEqual(self.NetworkCls.CHANNELS, 128)
        self.assertEqual(self.NetworkCls.NUM_RES_BLOCKS, 4)

        self.assertIsInstance(self.network.res_blocks, torch.nn.ModuleList)
        self.assertEqual(len(self.network.res_blocks), 4,
                         f"res_blocks 长度应为 4，"
                         f"实际 {len(self.network.res_blocks)}")

        for idx, block in enumerate(self.network.res_blocks):
            self.assertEqual(block.conv1.in_channels, 128,
                             f"第 {idx} 块 conv1.in_channels 应为 128")
            self.assertEqual(block.conv1.out_channels, 128,
                             f"第 {idx} 块 conv1.out_channels 应为 128")
            self.assertEqual(block.conv2.in_channels, 128,
                             f"第 {idx} 块 conv2.in_channels 应为 128")
            self.assertEqual(block.conv2.out_channels, 128,
                             f"第 {idx} 块 conv2.out_channels 应为 128")

        # 3×3 卷积共 1（初始 Conv）+ 4×2（残差块）= 9 层
        conv_3x3 = sum(1 for m in self.network.modules()
                       if isinstance(m, torch.nn.Conv2d)
                       and m.kernel_size == (3, 3))
        self.assertEqual(conv_3x3, 9,
                         f"3×3 Conv2d 应为 9 层，实际 {conv_3x3}")

    # ---------- TC1.8 相对 cnn_v11 的瘦身比例 ----------

    def test_slimmer_than_v11(self):
        """v12 参数量不足 cnn_v11 的 1/6"""
        v11_params = sum(
            p.numel() for p in self.V11NetworkCls().parameters())
        v12_params = sum(p.numel() for p in self.network.parameters())
        self.assertLess(v12_params, v11_params / 6,
                        f"v12 参数 {v12_params:,} 应 < v11 参数 "
                        f"{v11_params:,} 的 1/6")

    # ---------- TC1.9 与 cnn_v11 权重不兼容 ----------

    def test_state_dict_incompatible_with_v11(self):
        """载入 cnn_v11 的 state_dict 抛 RuntimeError（形状不匹配）"""
        v11_state = self.V11NetworkCls().state_dict()
        with self.assertRaises(RuntimeError):
            self.network.load_state_dict(v11_state)

    # ---------- TC1.10 TorchScript 可导出 ----------

    def test_torchscript_export(self):
        """torch.jit.script 后前向输出形状与数值均与 eager 一致"""
        # 构造权重完全相同的两个实例，避免随机初始化差异干扰比对
        eager_network = self.NetworkCls()
        eager_network.eval()

        script_target = self.NetworkCls()
        script_target.load_state_dict(eager_network.state_dict())
        scripted = torch.jit.script(script_target)
        scripted.eval()

        x = torch.randn(2, 3, 15, 15)
        with torch.no_grad():
            policy_eager, value_eager = eager_network(x)
            policy_scripted, value_scripted = scripted(x)

        self.assertEqual(tuple(policy_scripted.shape),
                         tuple(policy_eager.shape))
        self.assertEqual(tuple(value_scripted.shape),
                         tuple(value_eager.shape))
        self.assertTrue(
            torch.allclose(policy_scripted, policy_eager, atol=1e-5),
            "TorchScript 与 eager 的 policy 输出应一致")
        self.assertTrue(
            torch.allclose(value_scripted, value_eager, atol=1e-5),
            "TorchScript 与 eager 的 value 输出应一致")

    # ---------- TC1.11 梯度可回传 ----------

    def test_gradient_flow(self):
        """双头梯度均可达全部参数（无断链层）"""
        network = self.NetworkCls()
        network.train()

        policy, value = network(torch.randn(2, 3, 15, 15))
        (policy.sum() + value.sum()).backward()

        for name, param in network.named_parameters():
            self.assertIsNotNone(
                param.grad, f"参数 {name} 的梯度为 None（存在断链层）")


if __name__ == '__main__':
    unittest.main()
