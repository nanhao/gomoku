"""CNN v12 Trainer 单元测试

验证 CNNV12Trainer 的模型名、模型创建与代数透传、输入转换、
单步训练、优化器超参、损失曲线图命名。
测试覆盖 TC3.1 ~ TC3.6（设计 61 test_cases.md 三）。

约束：全部用例纯内存执行，不写任何文件；gen 透传使用 fresh=True，
不触发权重加载，因此不触碰真实模型目录。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch


class TestCNNV12Trainer(unittest.TestCase):
    """CNNV12Trainer 单元测试"""

    @classmethod
    def setUpClass(cls):
        """导入被测模块"""
        from ai.models.cnn_v12.model import CNNV12Model
        from ai.models.cnn_v12.trainer import CNNV12Trainer
        cls.TrainerCls = CNNV12Trainer
        cls.ModelCls = CNNV12Model

    # ---------- TC3.1 训练器模型名 ----------

    def test_model_name(self):
        """model_name 恒为 cnn_v12"""
        trainer = self.TrainerCls(fresh=True, device=torch.device("cpu"))
        self.assertEqual(trainer.model_name, "cnn_v12")

    # ---------- TC3.2 模型创建与代数透传 ----------

    def test_create_model_with_gen(self):
        """gen 透传给 CNNV12Model（fresh=True 不加载权重）"""
        trainer = self.TrainerCls(
            fresh=True, gen=2, device=torch.device("cpu"))
        self.assertIsInstance(trainer.model, self.ModelCls)
        self.assertEqual(trainer.model.gen, 2)
        self.assertIsInstance(trainer.model.network,
                              type(trainer.model._create_network()))

    # ---------- TC3.3 单步训练可执行 ----------

    def test_train_step_loss_finite(self):
        """_train_step 返回有限的 total_loss（非 NaN / Inf）"""
        trainer = self.TrainerCls(fresh=True, device=torch.device("cpu"))
        trainer.model.network.train()

        inputs = torch.randn(4, 3, 15, 15)
        pi_targets = torch.zeros(4, 225)
        for row in range(4):
            pi_targets[row, row * 10] = 1.0
        z_targets = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])

        result = trainer._train_step(inputs, pi_targets, z_targets)

        for key in ("value_loss", "policy_loss", "total_loss"):
            self.assertIn(key, result)
            self.assertTrue(math.isfinite(result[key]),
                            f"{key} 应为有限值，实际 {result[key]}")

    # ---------- TC3.4 输入张量形状 ----------

    def test_convert_inputs_shape(self):
        """一条 226 维 state 经输入转换得到 (1, 3, 15, 15)"""
        tensor = self.TrainerCls._convert_inputs([[0] * 226])
        self.assertEqual(tuple(tensor.shape), (1, 3, 15, 15))
        self.assertEqual(tensor.dtype, torch.float32)

    # ---------- TC3.5 优化器超参 ----------

    def test_optimizer_hyperparams(self):
        """优化器为 AdamW，weight_decay=1e-4，lr 为传入值"""
        trainer = self.TrainerCls(
            fresh=True, lr=0.002, device=torch.device("cpu"))
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

        param_group = trainer.optimizer.param_groups[0]
        self.assertAlmostEqual(param_group["weight_decay"], 1e-4, places=10)
        self.assertAlmostEqual(param_group["lr"], 0.002, places=10)

    # ---------- TC3.6 曲线图改名 ----------

    def test_loss_plot_name_prefix(self):
        """模型名用于损失曲线图文件名，得到 cnn_v12_* 而非 cnn_v11_*"""
        trainer = self.TrainerCls(fresh=True, device=torch.device("cpu"))

        # 复现基类 _plot_loss_curve 的命名规则：
        # f"{model_name}_{timestamp}.png"
        filename = f"{trainer.model_name}_20260101_000000.png"
        self.assertTrue(filename.startswith("cnn_v12_"),
                        f"图名应以 cnn_v12_ 开头，实际 {filename}")
        self.assertFalse(filename.startswith("cnn_v11_"),
                         f"图名不应为 cnn_v11_*，实际 {filename}")


if __name__ == '__main__':
    unittest.main()
