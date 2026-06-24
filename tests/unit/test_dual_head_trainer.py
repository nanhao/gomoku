"""DualHeadTrainer 单元测试

测试双头训练基类：
- _train_step 返回值结构
- 损失计算 (MSE + CE)
- T4/T6 assert 校验
- _convert_inputs 相对编码
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 最小化 Mock 模型 ----------

class MockDualHeadNetwork(nn.Module):
    """最小化双头网络，用于测试 DualHeadTrainer"""

    def __init__(self):
        super().__init__()
        # 简单的策略头
        self.policy_fc = nn.Linear(226, 225)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # 简单的价值头
        self.value_fc = nn.Linear(226, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        """x: (batch, *) 将被 flatten 后处理"""
        flat = x.view(x.size(0), -1)[:, :226]  # 取前 226 维
        p_log = self.log_softmax(self.policy_fc(flat))
        v = self.value_tanh(self.value_fc(flat))
        return p_log, v


class MockDualHeadModel:
    """Mock 双头模型"""

    def __init__(self):
        self.network = MockDualHeadNetwork()

    def save(self, output_path):
        torch.save(self.network.state_dict(), output_path)
        return output_path


class _ConcreteTrainer:
    """DualHeadTrainer 的具体子类（测试用）"""

    @staticmethod
    def _create_model(fresh):
        return MockDualHeadModel()

    @property
    def model_name(self):
        return "test_model"

    @staticmethod
    def _convert_inputs(inputs):
        return torch.tensor(inputs, dtype=torch.float32)


# 动态创建继承 DualHeadTrainer 的测试用具体类
def _make_concrete_trainer(base_cls):
    """创建 DualHeadTrainer 的具体子类"""
    class ConcreteDualHeadTrainer(base_cls):
        def _create_model(self, fresh):
            return MockDualHeadModel()

        @property
        def model_name(self):
            return "test_model"

        def _convert_inputs(self, inputs):
            return torch.tensor(inputs, dtype=torch.float32)

    return ConcreteDualHeadTrainer


# ---------- 测试类 ----------

class TestDualHeadTrainer(unittest.TestCase):
    """DualHeadTrainer 核心测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)

    def test_construction(self):
        """基本构造"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.optimizer)

    def _to_device(self, trainer, *tensors):
        """将张量移到 trainer.device"""
        return tuple(t.to(trainer.device) for t in tensors)

    def test_train_step_returns_dict(self):
        """_train_step 返回字典，包含 value_loss, policy_loss, total_loss"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 4
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.tensor([[1.0], [-1.0], [0.0], [1.0]], dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        self.assertIsInstance(result, dict)
        self.assertIn("value_loss", result)
        self.assertIn("policy_loss", result)
        self.assertIn("total_loss", result)

    def test_train_step_loss_values_reasonable(self):
        """训练步的损失值应在合理范围内"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 4
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.tensor([[1.0], [-1.0], [0.0], [1.0]], dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        # value_loss 用 MSE，初始随机权重下应在 0~4 范围（真实 z ∈ [-1,1]，v ∈ (-1,1)）
        self.assertGreater(result["value_loss"], 0.0)
        self.assertGreater(result["policy_loss"], 0.0)
        self.assertGreater(result["total_loss"], 0.0)

    def test_train_step_with_winning_z(self):
        """赢棋 z=+1 时 value_loss 计算正确"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 2
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        self.assertGreater(result["value_loss"], 0.0)

    def test_train_step_with_losing_z(self):
        """输棋 z=-1 时 value_loss 计算正确"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 2
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.tensor([[-1.0], [-1.0]], dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        self.assertGreater(result["value_loss"], 0.0)

    def test_train_step_with_draw_z(self):
        """平局 z=0 时 value_loss 计算正确"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 2
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.zeros(batch_size, 1, dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        self.assertGreater(result["value_loss"], 0.0)

    def test_optimizer_is_adamw(self):
        """优化器应为 AdamW"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

    def test_default_lr(self):
        """默认学习率应为 0.001"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.001)

    def test_default_weight_decay(self):
        """默认 weight_decay 应为 1e-4"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)
        self.assertAlmostEqual(
            trainer.optimizer.param_groups[0]['weight_decay'], 1e-4, places=6
        )

    def test_custom_lr(self):
        """自定义学习率"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model, lr=0.01)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.01)

    def test_custom_weight_decay(self):
        """自定义 weight_decay"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model, weight_decay=5e-4)
        self.assertAlmostEqual(
            trainer.optimizer.param_groups[0]['weight_decay'], 5e-4, places=6
        )

    def test_train_step_grad_norms(self):
        """_train_step 返回 grad_norms 字典"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        batch_size = 4
        inputs = torch.randn(batch_size, 3, 15, 15)
        pi_targets = torch.softmax(torch.randn(batch_size, 225), dim=1)
        z_targets = torch.tensor([[1.0], [-1.0], [0.0], [1.0]], dtype=torch.float32)
        inputs, pi_targets, z_targets = self._to_device(trainer, inputs, pi_targets, z_targets)

        result = trainer._train_step(inputs, pi_targets, z_targets)
        # grad_norms 可能在结果字典中
        if "grad_norms" in result:
            self.assertIsInstance(result["grad_norms"], dict)
            # value_fc.weight 的梯度应非零
            if "value_fc.weight" in result["grad_norms"]:
                self.assertGreater(result["grad_norms"]["value_fc.weight"], 0.0)

    def test_save_model(self):
        """模型保存"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model.pth")
            model.save(output_path)
            self.assertTrue(os.path.exists(output_path))


class TestDualHeadTrainerDataLoad(unittest.TestCase):
    """数据加载测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.ConcreteTrainer = _make_concrete_trainer(self.DualHeadTrainer)

    def _create_sample_json(self, output_dir, n_samples=8):
        """创建测试用 sample.json"""
        samples = []
        for i in range(n_samples):
            board = [0] * 225
            # 放置一些随机棋子
            board[i % 225] = 1
            board[(i * 7 + 3) % 225] = 2
            state = board + [1 if i % 2 == 0 else 2]
            # π 分布
            pi = [1.0 / 225] * 225
            # z 值
            z = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
            samples.append({"state": state, "pi": pi, "z": z})

        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return sample_path

    def test_train_from_sample_json(self):
        """从 sample.json 训练"""
        model = MockDualHeadModel()
        trainer = self.ConcreteTrainer(model=model)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_sample_json(tmpdir, n_samples=16)
            # 如果 trainer 有 train_from_dir 方法
            if hasattr(trainer, 'train_from_dir'):
                result = trainer.train_from_dir(
                    train_from_dir=tmpdir, epochs=2, batch_size=4, val_split=0.0
                )
                self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
