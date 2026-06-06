"""BaseTrainer 构造器 lr 参数单元测试

测试 BaseTrainer.__init__ 新增 lr 参数的各种场景，包括默认值、
显式传入、非法值校验、与 weight_decay 共存、子类兼容性。
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ai.base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# 测试辅助：最小化具体训练器子类，仅用于测试构造参数传递
# ---------------------------------------------------------------------------

class _MockModel:
    """Mock 模型：提供一个简单的 nn.Module 作为 network"""
    def __init__(self):
        self.network = nn.Linear(10, 225)
        self.full_name = "mock_model"

    def save(self):
        return "/tmp/mock.pth"


class _ConcreteTrainer(BaseTrainer):
    """可实例化的具体训练器子类，用于测试 BaseTrainer 构造参数"""

    @property
    def model_name(self) -> str:
        return "mock_trainer"

    def _create_model(self, fresh: bool):
        return _MockModel()


# ---------------------------------------------------------------------------
# UT-1: BaseTrainer 构造器 — lr 参数
# ---------------------------------------------------------------------------

class TestBaseTrainerLrParam(unittest.TestCase):
    """UT-1.1 ~ UT-1.7：BaseTrainer 构造器 lr 参数行为测试"""

    def test_ut_1_1_default_lr_when_not_provided(self):
        """UT-1.1：不传 lr 时使用默认值 DEFAULT_LR = 0.001"""
        trainer = _ConcreteTrainer(fresh=True)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.001)

    def test_ut_1_2_explicit_lr_0005(self):
        """UT-1.2：显式传入 lr=0.005"""
        trainer = _ConcreteTrainer(fresh=True, lr=0.005)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.005)

    def test_ut_1_3_scientific_notation_lr(self):
        """UT-1.3：传入科学计数法 lr=1e-4，应得到 0.0001"""
        trainer = _ConcreteTrainer(fresh=True, lr=1e-4)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.0001)

    def test_ut_1_4_lr_zero_raises_valueerror(self):
        """UT-1.4：传入 lr=0 应抛 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            _ConcreteTrainer(fresh=True, lr=0)
        self.assertIn("lr", str(ctx.exception).lower())

    def test_ut_1_5_lr_negative_raises_valueerror(self):
        """UT-1.5：传入 lr=-0.01 应抛 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            _ConcreteTrainer(fresh=True, lr=-0.01)
        self.assertIn("lr", str(ctx.exception).lower())

    def test_ut_1_6_lr_none_equals_default(self):
        """UT-1.6：lr=None 语义等于不传，使用默认值 0.001"""
        trainer = _ConcreteTrainer(fresh=True, lr=None)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.001)

    def test_ut_1_7_lr_with_weight_decay_no_conflict(self):
        """UT-1.7：lr 与 weight_decay 同时传入不冲突"""
        trainer = _ConcreteTrainer(fresh=True, lr=0.005, weight_decay=0.02)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        actual_wd = trainer.optimizer.param_groups[0]['weight_decay']
        self.assertEqual(actual_lr, 0.005)
        self.assertEqual(actual_wd, 0.02)


# ---------------------------------------------------------------------------
# UT-2: BaseTrainer 子类兼容性（通过 get_trainer）
# ---------------------------------------------------------------------------

class TestSubclassCompatibility(unittest.TestCase):
    """UT-2.1 ~ UT-2.3：子类训练器透传 lr 参数"""

    def test_ut_2_1_cnn_v3_trainer_passes_lr(self):
        """UT-2.1：get_trainer("cnn_v3.2", lr=0.005) 应正确透传 lr

        注：此测试需要 cnn_v3 gen2 权重存在。若不存在则跳过。
        """
        from ai.registry import get_trainer
        import os as _os

        gen2_path = _os.path.join(
            _os.path.dirname(__file__), '..', '..', '..',
            'ai', 'models', 'cnn_v3', 'gen2', 'cnn_v3.pth'
        )
        if not _os.path.exists(gen2_path):
            self.skipTest("cnn_v3 gen2 权重不存在，跳过集成子类测试")

        trainer = get_trainer("cnn_v3.2", lr=0.005)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.005)

    def test_ut_2_2_cnn_v3_trainer_default_lr(self):
        """UT-2.2：get_trainer("cnn_v3.2") 不传 lr 使用默认值 0.001"""
        from ai.registry import get_trainer
        import os as _os

        gen2_path = _os.path.join(
            _os.path.dirname(__file__), '..', '..', '..',
            'ai', 'models', 'cnn_v3', 'gen2', 'cnn_v3.pth'
        )
        if not _os.path.exists(gen2_path):
            self.skipTest("cnn_v3 gen2 权重不存在，跳过集成子类测试")

        trainer = get_trainer("cnn_v3.2")
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.001)

    def test_ut_2_3_fc_v1_trainer_passes_lr(self):
        """UT-2.3：get_trainer("fc_v1", lr=0.01) 应正确透传 lr"""
        from ai.registry import get_trainer

        trainer = get_trainer("fc_v1", lr=0.01)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(actual_lr, 0.01)


if __name__ == '__main__':
    unittest.main()
