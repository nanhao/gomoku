"""BaseTrainer val_ratio 相关功能测试

测试 _evaluate_val_loss()、train_batch/mini_batch/auto 的 val_ratio 参数。
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.base_trainer import BaseTrainer


def _create_sample_dir(tmp_dir, name="sample_0", n_items=1):
    """辅助函数：在临时目录下创建包含 sample.json 的子目录"""
    sample_dir = os.path.join(tmp_dir, name)
    os.makedirs(sample_dir, exist_ok=True)
    sample_data = []
    for i in range(n_items):
        sample_data.append({
            "input": [0.0] * 226,
            "output": [1.0 / 225.0] * 225,
            "mask": [1] * 225
        })
    with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)
    return sample_dir


class _ConcreteTrainer(BaseTrainer):
    """用于测试的具体训练器子类"""

    def _create_model(self, fresh):
        from ai.models.fc_v1.model import FCv1Model
        return FCv1Model(fresh=fresh)

    @property
    def model_name(self):
        return "test_val"


class TestEvaluateValLoss(unittest.TestCase):
    """测试 _evaluate_val_loss() 方法"""

    def setUp(self):
        from ai.models.fc_v1.model import FCv1Model
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
        self.trainer = _ConcreteTrainer(fresh=True)

    def tearDown(self):
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_val_loss_returns_float(self):
        """返回值为 float"""
        val_inputs = torch.zeros(2, 226, device=self.trainer.device)
        val_targets = torch.ones(2, 225, device=self.trainer.device) / 225.0
        val_masks = torch.ones(2, 225, device=self.trainer.device)

        val_loss = self.trainer._evaluate_val_loss(val_inputs, val_targets, val_masks)
        self.assertIsInstance(val_loss, float)

    def test_evaluate_val_loss_no_grad(self):
        """评估时不计算梯度（参数不变）"""
        val_inputs = torch.zeros(2, 226, device=self.trainer.device)
        val_targets = torch.ones(2, 225, device=self.trainer.device) / 225.0
        val_masks = torch.ones(2, 225, device=self.trainer.device)

        # 记录评估前的参数
        params_before = [p.clone() for p in self.trainer.model.network.parameters()]

        self.trainer._evaluate_val_loss(val_inputs, val_targets, val_masks)

        # 验证参数未改变
        for p_before, p_after in zip(params_before, self.trainer.model.network.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))

    def test_evaluate_val_loss_model_state(self):
        """评估后模型恢复 train 模式"""
        val_inputs = torch.zeros(2, 226, device=self.trainer.device)
        val_targets = torch.ones(2, 225, device=self.trainer.device) / 225.0
        val_masks = torch.ones(2, 225, device=self.trainer.device)

        # 确保初始为 train 模式
        self.trainer.model.network.train()
        self.assertTrue(self.trainer.model.network.training)

        self.trainer._evaluate_val_loss(val_inputs, val_targets, val_masks)

        # 评估后应恢复 train 模式
        self.assertTrue(self.trainer.model.network.training)


class TestTrainBatchWithValRatio(unittest.TestCase):
    """测试 train_batch 的 val_ratio 参数"""

    def setUp(self):
        from ai.models.fc_v1.model import FCv1Model
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")

    def tearDown(self):
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_batch_with_val_ratio(self):
        """val_ratio > 0 时 val_loss 被记录

        关注点：log_val 调用契约。真实 forward/backward 与 val 前向的覆盖由
        TestEvaluateValLoss 等其他测试保障，此处 mock 掉避免过度测试。
        """
        # 创建足够多样本的数据（>10 个，确保有验证集）
        sample_dir = _create_sample_dir(self.temp_dir, "data", n_items=20)
        trainer = _ConcreteTrainer(fresh=True)
        mock_logger = MagicMock()
        mock_logger.dense_log = False
        trainer.loss_logger = mock_logger

        fake_step = (torch.tensor(0.5), 0.1)
        with patch.object(trainer, '_forward_backward_step', return_value=fake_step), \
             patch.object(trainer, '_evaluate_val_loss', return_value=0.3):
            trainer.train_batch([sample_dir], epochs=200, verbose=False, val_ratio=0.2)

        # val_loss 应在 LOSS_LOG_INTERVAL (200) 的 epoch 被记录
        self.assertTrue(mock_logger.log_val.called)

    def test_train_batch_val_ratio_zero(self):
        """val_ratio=0 时行为与原逻辑一致（不记录 val_loss）

        关注点：log_val 不被调用 + log 被调用。真实 forward/backward 覆盖由其他测试保障。
        """
        sample_dir = _create_sample_dir(self.temp_dir, "data", n_items=5)
        trainer = _ConcreteTrainer(fresh=True)
        mock_logger = MagicMock()
        mock_logger.dense_log = False
        trainer.loss_logger = mock_logger

        fake_step = (torch.tensor(0.5), 0.1)
        with patch.object(trainer, '_forward_backward_step', return_value=fake_step):
            trainer.train_batch([sample_dir], epochs=200, verbose=False, val_ratio=0.0)

        # log_val 不应被调用
        self.assertFalse(mock_logger.log_val.called)
        # 但 log 仍应被正常调用
        self.assertTrue(mock_logger.log.called)


class TestTrainMiniBatchWithValRatio(unittest.TestCase):
    """测试 train_mini_batch 的 val_ratio 参数"""

    def setUp(self):
        from ai.models.fc_v1.model import FCv1Model
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")

    def tearDown(self):
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_mini_batch_with_val_ratio(self):
        """mini-batch 模式下 val_loss 每 epoch 记录"""
        sample_dir = _create_sample_dir(self.temp_dir, "data", n_items=20)
        trainer = _ConcreteTrainer(fresh=True)
        mock_logger = MagicMock()
        mock_logger.dense_log = True
        trainer.loss_logger = mock_logger

        trainer.train_mini_batch([sample_dir], epochs=2, batch_size=4, verbose=False, val_ratio=0.2)

        # mini-batch 模式下每个 epoch 结束后都记录 val_loss
        self.assertTrue(mock_logger.log_val.called)
        self.assertEqual(mock_logger.log_val.call_count, 2)

    def test_train_mini_batch_val_ratio_zero(self):
        """val_ratio=0 时 mini-batch 不记录 val_loss"""
        sample_dir = _create_sample_dir(self.temp_dir, "data", n_items=10)
        trainer = _ConcreteTrainer(fresh=True)
        mock_logger = MagicMock()
        mock_logger.dense_log = True
        trainer.loss_logger = mock_logger

        trainer.train_mini_batch([sample_dir], epochs=2, batch_size=4, verbose=False, val_ratio=0.0)

        self.assertFalse(mock_logger.log_val.called)


class TestTrainAutoPassesValRatio(unittest.TestCase):
    """测试 train_auto 透传 val_ratio"""

    def setUp(self):
        from ai.models.fc_v1.model import FCv1Model
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
        self.trainer = _ConcreteTrainer(fresh=True)
        # mock 底层方法避免真实训练
        self.trainer.train_batch = MagicMock(return_value=0.123)
        self.trainer.train_mini_batch = MagicMock(return_value=0.456)

    def tearDown(self):
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_auto_passes_val_ratio_to_train_batch(self):
        """train_auto 透传 val_ratio 给 train_batch"""
        self.trainer.train_auto(["dir1"], epochs=10, batch_size=None, val_ratio=0.2)
        self.trainer.train_batch.assert_called_once()
        call_kwargs = self.trainer.train_batch.call_args[1]
        self.assertEqual(call_kwargs['val_ratio'], 0.2)

    def test_train_auto_passes_val_ratio_to_train_mini_batch(self):
        """train_auto 透传 val_ratio 给 train_mini_batch"""
        self.trainer.train_auto(["dir1"], epochs=10, batch_size=32, val_ratio=0.2)
        self.trainer.train_mini_batch.assert_called_once()
        call_kwargs = self.trainer.train_mini_batch.call_args[1]
        self.assertEqual(call_kwargs['val_ratio'], 0.2)

    def test_train_auto_default_val_ratio_zero(self):
        """未传 val_ratio 时默认值为 0.0"""
        self.trainer.train_auto(["dir1"], epochs=10)
        self.trainer.train_batch.assert_called_once()
        call_kwargs = self.trainer.train_batch.call_args[1]
        self.assertEqual(call_kwargs.get('val_ratio', 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()
