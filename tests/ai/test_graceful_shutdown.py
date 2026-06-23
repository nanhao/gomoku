"""测试训练过程中 Ctrl+C 优雅退出

验证训练被 KeyboardInterrupt 中断时：
- 保存当前模型
- 生成 loss 曲线图
- 重新抛出 KeyboardInterrupt 以便上层调用者感知中断并终止流程
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock

import torch
from ai.base_trainer import BaseTrainer
from ai.models.fc_v1.model import FCv1Model


class MockTrainer(BaseTrainer):
    """用于测试的 mock 训练器，使用 FC v1 模型"""

    def _create_model(self, fresh):
        return FCv1Model(fresh=fresh)

    @property
    def model_name(self):
        return "mock"


class TestGracefulShutdown(unittest.TestCase):
    """测试优雅退出行为"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_path = os.path.join(self.test_dir, "sample.json")
        # 创建测试数据：5 条样本
        sample_data = [
            {
                "input": [0.0] * 226,
                "output": [1.0 / 225] * 225,
                "mask": [1] * 225
            }
            for _ in range(5)
        ]
        with open(self.sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_interrupting_forward(self, original_forward, trigger_after=2):
        """创建在第 N 次调用时抛出 KeyboardInterrupt 的 forward 函数"""
        call_count = [0]

        def forward_with_interrupt(x):
            call_count[0] += 1
            if call_count[0] >= trigger_after:
                raise KeyboardInterrupt()
            return original_forward(x)

        return forward_with_interrupt

    def _setup_mock_logger(self, trainer):
        """为训练器设置 mock loss_logger"""
        plot_called = []
        flush_called = []

        def mock_plot():
            plot_called.append(True)
            return "test.png"

        def mock_flush():
            flush_called.append(True)

        mock_logger = MagicMock()
        mock_logger.plot = mock_plot
        mock_logger.flush = mock_flush
        mock_logger.dense_log = False
        trainer.loss_logger = mock_logger
        return plot_called, flush_called

    def test_train_single_saves_on_interrupt(self):
        """train_single 被中断时应保存模型并生成图表"""
        trainer = MockTrainer(fresh=True, device=torch.device("cpu"))

        # Mock model.save
        save_called = []

        def mock_save():
            save_called.append(True)
            return "test.pth"

        trainer.model.save = mock_save

        # 第 2 次 forward 后中断
        original_forward = trainer.model.network.forward
        trainer.model.network.forward = self._create_interrupting_forward(
            original_forward, trigger_after=2
        )

        plot_called, flush_called = self._setup_mock_logger(trainer)

        # 执行训练，应抛出 KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            trainer.train_single(self.test_dir, epochs=2)

        self.assertTrue(save_called, "中断时应保存模型")
        self.assertTrue(plot_called, "中断时应生成图表")
        self.assertTrue(flush_called, "中断时应 flush 日志")

    def test_train_batch_saves_on_interrupt(self):
        """train_batch 被中断时应保存模型并生成图表"""
        trainer = MockTrainer(fresh=True, device=torch.device("cpu"))

        save_called = []

        def mock_save():
            save_called.append(True)
            return "test.pth"

        trainer.model.save = mock_save

        original_forward = trainer.model.network.forward
        trainer.model.network.forward = self._create_interrupting_forward(
            original_forward, trigger_after=2
        )

        plot_called, flush_called = self._setup_mock_logger(trainer)

        with self.assertRaises(KeyboardInterrupt):
            trainer.train_batch([self.test_dir], epochs=2)

        self.assertTrue(save_called, "中断时应保存模型")
        self.assertTrue(plot_called, "中断时应生成图表")
        self.assertTrue(flush_called, "中断时应 flush 日志")

    def test_train_mini_batch_saves_on_interrupt(self):
        """train_mini_batch 被中断时应保存模型并生成图表"""
        trainer = MockTrainer(fresh=True, device=torch.device("cpu"))

        save_called = []

        def mock_save():
            save_called.append(True)
            return "test.pth"

        trainer.model.save = mock_save

        original_forward = trainer.model.network.forward
        trainer.model.network.forward = self._create_interrupting_forward(
            original_forward, trigger_after=2
        )

        plot_called, flush_called = self._setup_mock_logger(trainer)

        with self.assertRaises(KeyboardInterrupt):
            trainer.train_mini_batch(
                [self.test_dir], epochs=2, batch_size=2
            )

        self.assertTrue(save_called, "中断时应保存模型")
        self.assertTrue(plot_called, "中断时应生成图表")
        self.assertTrue(flush_called, "中断时应 flush 日志")

    def test_train_single_no_save_on_normal_completion(self):
        """train_single 正常完成时不应由 _finalize_training 保存模型"""
        trainer = MockTrainer(fresh=True, device=torch.device("cpu"))

        save_called = []

        def mock_save():
            save_called.append(True)
            return "test.pth"

        trainer.model.save = mock_save

        plot_called, _ = self._setup_mock_logger(trainer)

        avg_loss = trainer.train_single(self.test_dir, epochs=2)

        self.assertFalse(save_called, "正常完成时不应由训练器保存模型")
        self.assertTrue(plot_called, "正常完成时应生成图表")
        self.assertIsInstance(avg_loss, float)

    def test_train_single_zero_epoch_interrupt(self):
        """第 0 个 epoch 就被中断时也应保存模型"""
        trainer = MockTrainer(fresh=True, device=torch.device("cpu"))

        save_called = []

        def mock_save():
            save_called.append(True)
            return "test.pth"

        trainer.model.save = mock_save

        # 第 1 次 forward 就中断
        original_forward = trainer.model.network.forward
        trainer.model.network.forward = self._create_interrupting_forward(
            original_forward, trigger_after=1
        )

        plot_called, _ = self._setup_mock_logger(trainer)

        with self.assertRaises(KeyboardInterrupt):
            trainer.train_single(self.test_dir, epochs=2)

        self.assertTrue(save_called, "第 0 个 epoch 中断时也应保存模型")


if __name__ == "__main__":
    unittest.main()
