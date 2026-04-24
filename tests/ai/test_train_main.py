"""train_main.py 测试

测试 get_trainer 工厂函数的模型名称映射和参数解析。
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.train_main import get_trainer, main


class TestGetTrainer(unittest.TestCase):
    """测试 get_trainer 函数"""

    def test_get_trainer_fc_v1(self):
        """fc_v1 返回 FCv1Trainer"""
        from ai.models.fc_v1.trainer import FCv1Trainer
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsInstance(trainer, FCv1Trainer)

    def test_get_trainer_cnn_v1(self):
        """cnn_v1 返回 CNNV1Trainer"""
        from ai.models.cnn_v1.trainer import CNNV1Trainer
        trainer = get_trainer("cnn_v1", fresh=True)
        self.assertIsInstance(trainer, CNNV1Trainer)

    def test_get_trainer_cnn_v2(self):
        """cnn_v2 返回 CNNV2Trainer"""
        from ai.models.cnn_v2.trainer import CNNV2Trainer
        trainer = get_trainer("cnn_v2", fresh=True)
        self.assertIsInstance(trainer, CNNV2Trainer)

    def test_get_trainer_random_raises(self):
        """random 不是有效的 trainer 模型名，应抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("random")
        self.assertIn("不支持的模型", str(ctx.exception))
        self.assertIn("random", str(ctx.exception))

    def test_get_trainer_invalid_raises(self):
        """无效模型名抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("invalid_model")
        self.assertIn("不支持的模型", str(ctx.exception))
        self.assertIn("invalid_model", str(ctx.exception))

    def test_get_trainer_fc_v1_with_fresh(self):
        """fc_v1 支持 fresh 参数"""
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)

    def test_get_trainer_cnn_v1_with_fresh(self):
        """cnn_v1 支持 fresh 参数"""
        trainer = get_trainer("cnn_v1", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)

    def test_get_trainer_cnn_v2_with_fresh(self):
        """cnn_v2 支持 fresh 参数"""
        trainer = get_trainer("cnn_v2", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)

    def test_get_trainer_passes_kwargs(self):
        """额外 kwargs 应传递给 Trainer 构造函数"""
        trainer = get_trainer("fc_v1", fresh=True, lr_scheduler_type="step")
        # 不崩溃即通过
        self.assertIsNotNone(trainer)


class TestTrainMainArgParsing(unittest.TestCase):
    """测试 train_main 的参数解析"""

    def test_cycle_mode_parses_rounds_and_games(self):
        """--cycle 模式应解析 --rounds 和 --games"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--cycle', '--rounds', '5', '--games', '10']):
            # 验证不崩溃（使用 mock 替代实际训练）
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_trainer.train_cycle = MagicMock()
                mock_get.return_value = mock_trainer
                main()

                mock_trainer.train_cycle.assert_called_once()
                call_kwargs = mock_trainer.train_cycle.call_args[1]
                self.assertEqual(call_kwargs['rounds'], 5)
                self.assertEqual(call_kwargs['games'], 10)

    def test_cycle_mode_with_batch_size(self):
        """--cycle 模式应传递 --batch-size 和 --shuffle"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--cycle', '--batch-size', '32', '--shuffle']):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_trainer.train_cycle = MagicMock()
                mock_get.return_value = mock_trainer
                main()

                call_kwargs = mock_trainer.train_cycle.call_args[1]
                self.assertEqual(call_kwargs['batch_size'], 32)
                self.assertTrue(call_kwargs['shuffle'])

    def test_cycle_and_sample_dir_mutually_exclusive(self):
        """--cycle 和 --sample-dir 不能同时使用"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--cycle', '--sample-dir', '/tmp/data']):
            with self.assertRaises(SystemExit):
                main()

    def test_neither_cycle_nor_sample_dir_fails(self):
        """不指定 --cycle 和 --sample-dir 应报错"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1']):
            with self.assertRaises(SystemExit):
                main()

    def test_fresh_flag(self):
        """--fresh 标志应传递给 get_trainer"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1', '--cycle', '--fresh', '--rounds', '1', '--games', '1']):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_trainer.train_cycle = MagicMock()
                mock_get.return_value = mock_trainer
                main()

                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args[1]
                self.assertTrue(call_kwargs['fresh'])

    def test_lr_scheduler_args(self):
        """学习率调度参数应传递给 get_trainer"""
        with patch('sys.argv', [
            'train_main.py', '--model', 'fc_v1', '--cycle',
            '--lr-scheduler', 'step', '--lr-step-size', '300', '--lr-gamma', '0.8',
            '--rounds', '1', '--games', '1'
        ]):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_trainer.train_cycle = MagicMock()
                mock_get.return_value = mock_trainer
                main()

                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs['lr_scheduler_type'], 'step')
                self.assertEqual(call_kwargs['lr_step_size'], 300)
                self.assertEqual(call_kwargs['lr_gamma'], 0.8)

    def test_default_model_is_fc_v1(self):
        """默认模型应为 fc_v1"""
        with patch('sys.argv', ['train_main.py', '--cycle', '--rounds', '1', '--games', '1']):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_trainer.train_cycle = MagicMock()
                mock_get.return_value = mock_trainer
                main()

                mock_get.assert_called_once()
                self.assertEqual(mock_get.call_args[0][0], 'fc_v1')


if __name__ == '__main__':
    unittest.main()
