"""train_main.py 测试

测试 get_trainer 工厂函数的模型名称映射和参数解析。
"""

import json
import os
import sys
import tempfile
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


class TestSelfPlayMergeOutput(unittest.TestCase):
    """测试 --self-play 模式的 --merge-output 功能"""

    def test_self_play_without_merge_output(self):
        """--self-play 不指定 --merge-output 时不调用合并"""
        with patch('sys.argv', [
            'train_main.py', '--model', 'fc_v1', '--self-play', '--games', '1'
        ]):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_trainer = MagicMock()
                mock_get.return_value = mock_trainer

                with patch('ai.self_play_generator.SelfPlayGenerator') as mock_gen_cls:
                    mock_generator = MagicMock()
                    mock_generator.generate.return_value = {
                        "results": [{"winner": 1}],
                        "sample_count": 1,
                        "train_data_dir": "/tmp/fake_data"
                    }
                    mock_gen_cls.return_value = mock_generator
                
                    with patch('ai.train_main.merge_directories') as mock_merge:
                        main()
                
                        mock_generator.generate.assert_called_once()
                        mock_merge.assert_not_called()

    def test_self_play_with_merge_output(self):
        """--self-play --merge-output 自动调用合并"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_data_dir = os.path.join(tmp_dir, "train_data")
            os.makedirs(train_data_dir)
            merge_output = os.path.join(tmp_dir, "merged.json")

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--self-play', '--games', '1',
                '--merge-output', merge_output
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_get.return_value = mock_trainer

                    with patch('ai.self_play_generator.SelfPlayGenerator') as mock_gen_cls:
                        mock_generator = MagicMock()
                        mock_generator.generate.return_value = {
                            "results": [{"winner": 1}],
                            "sample_count": 2,
                            "train_data_dir": train_data_dir
                        }
                        mock_gen_cls.return_value = mock_generator

                        main()

                        mock_generator.generate.assert_called_once()
                        # merge 应被调用且输出文件已生成
                        self.assertTrue(os.path.exists(merge_output))


class TestTrainFromDirModes(unittest.TestCase):
    """测试 --train-from-dir 支持目录和 .json 文件"""

    def test_train_from_dir_json_file(self):
        """--train-from-dir 直接指定 .json 文件时正确加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "data.json")
            sample_data = [
                {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}
            ]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', json_path,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_batch = MagicMock(return_value=0.5)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_batch.assert_called_once()
                    call_args = mock_trainer.train_batch.call_args
                    self.assertEqual(call_args[1]['epochs'], 1)

    def test_train_from_dir_with_merge(self):
        """--train-from-dir 目录含 merge 文件和 sample 时全部加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建 merge 文件（1 条样本）
            merge_path = os.path.join(tmp_dir, "merge_fc_v1.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            # 创建个体 sample 目录（2 条样本）
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                    {"input": [3.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_batch = MagicMock(return_value=0.3)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_batch.assert_called_once()
                    # train_batch 接收的是临时目录列表，验证样本数需检查内部逻辑
                    call_args = mock_trainer.train_batch.call_args
                    self.assertEqual(call_args[1]['epochs'], 1)

    def test_train_from_dir_empty_raises(self):
        """--train-from-dir 指向空目录时应报错退出"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_get.return_value = mock_trainer

                    with self.assertRaises(SystemExit):
                        main()

    def test_train_from_dir_with_mini_batch(self):
        """--train-from-dir 目录支持 mini-batch 训练"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1', '--batch-size', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_mini_batch = MagicMock(return_value=0.2)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_mini_batch.assert_called_once()
                    call_kwargs = mock_trainer.train_mini_batch.call_args[1]
                    self.assertEqual(call_kwargs['batch_size'], 1)
                    self.assertEqual(call_kwargs['epochs'], 1)

    def test_train_from_dir_mini_batch_enables_dense_log(self):
        """--train-from-dir + --batch-size 应自动启用密集日志（dense_log=True）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1', '--batch-size', '32'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_mini_batch = MagicMock(return_value=0.1)
                    mock_get.return_value = mock_trainer

                    main()

                    call_kwargs = mock_trainer.train_mini_batch.call_args[1]
                    self.assertTrue(call_kwargs.get('dense_log', False),
                                    "--train-from-dir + --batch-size 应传 dense_log=True")

    def test_train_from_dir_no_batch_no_dense_log(self):
        """--train-from-dir 不指定 --batch-size 时不传 dense_log（全量更新模式）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_batch = MagicMock(return_value=0.3)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_batch.assert_called_once()
                    call_kwargs = mock_trainer.train_batch.call_args[1]
                    self.assertNotIn('dense_log', call_kwargs,
                                     "全量更新模式不应传 dense_log")


if __name__ == '__main__':
    unittest.main()
