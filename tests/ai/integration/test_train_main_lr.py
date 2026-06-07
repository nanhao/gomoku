"""train_main.py --lr 参数集成测试

测试从 CLI → get_trainer → BaseTrainer 的完整 lr 参数传递链路。
使用 mock 隔离子进程和文件系统，验证参数正确传递和错误处理。
"""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ---------------------------------------------------------------------------
# 共享测试辅助函数
# ---------------------------------------------------------------------------

def _create_sample_file(tmp_dir):
    """在临时目录创建最小样本文件，返回文件路径

    供多个测试类共用，避免重复代码。
    """
    json_path = os.path.join(tmp_dir, "data.json")
    sample = [{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f)
    return json_path


# ---------------------------------------------------------------------------
# IT-1: 端到端参数传递
# ---------------------------------------------------------------------------

class TestEndToEndLrFlow(unittest.TestCase):
    """IT-1.1 ~ IT-1.4：--lr 值完整传递到训练器"""

    def test_it_1_1_lr_passed_to_trainer(self):
        """IT-1.1：--lr 0.005 正确传递到 get_trainer 和 optimizer"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = _create_sample_file(tmp_dir)

            mock_trainer = MagicMock()
            mock_trainer.train_auto = MagicMock(return_value=0.5)
            mock_trainer.optimizer = MagicMock()
            mock_trainer.optimizer.param_groups = [{'lr': 0.005}]
            mock_trainer.model = MagicMock()
            mock_trainer.model.save = MagicMock(return_value="/tmp/model.pth")

            # 捕获 stdout 以验证打印
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                with patch('sys.argv', [
                    'train_main.py', '--model', 'fc_v1',
                    '--train-from-dir', json_path,
                    '--lr', '0.005', '--epochs', '1'
                ]):
                    with patch('ai.train_main.get_trainer') as mock_get:
                        mock_get.return_value = mock_trainer
                        from ai.train_main import main
                        main()

                # ① 验证 get_trainer 收到的 lr 参数
                call_kwargs = mock_get.call_args[1]
                self.assertIn('lr', call_kwargs)
                self.assertEqual(call_kwargs['lr'], 0.005)

                # ② 验证打印信息含 0.005
                output = captured.getvalue()
                self.assertIn("初始学习率", output)
                self.assertIn("0.005", output)

                # ③ 验证训练正常完成
                mock_trainer.train_auto.assert_called_once()

    def test_it_1_2_default_lr_when_not_specified(self):
        """IT-1.2：不指定 --lr 时使用默认值 0.001"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = _create_sample_file(tmp_dir)

            mock_trainer = MagicMock()
            mock_trainer.train_auto = MagicMock(return_value=0.3)
            mock_trainer.optimizer = MagicMock()
            mock_trainer.optimizer.param_groups = [{'lr': 0.001}]
            mock_trainer.model = MagicMock()
            mock_trainer.model.save = MagicMock(return_value="/tmp/model.pth")

            captured = io.StringIO()
            with patch('sys.stdout', captured):
                with patch('sys.argv', [
                    'train_main.py', '--model', 'fc_v1',
                    '--train-from-dir', json_path,
                    '--epochs', '1'
                ]):
                    with patch('ai.train_main.get_trainer') as mock_get:
                        mock_get.return_value = mock_trainer
                        from ai.train_main import main
                        main()

                # ① 验证 get_trainer 收到的 lr 为默认值 0.001
                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs.get('lr'), 0.001)

                # ② 验证打印信息含 0.001
                output = captured.getvalue()
                self.assertIn("初始学习率", output)
                self.assertIn("0.001", output)

                # ③ 验证训练正常完成
                mock_trainer.train_auto.assert_called_once()

    def test_it_1_3_lr_with_step_scheduler(self):
        """IT-1.3：--lr 配合 --lr-scheduler step 传递正确"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = _create_sample_file(tmp_dir)

            mock_trainer = MagicMock()
            mock_trainer.train_auto = MagicMock(return_value=0.4)
            mock_trainer.optimizer = MagicMock()
            mock_trainer.optimizer.param_groups = [{'lr': 0.01}]
            mock_trainer.model = MagicMock()
            mock_trainer.model.save = MagicMock(return_value="/tmp/model.pth")

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1',
                '--train-from-dir', json_path,
                '--lr', '0.01',
                '--lr-scheduler', 'step',
                '--lr-step-size', '5',
                '--lr-gamma', '0.5',
                '--epochs', '12'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_get.return_value = mock_trainer
                    from ai.train_main import main
                    main()

                # 验证 lr 和 scheduler 参数同时传入
                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs['lr'], 0.01)
                self.assertEqual(call_kwargs['lr_scheduler_type'], 'step')
                self.assertEqual(call_kwargs['lr_step_size'], 5)
                self.assertEqual(call_kwargs['lr_gamma'], 0.5)

    def test_it_1_4_lr_with_plateau_scheduler(self):
        """IT-1.4：--lr 配合 --lr-scheduler plateau 传递正确"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = _create_sample_file(tmp_dir)

            mock_trainer = MagicMock()
            mock_trainer.train_auto = MagicMock(return_value=0.3)
            mock_trainer.optimizer = MagicMock()
            mock_trainer.optimizer.param_groups = [{'lr': 0.01}]
            mock_trainer.model = MagicMock()
            mock_trainer.model.save = MagicMock(return_value="/tmp/model.pth")

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1',
                '--train-from-dir', json_path,
                '--lr', '0.01',
                '--lr-scheduler', 'plateau',
                '--lr-factor', '0.5',
                '--lr-patience', '1000',
                '--epochs', '3'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_get.return_value = mock_trainer
                    from ai.train_main import main
                    main()

                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs['lr'], 0.01)
                self.assertEqual(call_kwargs['lr_scheduler_type'], 'plateau')
                self.assertEqual(call_kwargs['lr_factor'], 0.5)
                self.assertEqual(call_kwargs['lr_patience'], 1000)


# ---------------------------------------------------------------------------
# IT-2: 向后兼容
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    """IT-2.1 ~ IT-2.2：向后兼容性测试"""

    def test_it_2_1_intent6_call_unaffected(self):
        """IT-2.1：意图 6 中 train_main 调用不受 --lr 新增影响"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = _create_sample_file(tmp_dir)

            mock_trainer = MagicMock()
            mock_trainer.train_auto = MagicMock(return_value=0.5)
            mock_trainer.optimizer = MagicMock()
            mock_trainer.optimizer.param_groups = [{'lr': 0.001}]
            mock_trainer.model = MagicMock()
            mock_trainer.model.save = MagicMock(return_value="/tmp/model.pth")

            # 模拟意图 6 的命令（无 --lr）
            with patch('sys.argv', [
                'train_main.py', '--model', 'cnn_v3.5',
                '--train-from-dir', json_path,
                '--epochs', '1', '--batch-size', '4096', '--shuffle'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_get.return_value = mock_trainer
                    from ai.train_main import main
                    main()

                # ① 不指定 --lr 时正常执行
                call_kwargs = mock_get.call_args[1]
                self.assertEqual(call_kwargs.get('lr'), 0.001)

                # ② 训练正常完成
                mock_trainer.train_auto.assert_called_once()
                # ③ optimizer lr 为默认值
                self.assertEqual(
                    mock_trainer.optimizer.param_groups[0]['lr'], 0.001
                )

    def test_it_2_2_get_trainer_indirect_call_unaffected(self):
        """IT-2.2：通过 get_trainer 间接创建训练器，不传 lr 行为不变"""
        from ai.registry import get_trainer

        trainer = get_trainer("fc_v1")
        # 不传 lr 时使用默认值
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.001)


# ---------------------------------------------------------------------------
# IT-3: 错误处理
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):
    """IT-3.1 ~ IT-3.2：错误处理验证"""

    def test_it_3_1_lr_zero_prevents_training(self):
        """IT-3.1：--lr 0 时训练不启动，argparse 报错退出"""
        with patch('sys.argv', [
            'train_main.py', '--model', 'fc_v1',
            '--train-from-dir', '/tmp/data.json',
            '--lr', '0'
        ]):
            with patch('ai.train_main.get_trainer') as mock_get:
                mock_get.return_value = MagicMock()
                from ai.train_main import main
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 2)
                # get_trainer 不应被调用
                mock_get.assert_not_called()

    def test_it_3_2_lr_non_numeric_prevents_training(self):
        """IT-3.2：--lr abc 时训练不启动，argparse 报错退出"""
        with patch('sys.argv', [
            'train_main.py', '--model', 'fc_v1',
            '--train-from-dir', '/tmp/data.json',
            '--lr', 'abc'
        ]):
            with patch('ai.train_main.get_trainer') as mock_get:
                with self.assertRaises(SystemExit) as cm:
                    from ai.train_main import main
                    main()
                self.assertEqual(cm.exception.code, 2)
                mock_get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
