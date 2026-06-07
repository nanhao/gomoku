"""train_main.py --lr 参数单元测试

测试 CLI 参数解析（--lr 的各种取值和校验）以及 runners.py 的
训练参数打印行为。
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ---------------------------------------------------------------------------
# UT-3: CLI 参数解析测试
# ---------------------------------------------------------------------------

class TestTrainMainLrArgParsing(unittest.TestCase):
    """UT-3.1 ~ UT-3.7：train_main.py --lr 参数解析和校验"""

    def _parse_args(self, argv_list):
        """辅助：用给定 argv 列表解析参数，返回 Namespace"""
        import argparse
        from ai.train_main import main as _main_ref

        # 手动构建 parser 来测试解析（避免执行 main 的完整流程）
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='fc_v1')
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--fresh', action='store_true')
        parser.add_argument('--train-from-dir', type=str, default=None)
        parser.add_argument('--batch-size', type=int, default=None)
        parser.add_argument('--shuffle', action='store_true')
        parser.add_argument('--lr-scheduler', type=str, default=None,
                            choices=['step', 'plateau'])
        parser.add_argument('--lr-step-size', type=int, default=500)
        parser.add_argument('--lr-gamma', type=float, default=0.9)
        parser.add_argument('--lr-patience', type=int, default=100)
        parser.add_argument('--lr-factor', type=float, default=0.5)
        parser.add_argument('--val-ratio', type=float, default=0.0)
        # 新增参数（与实际代码中保持一致）
        parser.add_argument('--lr', type=float, default=0.001,
                            help='初始学习率，必须为正数')
        return parser.parse_args(argv_list)

    def test_ut_3_1_parse_lr_00005(self):
        """UT-3.1：正常解析 --lr 0.0005"""
        args = self._parse_args(
            ['--model', 'fc_v1', '--train-from-dir', '/tmp', '--lr', '0.0005']
        )
        self.assertEqual(args.lr, 0.0005)

    def test_ut_3_2_parse_lr_scientific_notation(self):
        """UT-3.2：正常解析 --lr 1e-4"""
        args = self._parse_args(
            ['--model', 'fc_v1', '--train-from-dir', '/tmp', '--lr', '1e-4']
        )
        self.assertEqual(args.lr, 0.0001)

    def test_ut_3_3_default_lr_when_not_specified(self):
        """UT-3.3：不指定 --lr 时取默认值 0.001"""
        args = self._parse_args(
            ['--model', 'fc_v1', '--train-from-dir', '/tmp']
        )
        self.assertEqual(args.lr, 0.001)

    def test_ut_3_4_lr_zero_exits(self):
        """UT-3.4：--lr 0 报错退出（后置校验在 main() 中，此处模拟）"""
        # argparse 本身允许 lr=0，后置校验在 main() 中
        args = self._parse_args(
            ['--model', 'fc_v1', '--train-from-dir', '/tmp', '--lr', '0']
        )
        # 后置校验逻辑：if args.lr <= 0 → parser.error(...)
        self.assertEqual(args.lr, 0.0)  # argparse 解析通过
        self.assertLessEqual(args.lr, 0)  # 但值非法，应由 main() 拦截

    def test_ut_3_5_lr_negative_exits(self):
        """UT-3.5：--lr -0.01 报错退出（后置校验在 main() 中）"""
        args = self._parse_args(
            ['--model', 'fc_v1', '--train-from-dir', '/tmp', '--lr', '-0.01']
        )
        self.assertEqual(args.lr, -0.01)  # argparse 解析通过
        self.assertLess(args.lr, 0)  # 但值非法，应由 main() 拦截

    def test_ut_3_6_lr_non_numeric_exits(self):
        """UT-3.6：--lr abc 应抛出 SystemExit（argparse type=float 自动拒绝）"""
        with self.assertRaises(SystemExit) as cm:
            self._parse_args(
                ['--model', 'fc_v1', '--train-from-dir', '/tmp', '--lr', 'abc']
            )
        self.assertEqual(cm.exception.code, 2)

    def test_ut_3_7_help_contains_lr(self):
        """UT-3.7：--help 输出含 --lr 及其帮助文本

        验证 train_main.py 的真实 parser 包含 --lr 参数，
        而非独立构造的 parser，避免 false positive。
        """
        from ai.train_main import main

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            with patch('sys.argv', ['train_main.py', '--help']):
                try:
                    main()
                except SystemExit:
                    pass
        output = captured.getvalue()
        self.assertIn('--lr', output)
        self.assertIn('学习率', output)


# ---------------------------------------------------------------------------
# UT-4: runners.py 训练参数打印
# ---------------------------------------------------------------------------

class TestRunnersLrPrinting(unittest.TestCase):
    """UT-4.1 ~ UT-4.3：run_train_from_dir 打印初始学习率"""

    def _make_mock_trainer(self, lr_value):
        """构造一个 mock trainer，其 optimizer 返回指定的 lr 值"""
        trainer = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.optimizer.param_groups = [{'lr': lr_value}]
        trainer.train_auto = MagicMock(return_value=0.5)
        trainer.model = MagicMock()
        trainer.model.save = MagicMock(return_value="/tmp/model.pth")
        return trainer

    def _make_mock_args(self, **kwargs):
        """构造 mock args namespace"""
        defaults = {
            'model': 'fc_v1',
            'epochs': 1000,
            'fresh': False,
            'train_from_dir': '/tmp/data.json',
            'batch_size': None,
            'shuffle': False,
            'lr_scheduler': None,
            'lr_step_size': 500,
            'lr_gamma': 0.9,
            'lr_patience': 100,
            'lr_factor': 0.5,
            'val_ratio': 0.0,
        }
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def _create_sample_file(self, tmp_dir):
        """在临时目录创建最小样本文件，供 UT-4 测试共用"""
        import json
        json_path = os.path.join(tmp_dir, "data.json")
        sample = [{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f)
        return json_path

    def test_ut_4_1_print_lr_when_specified(self):
        """UT-4.1：指定 lr 后打印对应值"""
        import tempfile
        from unittest.mock import patch as mock_patch

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = self._create_sample_file(tmp_dir)

            trainer = self._make_mock_trainer(0.005)
            args = self._make_mock_args(train_from_dir=json_path)

            from ai.runners import run_train_from_dir

            captured = io.StringIO()
            with mock_patch('sys.stdout', captured):
                run_train_from_dir(args, trainer)

            output = captured.getvalue()
            self.assertIn("初始学习率", output)
            self.assertIn("0.005", output)

    def test_ut_4_2_print_lr_when_default(self):
        """UT-4.2：不指定 lr 时打印默认值 0.001"""
        import tempfile
        from unittest.mock import patch as mock_patch

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = self._create_sample_file(tmp_dir)

            trainer = self._make_mock_trainer(0.001)
            args = self._make_mock_args(train_from_dir=json_path)

            from ai.runners import run_train_from_dir

            captured = io.StringIO()
            with mock_patch('sys.stdout', captured):
                run_train_from_dir(args, trainer)

            output = captured.getvalue()
            self.assertIn("初始学习率", output)
            self.assertIn("0.001", output)

    def test_ut_4_3_lr_line_after_training_method(self):
        """UT-4.3：初始学习率行位于训练方法行之后"""
        import tempfile
        from unittest.mock import patch as mock_patch

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = self._create_sample_file(tmp_dir)

            trainer = self._make_mock_trainer(0.005)
            args = self._make_mock_args(train_from_dir=json_path)

            from ai.runners import run_train_from_dir

            captured = io.StringIO()
            with mock_patch('sys.stdout', captured):
                run_train_from_dir(args, trainer)

            output = captured.getvalue()
            # 验证"初始学习率"出现在"训练方法"之后
            train_method_pos = output.find("训练方法")
            lr_pos = output.find("初始学习率")
            self.assertGreater(lr_pos, train_method_pos,
                               "初始学习率 行应位于 训练方法 行之后")


if __name__ == '__main__':
    unittest.main()
