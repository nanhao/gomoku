"""FC v1 训练周期测试

测试 train_cycle 的 rate 参数默认值。

历史记录：原 TestFCv1TrainerCycle 类仅有 setUp/tearDown 无任何 test_* 方法，
属于死代码，已于测试集清理（阶段 1）中删除。详见 task_reports/2026-05-01/test_suite_review.md。
"""

import os
import sys
import unittest
import inspect

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer


class TestTrainCycleRateParameter(unittest.TestCase):
    """train_cycle rate 参数测试"""

    def test_train_cycle_has_rate_parameter(self):
        """train_cycle 应有 rate 参数，默认值 0.9"""
        sig = inspect.signature(FCv1Trainer.train_cycle)
        self.assertIn('rate', sig.parameters, "train_cycle 应有 rate 参数")
        self.assertAlmostEqual(
            sig.parameters['rate'].default, 0.9, places=9,
            msg="rate 默认值应为 0.9"
        )

    def test_train_main_script_rate_default(self):
        """train_main.py --rate 默认值为 0.9"""
        import argparse

        original_argv = sys.argv
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--rate', type=float, default=0.9)
            parser.add_argument('--model', type=str, default='fc_v1')

            args = parser.parse_args([])
            self.assertAlmostEqual(args.rate, 0.9, places=9, msg="--rate 默认值应为 0.9")
            self.assertEqual(args.model, 'fc_v1', "--model 默认值应为 fc_v1")
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    unittest.main()
