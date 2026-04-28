"""SelfPlayCycle 编排逻辑补充测试

测试训练周期的各个阶段衔接、参数传递、结果统计。
对弈+评价步骤已委托给 run_self_play_and_evaluate，本测试重点验证编排逻辑。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.self_play_cycle import SelfPlayCycle


def _create_mock_trainer():
    """创建 Mock Trainer"""
    trainer = MagicMock()
    trainer.model_name = "fc_v1"
    trainer.loss_logger = MagicMock()
    trainer.model.save.return_value = "/tmp/model.pth"
    # train_auto 是 SelfPlayCycle.run 的入口方法，需要返回 float
    trainer.train_auto.return_value = 0.0050
    return trainer


def _mock_run_self_play_and_evaluate(mock_func):
    """统一 Mock run_self_play_and_evaluate 的返回值

    Args:
        mock_func: run_self_play_and_evaluate 的 mock 对象
    """
    mock_func.return_value = {
        "round_dir": "/tmp/round",
        "train_data_dir": "/tmp/train_data/round",
        "sample_dirs": ["/tmp/train_data/1", "/tmp/train_data/2"],
        "sample_count": 2,
        "results": [{"winner": 1}, {"winner": 2}],
        "recorder": MagicMock(),
        "interrupted": False
    }


class TestSelfPlayCycleOrchestration(unittest.TestCase):
    """测试 SelfPlayCycle 训练周期编排逻辑"""

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_calls_run_self_play_and_evaluate_with_correct_params(self, mock_func):
        """run_self_play_and_evaluate 应被正确调用，传入 games、rate 等参数"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer, max_workers=4)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1', 'game_2']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=5, epochs=10, rate=0.85)

        # 验证 run_self_play_and_evaluate 被调用
        mock_func.assert_called_once()
        call_kwargs = mock_func.call_args[1]
        # 验证关键参数
        self.assertEqual(call_kwargs['games'], 5)
        self.assertEqual(call_kwargs['rate'], 0.85)
        self.assertEqual(call_kwargs['max_workers'], 4)
        self.assertEqual(call_kwargs['name_prefix'], 'cycle_1')

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_multiple_rounds_iteration(self, mock_func):
        """多轮训练应正确迭代，每轮调用一次 run_self_play_and_evaluate"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=3, games=2, epochs=10)

        # 验证 3 轮对弈（run_self_play_and_evaluate 调用 3 次）
        self.assertEqual(mock_func.call_count, 3)
        # 验证 3 轮训练（train_auto 被调用 3 次）
        self.assertEqual(trainer.train_auto.call_count, 3)

        # 验证每轮的 name_prefix 递增
        call_kwargs_list = [call[1] for call in mock_func.call_args_list]
        self.assertEqual(call_kwargs_list[0]['name_prefix'], 'cycle_1')
        self.assertEqual(call_kwargs_list[1]['name_prefix'], 'cycle_2')
        self.assertEqual(call_kwargs_list[2]['name_prefix'], 'cycle_3')

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_model_saved_after_each_round(self, mock_func):
        """每轮训练后应保存模型"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=2, games=2, epochs=10)

        # 验证每轮保存一次模型
        self.assertEqual(trainer.model.save.call_count, 2)

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_no_sample_dirs_skips_training(self, mock_func):
        """无 sample 时跳过训练"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)

        # 修改 mock 返回空 sample
        mock_func.return_value = {
            "round_dir": "/tmp/round",
            "train_data_dir": "/tmp/train_data/round",
            "sample_dirs": [],
            "sample_count": 0,
            "results": [],
            "recorder": MagicMock(),
            "interrupted": False
        }

        with patch('os.listdir', return_value=[]):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 不应调用训练
        trainer.train_auto.assert_not_called()

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_rate_passed_to_run_self_play(self, mock_func):
        """rate 参数应传递给 run_self_play_and_evaluate"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10, rate=0.85)

        # 验证 rate 正确传递
        call_kwargs = mock_func.call_args[1]
        self.assertEqual(call_kwargs['rate'], 0.85)

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_max_workers_passed_to_run_self_play(self, mock_func):
        """max_workers 应传递给 run_self_play_and_evaluate"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer, max_workers=4)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 验证 max_workers 正确传递
        call_kwargs = mock_func.call_args[1]
        self.assertEqual(call_kwargs['max_workers'], 4)

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_epochs_passed_to_train_auto(self, mock_func):
        """epochs 参数应正确传递给 train_auto"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=500)

        call_kwargs = trainer.train_auto.call_args[1]
        self.assertEqual(call_kwargs['epochs'], 500)

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_interrupted_result_breaks_loop(self, mock_func):
        """对弈阶段中断时应跳出循环"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)

        # 第一轮中断
        mock_func.return_value = {
            "round_dir": "/tmp/round",
            "train_data_dir": "/tmp/train_data/round",
            "sample_dirs": [],
            "sample_count": 0,
            "results": [],
            "recorder": MagicMock(),
            "interrupted": True
        }

        with patch('os.listdir', return_value=[]):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=3, games=2, epochs=10)

        # 只调用了一次 run_self_play_and_evaluate（第一轮中断后跳出）
        self.assertEqual(mock_func.call_count, 1)
        # 不应调用训练
        trainer.train_auto.assert_not_called()

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_project_root_passed_to_run_self_play(self, mock_func):
        """project_root 应传递给 run_self_play_and_evaluate"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10, project_root="/tmp/project")

        # 验证 project_root 和计算后的 ai_data_dir
        call_kwargs = mock_func.call_args[1]
        self.assertEqual(call_kwargs['project_root'], "/tmp/project")
        self.assertEqual(call_kwargs['ai_data_dir'], os.path.join("/tmp/project", "ai", "ai_data"))
        self.assertEqual(call_kwargs['train_data_dir'], os.path.join("/tmp/project", "ai", "train_data"))


class TestSelfPlayCycleLossLogging(unittest.TestCase):
    """测试 SelfPlayCycle 的 loss 记录行为"""

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_loss_logger_logs_each_round(self, mock_func):
        """每轮训练后应记录 avg_loss"""
        trainer = _create_mock_trainer()
        trainer.train_auto.return_value = 0.1234
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 验证 loss_logger.log 被调用
        trainer.loss_logger.log.assert_called_once()
        call_args = trainer.loss_logger.log.call_args[0]
        self.assertEqual(call_args[0], 1)  # round_num=1
        self.assertAlmostEqual(call_args[1], 0.1234)

    @patch('ai.self_play_cycle.run_self_play_and_evaluate')
    def test_loss_logger_plot_called_at_end(self, mock_func):
        """训练结束应生成 loss 曲线图"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_run_self_play_and_evaluate(mock_func)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    with patch('ai.self_play_cycle.LossLogger.plot_comparison', return_value=None):
                        cycle.run(rounds=1, games=2, epochs=10)

        trainer.loss_logger.plot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
