"""SelfPlayCycle 编排逻辑补充测试

测试训练周期的各个阶段衔接、参数传递、结果统计。
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
    trainer.train_batch.return_value = 0.0050
    trainer.train_mini_batch.return_value = 0.0045
    return trainer


def _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
    """统一 Mock SelfPlayCycle.run 的外部依赖"""
    mock_start.return_value = [{"winner": 1}, {"winner": 2}]

    recorder = MagicMock()
    recorder.get_round_dir.return_value = "/tmp/round"
    mock_recorder_cls.return_value = recorder

    evaluator = MagicMock()
    evaluator.evaluate_round.return_value = {
        "sample_files": ["/tmp/train_data/1/sample.json", "/tmp/train_data/2/sample.json"],
        "errors": []
    }
    mock_evaluator_cls.return_value = evaluator


class TestSelfPlayCycleOrchestration(unittest.TestCase):
    """测试 SelfPlayCycle 训练周期编排逻辑"""

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_ai_vs_ai_called_with_correct_params(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """AIGameManager.start_ai_vs_ai 应使用正确的 engine 和 games 参数"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1', 'game_2']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=5, epochs=10)

        # 验证 start_ai_vs_ai 被调用
        mock_start.assert_called_once()
        # 验证 games 参数
        call_kwargs = mock_start.call_args[1]
        self.assertEqual(call_kwargs['num_games'], 5)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_multiple_rounds_iteration(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """多轮训练应正确迭代"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=3, games=2, epochs=10)

        # 验证 3 轮对局（start_ai_vs_ai 调用 3 次）
        self.assertEqual(mock_start.call_count, 3)
        # 验证 3 轮训练
        self.assertEqual(trainer.train_batch.call_count, 3)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_model_saved_after_each_round(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """每轮训练后应保存模型"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=2, games=2, epochs=10)

        # 验证每轮保存一次模型
        self.assertEqual(trainer.model.save.call_count, 2)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_no_sample_dirs_skips_training(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """无 sample 时跳过训练"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        # 修改 evaluator 返回空 sample
        evaluator = MagicMock()
        evaluator.evaluate_round.return_value = {
            "sample_files": [],
            "errors": []
        }
        mock_evaluator_cls.return_value = evaluator

        with patch('os.listdir', return_value=[]):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 不应调用训练
        trainer.train_batch.assert_not_called()
        trainer.train_mini_batch.assert_not_called()

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_evaluator_receives_decay_rate(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """Evaluator 应接收正确的 decay_rate 参数"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10, rate=0.85)

        # 验证 Evaluator 创建时传入 decay_rate=0.85
        eval_call_kwargs = mock_evaluator_cls.call_args[1]
        self.assertEqual(eval_call_kwargs['decay_rate'], 0.85)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_max_workers_passed_to_evaluator(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """max_workers 应传递给 Evaluator"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer, max_workers=4)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        eval_call_kwargs = mock_evaluator_cls.call_args[1]
        self.assertEqual(eval_call_kwargs['max_workers'], 4)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_epochs_passed_to_train_batch(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """epochs 参数应正确传递给 train_batch"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=500)

        call_kwargs = trainer.train_batch.call_args[1]
        self.assertEqual(call_kwargs['epochs'], 500)


class TestSelfPlayCycleLossLogging(unittest.TestCase):
    """测试 SelfPlayCycle 的 loss 记录行为"""

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_loss_logger_logs_each_round(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """每轮训练后应记录 avg_loss"""
        trainer = _create_mock_trainer()
        trainer.train_batch.return_value = 0.1234
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 验证 loss_logger.log 被调用
        trainer.loss_logger.log.assert_called_once()
        call_args = trainer.loss_logger.log.call_args[0]
        self.assertEqual(call_args[0], 1)  # round_num=1
        self.assertAlmostEqual(call_args[1], 0.1234)

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_loss_logger_plot_called_at_end(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """训练结束应生成 loss 曲线图"""
        trainer = _create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        _mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    with patch('ai.self_play_cycle.LossLogger.plot_comparison', return_value=None):
                        cycle.run(rounds=1, games=2, epochs=10)

        trainer.loss_logger.plot.assert_called_once()


if __name__ == '__main__':
    unittest.main()
