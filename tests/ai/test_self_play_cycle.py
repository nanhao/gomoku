"""SelfPlayCycle 单元测试

测试 SelfPlayCycle 的训练方法切换、参数传递及日志行为。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.self_play_cycle import SelfPlayCycle


class TestSelfPlayCycleBatchSwitch(unittest.TestCase):
    """测试 SelfPlayCycle 根据 batch_size 切换训练方法"""

    def _create_mock_trainer(self):
        """创建 Mock Trainer，用于隔离外部依赖"""
        trainer = MagicMock()
        trainer.model_name = "fc_v1"
        trainer.loss_logger = MagicMock()
        trainer.model.save.return_value = "/tmp/model.pth"
        # 训练方法返回 float，避免 format 错误
        trainer.train_batch.return_value = 0.0050
        trainer.train_mini_batch.return_value = 0.0045
        return trainer

    def _mock_cycle_dependencies(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """统一 Mock SelfPlayCycle.run 的外部依赖"""
        mock_start.return_value = [{"winner": 1}, {"winner": 2}]

        recorder = MagicMock()
        recorder.get_round_dir.return_value = "/tmp/round"
        mock_recorder_cls.return_value = recorder

        evaluator = MagicMock()
        # evaluate_round 返回格式：{"sample_files": [...], "errors": [...]}
        evaluator.evaluate_round.return_value = {
            "sample_files": ["/tmp/train_data/1/sample.json", "/tmp/train_data/2/sample.json"],
            "errors": []
        }
        mock_evaluator_cls.return_value = evaluator

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_with_batch_size_calls_train_mini_batch(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """提供 batch_size 时调用 train_mini_batch，不调用 train_batch"""
        trainer = self._create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        self._mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1', 'game_2']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10, batch_size=32, shuffle=True)

        # 断言 train_mini_batch 被调用，train_batch 未被调用
        trainer.train_mini_batch.assert_called_once()
        trainer.train_batch.assert_not_called()

        # 验证 batch_size 和 shuffle 正确传递
        call_kwargs = trainer.train_mini_batch.call_args[1]
        self.assertEqual(call_kwargs['batch_size'], 32)
        self.assertTrue(call_kwargs['shuffle'])

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_without_batch_size_calls_train_batch(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """不提供 batch_size 时调用 train_batch，不调用 train_mini_batch"""
        trainer = self._create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        self._mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1', 'game_2']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10)

        # 断言 train_batch 被调用，train_mini_batch 未被调用
        trainer.train_batch.assert_called_once()
        trainer.train_mini_batch.assert_not_called()

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_with_batch_size_false_shuffle(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """提供 batch_size 但不传 shuffle 时，shuffle 默认为 False"""
        trainer = self._create_mock_trainer()
        cycle = SelfPlayCycle(trainer)
        self._mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('os.listdir', return_value=['game_1', 'game_2']):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.exists', return_value=True):
                    cycle.run(rounds=1, games=2, epochs=10, batch_size=16)

        trainer.train_mini_batch.assert_called_once()
        call_kwargs = trainer.train_mini_batch.call_args[1]
        self.assertEqual(call_kwargs['batch_size'], 16)
        self.assertFalse(call_kwargs['shuffle'])

    @patch('ai.self_play_cycle.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_cycle.DataRecorder')
    @patch('ai.self_play_cycle.Evaluator')
    @patch('ai.self_play_cycle.AIEngine')
    def test_session_dir_passed_to_both_loggers(self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
        """session_dir 应传递给内层和外层 LossLogger"""
        trainer = self._create_mock_trainer()
        trainer.loss_logger = None  # 让 SelfPlayCycle 自动创建
        cycle = SelfPlayCycle(trainer)
        self._mock_cycle_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        with patch('ai.self_play_cycle.LossLogger') as mock_loss_logger:
            logger_instance = MagicMock()
            mock_loss_logger.return_value = logger_instance

            with patch('os.listdir', return_value=['game_1']):
                with patch('os.path.isdir', return_value=True):
                    with patch('os.path.exists', return_value=True):
                        cycle.run(rounds=1, games=1, epochs=10, batch_size=8)

            # 外层 LossLogger 应被创建，且携带 session_dir
            outer_call = mock_loss_logger.call_args_list[0]
            self.assertIsNotNone(outer_call[1].get('session_dir'))
            self.assertEqual(outer_call[1]['log_type'], 'round_level')

            # 验证 train_mini_batch 被调用时传了 session_dir
            call_kwargs = trainer.train_mini_batch.call_args[1]
            self.assertIsNotNone(call_kwargs.get('session_dir'))


if __name__ == '__main__':
    unittest.main()
