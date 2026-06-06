"""SelfPlayGenerator 单元测试

测试自我对弈数据生成器的数据生成逻辑，包括：
- DataRecorder 使用正确的 ai_name
- AIGameManager.start_ai_vs_ai 被正确调用
- Evaluator.evaluate_round 被正确调用
- 返回结果字典包含所有必需字段
- KeyboardInterrupt 时优雅退出
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.self_play_generator import SelfPlayGenerator


def _create_mock_model():
    """创建 Mock Model

    SelfPlayGenerator 只依赖 model，不依赖 BaseTrainer。
    """
    model = MagicMock()
    model.name = "cnn_v1"
    return model


def _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start):
    """统一 Mock SelfPlayGenerator.generate 的外部依赖

    Args:
        mock_engine: AIEngine mock
        mock_evaluator_cls: Evaluator 类 mock
        mock_recorder_cls: DataRecorder 类 mock
        mock_start: AIGameManager.start_ai_vs_ai mock
    """
    # 模拟对局结果：1 局黑胜、1 局白胜
    mock_start.return_value = [{"winner": 1}, {"winner": 2}]

    # 模拟 DataRecorder
    recorder = MagicMock()
    recorder.get_round_dir.return_value = "/tmp/ai_data/cnn_v1_self_play_1"
    mock_recorder_cls.return_value = recorder

    # 模拟 Evaluator
    evaluator = MagicMock()
    evaluator.evaluate_round.return_value = {
        "sample_files": [
            "/tmp/train_data/cnn_v1_self_play_1/1/sample.json",
            "/tmp/train_data/cnn_v1_self_play_1/2/sample.json"
        ],
        "errors": []
    }
    mock_evaluator_cls.return_value = evaluator


class TestSelfPlayGeneratorDataRecorder(unittest.TestCase):
    """测试 SelfPlayGenerator 创建 DataRecorder 的行为"""

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_creates_data_recorder_with_correct_name(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """DataRecorder 应使用 {model_name}_self_play 作为 ai_name"""
        model = _create_mock_model()
        model.name = "cnn_v1"
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        generator.generate(games=10)

        # 验证 DataRecorder 创建时 ai_name 为 "cnn_v1_self_play"
        mock_recorder_cls.assert_called_once()
        call_kwargs = mock_recorder_cls.call_args[1]
        self.assertEqual(call_kwargs['ai_name'], 'cnn_v1_self_play')


class TestSelfPlayGeneratorGameManager(unittest.TestCase):
    """测试 SelfPlayGenerator 调用 AIGameManager 的行为"""

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_calls_start_ai_vs_ai_with_correct_games(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """start_ai_vs_ai 应被调用，games 参数正确"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        generator.generate(games=50)

        # 验证 start_ai_vs_ai 被调用
        mock_start.assert_called_once()
        # 验证 num_games 参数
        call_kwargs = mock_start.call_args[1]
        self.assertEqual(call_kwargs['num_games'], 50)

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_calls_start_ai_vs_ai_with_same_engine(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """start_ai_vs_ai 应使用相同的 engine 作为 engine1 和 engine2（自我对弈）"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        generator.generate(games=10)

        # 验证 engine1 和 engine2 是同一个实例
        call_kwargs = mock_start.call_args[1]
        self.assertIs(call_kwargs['engine1'], call_kwargs['engine2'])


class TestSelfPlayGeneratorEvaluator(unittest.TestCase):
    """测试 SelfPlayGenerator 调用 Evaluator 的行为"""

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_calls_evaluator_evaluate_round(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """evaluate_round 应被调用，round_dir 正确"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        generator.generate(games=10)

        # 验证 Evaluator 被创建（不含 decay_rate）
        mock_evaluator_cls.assert_called_once()
        eval_call_kwargs = mock_evaluator_cls.call_args[1]
        self.assertNotIn('decay_rate', eval_call_kwargs,
                          "Evaluator 创建时不应包含 decay_rate 参数")
        # 验证 evaluate_round 被调用
        evaluator_instance = mock_evaluator_cls.return_value
        evaluator_instance.evaluate_round.assert_called_once_with("/tmp/ai_data/cnn_v1_self_play_1")

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_passes_max_workers_to_evaluator(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """max_workers 应传递给 Evaluator"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model, max_workers=4)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        generator.generate(games=10)

        # 验证 Evaluator 创建时 max_workers=4
        eval_call_kwargs = mock_evaluator_cls.call_args[1]
        self.assertEqual(eval_call_kwargs['max_workers'], 4)


class TestSelfPlayGeneratorResult(unittest.TestCase):
    """测试 SelfPlayGenerator.generate 返回结果"""

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_returns_correct_result_dict(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """返回字典应包含 round_dir、train_data_dir、sample_dirs、sample_count、results"""
        model = _create_mock_model()
        model.name = "cnn_v1"
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        result = generator.generate(games=10)

        # 验证返回字典包含所有必需字段
        self.assertIn("round_dir", result)
        self.assertIn("train_data_dir", result)
        self.assertIn("sample_dirs", result)
        self.assertIn("sample_count", result)
        self.assertIn("results", result)

        # 验证 round_dir 来自 DataRecorder
        self.assertEqual(result["round_dir"], "/tmp/ai_data/cnn_v1_self_play_1")

        # 验证 sample_dirs 来自 evaluate_round 的 sample_files
        self.assertEqual(len(result["sample_dirs"]), 2)

        # 验证 sample_count
        self.assertEqual(result["sample_count"], 2)

        # 验证 results 来自 start_ai_vs_ai
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["winner"], 1)
        self.assertEqual(result["results"][1]["winner"], 2)


class TestSelfPlayGeneratorKeyboardInterrupt(unittest.TestCase):
    """测试 SelfPlayGenerator 处理 KeyboardInterrupt 的行为"""

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_with_keyboard_interrupt_after_recorder_created(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """Ctrl+C 在 recorder 创建后触发时应优雅退出，返回已有结果"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model)
        _mock_generator_dependencies(mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start)

        # 模拟 start_ai_vs_ai 在执行中抛出 KeyboardInterrupt
        mock_start.side_effect = KeyboardInterrupt()

        # 调用 generate 不应抛出异常
        result = generator.generate(games=10)

        # 验证返回了结果字典（包含中断前的状态）
        self.assertIn("round_dir", result)
        # 中断时 results 应为空（还没执行完对弈）
        self.assertEqual(result["results"], [])

    @patch('ai.self_play_generator.AIGameManager.start_ai_vs_ai')
    @patch('ai.self_play_generator.DataRecorder')
    @patch('ai.self_play_generator.Evaluator')
    @patch('ai.self_play_generator.AIEngine')
    def test_generate_with_keyboard_interrupt_before_recorder_created(
        self, mock_engine, mock_evaluator_cls, mock_recorder_cls, mock_start
    ):
        """Ctrl+C 在 recorder 创建前触发时，不应抛出 NameError，应优雅退出"""
        model = _create_mock_model()
        generator = SelfPlayGenerator(model)

        # 模拟 AIEngine 初始化阶段抛出 KeyboardInterrupt
        # 此时 recorder 尚未创建
        mock_engine.side_effect = KeyboardInterrupt()

        # 调用 generate 不应抛出 NameError 或其他异常
        result = generator.generate(games=10)

        # 验证返回了结果字典
        self.assertIn("round_dir", result)
        self.assertIn("train_data_dir", result)
        self.assertIn("sample_dirs", result)
        self.assertIn("sample_count", result)
        self.assertIn("results", result)

        # recorder 未创建，round_dir 应为空字符串
        self.assertEqual(result["round_dir"], "")
        self.assertEqual(result["train_data_dir"], "")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["sample_dirs"], [])

        # 验证 DataRecorder 从未被创建
        mock_recorder_cls.assert_not_called()


if __name__ == '__main__':
    unittest.main()
