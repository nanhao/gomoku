"""evaluator.py 测试 - _evaluate_files 进程池崩溃回退逻辑

验证当 ProcessPoolExecutor 崩溃时，_evaluate_files 能正确回退到串行模式，
并仅重试尚未成功处理的 record 文件（不重复处理已成功的文件）。
"""

import json
import os
import sys
import tempfile
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.evaluator import Evaluator


def _create_record_file(base_dir, game_name, history_len=2):
    """在 ai_data_dir 下创建一个 record.json 测试文件

    Args:
        base_dir: ai_data_dir 根目录
        game_name: 游戏目录名（如 "game_1"）
        history_len: 历史记录步数

    Returns:
        record.json 的绝对路径
    """
    game_dir = os.path.join(base_dir, game_name)
    os.makedirs(game_dir, exist_ok=True)
    record_path = os.path.join(game_dir, "record.json")

    # 构造最简合法 record：2 步历史 + 胜者 + end_board
    history = []
    for i in range(history_len):
        step = {
            "input": [0] * 225 + [1 if i % 2 == 0 else 2],
            "action": i * 15  # 落子位置
        }
        history.append(step)

    record = {
        "history": history,
        "winner": 1,
        "end_board": [0] * 225
    }
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f)

    return record_path


class TestEvaluateFilesBrokenPoolFallback(unittest.TestCase):
    """测试 _evaluate_files 进程池崩溃时的回退逻辑"""

    def setUp(self):
        """创建临时目录和 Evaluator 实例"""
        self.tmp_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.tmp_dir, "ai_data")
        self.train_data_dir = os.path.join(self.tmp_dir, "train_data")
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
        self.evaluator = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=2
        )

    def tearDown(self):
        """清理临时目录"""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_broken_pool_fallback_processes_remaining_files(self):
        """进程池崩溃后，串行回退只处理剩余未完成的文件

        场景：
        - 提交 3 个 record_file 到进程池
        - 第 1 个成功完成，返回 sample_file
        - 第 2 个触发 BrokenProcessPool
        - 第 3 个未被处理
        - 回退串行模式应只处理第 2 和第 3 个文件
        """
        # 创建 3 个 record 文件
        record_files = [
            _create_record_file(self.ai_data_dir, "game_1"),
            _create_record_file(self.ai_data_dir, "game_2"),
            _create_record_file(self.ai_data_dir, "game_3"),
        ]

        # 构造第 1 个文件成功后的 sample 路径
        sample_file_1 = self.evaluator._get_sample_path(record_files[0])

        # mock 3 个 future
        mock_future_1 = MagicMock()
        mock_future_1.result.return_value = sample_file_1

        mock_future_2 = MagicMock()
        mock_future_2.result.side_effect = BrokenProcessPool("OOM")

        mock_future_3 = MagicMock()  # 不会被迭代到

        # executor.submit 依次返回 3 个不同的 future
        mock_executor = MagicMock()
        mock_executor.submit.side_effect = [mock_future_1, mock_future_2, mock_future_3]
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)

        call_count = [0]
        original_evaluate_record = self.evaluator.evaluate_record

        def track_evaluate_record(record_file):
            """记录串行回退时调用的 record_file"""
            call_count[0] += 1
            return original_evaluate_record(record_file)

        # as_completed 返回顺序：先成功，再崩溃
        with patch('ai.evaluator.ProcessPoolExecutor', return_value=mock_executor), \
             patch('ai.evaluator.as_completed', return_value=[mock_future_1, mock_future_2]), \
             patch.object(self.evaluator, 'evaluate_record', side_effect=track_evaluate_record):

            result = self.evaluator._evaluate_files(record_files, max_workers=2)

        # 应当有 2 个文件在串行回退中被处理（game_2 和 game_3）
        self.assertEqual(call_count[0], 2)
        # 总共应有 3 个 sample_file（1 个并行 + 2 个串行）
        self.assertEqual(len(result["sample_files"]), 3)
        # 第 2 个 record_file 的 BrokenProcessPool 会产生一个 error
        self.assertEqual(len(result["errors"]), 1)

    def test_broken_pool_does_not_reprocess_successful_files(self):
        """进程池崩溃后，已成功的文件不会出现在 remaining_files 中

        这是核心断言：验证 remaining_files 计算逻辑正确排除了已完成的文件。
        如果第 378-382 行的冗余错误代码未被删除，路径混用会导致
        remaining_files 包含所有文件（因路径无法匹配）。
        """
        # 创建 4 个 record 文件
        record_files = [
            _create_record_file(self.ai_data_dir, "game_a"),
            _create_record_file(self.ai_data_dir, "game_b"),
            _create_record_file(self.ai_data_dir, "game_c"),
            _create_record_file(self.ai_data_dir, "game_d"),
        ]

        # 模拟前 2 个成功，第 3 个触发 BrokenProcessPool
        sample_file_a = self.evaluator._get_sample_path(record_files[0])
        sample_file_b = self.evaluator._get_sample_path(record_files[1])

        mock_future_a = MagicMock()
        mock_future_a.result.return_value = sample_file_a

        mock_future_b = MagicMock()
        mock_future_b.result.return_value = sample_file_b

        mock_future_c = MagicMock()
        mock_future_c.result.side_effect = BrokenProcessPool("crash")

        mock_future_d = MagicMock()  # 不会被迭代到

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = [mock_future_a, mock_future_b, mock_future_c, mock_future_d]
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)

        # 收集串行回退时实际处理的 record_files
        serial_processed = []

        original_evaluate_record = self.evaluator.evaluate_record

        def track_serial(record_file):
            serial_processed.append(record_file)
            return original_evaluate_record(record_file)

        with patch('ai.evaluator.ProcessPoolExecutor', return_value=mock_executor), \
             patch('ai.evaluator.as_completed', return_value=[mock_future_a, mock_future_b, mock_future_c]), \
             patch.object(self.evaluator, 'evaluate_record', side_effect=track_serial):

            result = self.evaluator._evaluate_files(record_files, max_workers=4)

        # 串行回退只应处理 game_c 和 game_d（2 个）
        self.assertEqual(len(serial_processed), 2)
        # 已成功的 game_a 和 game_b 不应在串行回退中被重复处理
        self.assertNotIn(record_files[0], serial_processed)
        self.assertNotIn(record_files[1], serial_processed)
        # game_c 和 game_d 应该被处理
        self.assertIn(record_files[2], serial_processed)
        self.assertIn(record_files[3], serial_processed)

    def test_broken_pool_all_successful_before_crash(self):
        """进程池崩溃前所有已提交的 future 都已完成，串行回退只处理未提交的文件"""
        # 创建 3 个 record 文件
        record_files = [
            _create_record_file(self.ai_data_dir, "game_x"),
            _create_record_file(self.ai_data_dir, "game_y"),
            _create_record_file(self.ai_data_dir, "game_z"),
        ]

        sample_file_x = self.evaluator._get_sample_path(record_files[0])
        sample_file_y = self.evaluator._get_sample_path(record_files[1])

        mock_future_x = MagicMock()
        mock_future_x.result.return_value = sample_file_x

        mock_future_y = MagicMock()
        mock_future_y.result.return_value = sample_file_y

        mock_future_z = MagicMock()
        mock_future_z.result.side_effect = BrokenProcessPool("boom")

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = [mock_future_x, mock_future_y, mock_future_z]
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)

        serial_processed = []
        original_evaluate_record = self.evaluator.evaluate_record

        def track_serial(record_file):
            serial_processed.append(record_file)
            return original_evaluate_record(record_file)

        with patch('ai.evaluator.ProcessPoolExecutor', return_value=mock_executor), \
             patch('ai.evaluator.as_completed', return_value=[mock_future_x, mock_future_y, mock_future_z]), \
             patch.object(self.evaluator, 'evaluate_record', side_effect=track_serial):

            result = self.evaluator._evaluate_files(record_files, max_workers=3)

        # 只有 game_z 应在串行回退中被处理
        self.assertEqual(len(serial_processed), 1)
        self.assertIn(record_files[2], serial_processed)
        # game_x 和 game_y 不应重复处理
        self.assertNotIn(record_files[0], serial_processed)
        self.assertNotIn(record_files[1], serial_processed)

    def test_sample_to_record_path_conversion(self):
        """验证 _get_sample_path 的路径转换和回退逻辑中的反向推导一致

        回退逻辑中，从 sample_file 推导 record_file 的方式应与
        _get_sample_path 的正向转换互为逆操作。
        """
        # 模拟路径结构：ai_data/random_1/game_5/record.json
        record_file = os.path.join(self.ai_data_dir, "random_1", "game_5", "record.json")
        sample_file = self.evaluator._get_sample_path(record_file)

        # 正向：record → sample
        self.assertTrue(sample_file.startswith(self.train_data_dir))
        self.assertIn("sample.json", sample_file)
        self.assertNotIn("record.json", sample_file)

        # 反向：sample → record（模拟回退逻辑中的推导）
        rel = os.path.relpath(sample_file, self.train_data_dir)
        recovered_record = os.path.join(
            self.ai_data_dir,
            rel.replace("sample.json", "record.json")
        )
        self.assertEqual(recovered_record, record_file)

    def test_serial_mode_no_broken_pool(self):
        """串行模式（max_workers=1）下不涉及进程池，不会触发回退逻辑"""
        record_files = [
            _create_record_file(self.ai_data_dir, "game_s1"),
        ]

        result = self.evaluator._evaluate_files(record_files, max_workers=1)

        self.assertEqual(len(result["sample_files"]), 1)
        self.assertEqual(len(result["errors"]), 0)


if __name__ == '__main__':
    unittest.main()
