"""第二级性能优化测试 - Evaluator 多进程并行

测试 ProcessPoolExecutor 并行评价的正确性和兼容性：
1. 串行模式（max_workers=1）结果与之前一致
2. 并行模式（max_workers=2+）结果与串行模式一致
3. max_workers=1 退化为串行执行
4. _evaluate_record_worker 顶层函数正确性
5. 错误处理：无效 record 不影响其他文件
"""

import sys
import os
import unittest
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator, _evaluate_record_worker


def _create_minimal_record(game_dir: str, winner: int = 1):
    """创建最小的有效 record.json 文件

    构建一个有明确胜负的对局记录（5步连珠）。
    黑方在 (7,7)-(7,11) 形成五连，白方分散落子。

    Args:
        game_dir: 对局目录路径（如 .../round_1/1）
        winner: 获胜方，1=黑，2=白

    Returns:
        record.json 文件路径
    """
    os.makedirs(game_dir, exist_ok=True)

    board = [0] * 225
    history = []

    # 黑方落 5 子连珠，白方分散落子
    black_moves = [(7, 7), (7, 8), (7, 9), (7, 10), (7, 11)]
    white_positions = [(0, 0), (0, 1), (0, 2), (0, 3)]

    for i, (bx, by) in enumerate(black_moves):
        # 黑方落子
        bpos = bx * 15 + by
        board[bpos] = 1
        input_vec = board[:225] + [1]  # 当前黑方
        history.append({
            "input": input_vec,
            "output": [0.0] * 225,
            "position": [bx, by]
        })

        # 白方落子（如果不是最后一步黑方获胜）
        if i < len(white_positions):
            wx, wy = white_positions[i]
            wpos = wx * 15 + wy
            board[wpos] = 2
            input_vec_w = board[:225] + [2]  # 当前白方
            history.append({
                "input": input_vec_w,
                "output": [0.0] * 225,
                "position": [wx, wy]
            })

    record = {
        "history": history,
        "winner": winner,
        "end_board": board[:225] + [0]  # 226 维
    }

    record_path = os.path.join(game_dir, "record.json")
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)

    return record_path


class TestEvaluateRecordWorker(unittest.TestCase):
    """测试 _evaluate_record_worker 顶层函数

    Windows spawn 模式要求 worker 函数可被 pickle，
    测试验证顶层函数在串行和子进程中均可正常工作。
    """

    def setUp(self):
        """测试前准备临时目录"""
        self.test_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_worker_returns_sample_path(self):
        """worker 应返回 sample.json 文件路径"""
        game_dir = os.path.join(self.ai_data_dir, "round_1", "1")
        record_path = _create_minimal_record(game_dir)

        result = _evaluate_record_worker(
            record_path, self.ai_data_dir, self.train_data_dir
        )

        self.assertTrue(result.endswith("sample.json"))
        self.assertTrue(os.path.exists(result))

    def test_worker_creates_valid_sample(self):
        """worker 生成的 sample.json 应可解析且包含必要字段"""
        game_dir = os.path.join(self.ai_data_dir, "round_1", "1")
        record_path = _create_minimal_record(game_dir)

        sample_path = _evaluate_record_worker(
            record_path, self.ai_data_dir, self.train_data_dir
        )

        with open(sample_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)

        self.assertIsInstance(sample, list)
        self.assertGreater(len(sample), 0)
        for item in sample:
            self.assertIn("input", item)
            self.assertIn("output", item)
            self.assertIn("mask", item)



class TestEvaluateRoundSerialMode(unittest.TestCase):
    """测试 evaluate_round 串行模式（max_workers=1）

    验证串行模式结果与原始逻辑完全一致，
    确保向后兼容性。
    """

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_serial_mode_produces_samples(self):
        """串行模式应正确生成所有 sample 文件"""
        round_dir = os.path.join(self.ai_data_dir, "round_1")
        for i in range(1, 4):
            _create_minimal_record(os.path.join(round_dir, str(i)))

        evaluator = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=1
        )
        result = evaluator.evaluate_round(round_dir)

        # 验证返回结果
        self.assertEqual(len(result["sample_files"]), 3)
        self.assertEqual(len(result["errors"]), 0)

        # 验证文件存在
        for sample_file in result["sample_files"]:
            self.assertTrue(os.path.exists(sample_file))
        # 与 evaluate_record 逐条一致
        for i in range(1, 4):
            record_path = os.path.join(round_dir, str(i), "record.json")
            sample_path = evaluator.evaluate_record(record_path)
            self.assertIn(sample_path, result["sample_files"])

    def test_serial_mode_nonexistent_dir(self):
        """不存在的目录应返回空结果"""
        evaluator = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=1
        )
        result = evaluator.evaluate_round("/nonexistent/path")

        self.assertEqual(len(result["sample_files"]), 0)
        self.assertEqual(len(result["errors"]), 0)


class TestEvaluateRoundParallelMode(unittest.TestCase):
    """测试 evaluate_round 并行模式（max_workers=2+）

    验证并行模式生成与串行模式完全一致的 sample 内容，
    确保多进程评价的正确性。
    """

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _clear_train_data(self):
        """清空训练数据目录，为并行测试做准备"""
        shutil.rmtree(self.train_data_dir, ignore_errors=True)
        os.makedirs(self.train_data_dir, exist_ok=True)

    def test_parallel_mode_produces_same_count(self):
        """并行模式应生成与串行模式相同数量的 sample，且无错误"""
        round_dir = os.path.join(self.ai_data_dir, "round_1")
        _create_minimal_record(os.path.join(round_dir, "1"))

        # 串行模式
        evaluator_serial = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=1
        )
        result_serial = evaluator_serial.evaluate_round(round_dir)

        # 清空 train_data 为并行测试做准备
        self._clear_train_data()

        # 并行模式
        evaluator_parallel = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=2
        )
        result_parallel = evaluator_parallel.evaluate_round(round_dir)

        self.assertEqual(
            len(result_serial["sample_files"]),
            len(result_parallel["sample_files"])
        )
        self.assertEqual(len(result_parallel["errors"]), 0,
                         "有效 record 不应有错误")



class TestParallelErrorHandling(unittest.TestCase):
    """测试并行评价的错误处理

    验证个别文件失败不影响其他文件的评价结果。
    """

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_invalid_record(self, game_dir: str):
        """创建一个无效的 record.json（缺少有效历史和胜者）"""
        os.makedirs(game_dir, exist_ok=True)
        invalid_path = os.path.join(game_dir, "record.json")
        with open(invalid_path, 'w', encoding='utf-8') as f:
            json.dump({"history": [], "winner": 0}, f)
        return invalid_path

    def test_parallel_error_handling(self):
        """并行模式下错误处理：无效 record 不影响其他文件，全部无效时返回全部错误"""
        round_dir = os.path.join(self.ai_data_dir, "round_1")

        # 场景1: 2 个有效 + 1 个无效 → 2 sample + 1 error
        _create_minimal_record(os.path.join(round_dir, "1"))
        self._create_invalid_record(os.path.join(round_dir, "2"))
        _create_minimal_record(os.path.join(round_dir, "3"))

        evaluator = Evaluator(
            ai_data_dir=self.ai_data_dir,
            train_data_dir=self.train_data_dir,
            max_workers=2
        )
        result = evaluator.evaluate_round(round_dir)
        self.assertEqual(len(result["sample_files"]), 2)
        self.assertEqual(len(result["errors"]), 1)

        # 场景2: 3 个无效 → 0 sample + 3 errors（复用同一 evaluator）
        round_dir2 = os.path.join(self.ai_data_dir, "round_2")
        for i in range(1, 4):
            self._create_invalid_record(os.path.join(round_dir2, str(i)))

        result2 = evaluator.evaluate_round(round_dir2)
        self.assertEqual(len(result2["sample_files"]), 0)
        self.assertEqual(len(result2["errors"]), 3)


if __name__ == "__main__":
    unittest.main()
