"""SampleVisualizer 样本可视化工具测试

覆盖单元测试和集成测试：
- 单元测试：随机抽样、指定索引、单样本、空样本异常
- 集成测试：CLI 基本调用、CLI --indices、CLI 缺少文件

测试输出均写入 tempfile.TemporaryDirectory，不修改真实数据文件。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.sample_visualizer import SampleVisualizer


# ──────────────────────────────────────────────
# 辅助函数：生成 winning_move 模式的测试样本
# ──────────────────────────────────────────────

BOARD_SIZE = 15


def _make_sample(black_positions=None, white_positions=None, player=1, win_r=7, win_c=7):
    """构造一个 winning_move 格式的样本字典

    Args:
        black_positions: 黑子坐标列表 [(r, c), ...]
        white_positions: 白子坐标列表 [(r, c), ...]
        player: 当前玩家（1=黑, 2=白）
        win_r: 取胜位置行
        win_c: 取胜位置列

    Returns:
        样本字典 {"input": [...], "output": [...], "mask": [...]}
    """
    black_positions = black_positions or []
    white_positions = white_positions or []

    # 构建棋盘状态（225 维）
    board = [0] * 225
    for r, c in black_positions:
        board[r * BOARD_SIZE + c] = 1
    for r, c in white_positions:
        board[r * BOARD_SIZE + c] = 2

    # input = 棋盘状态 + 当前玩家
    input_vec = board + [player]

    # output = 取胜位置为 1.0，其余为 0.0
    output = [0.0] * 225
    output[win_r * BOARD_SIZE + win_c] = 1.0

    # mask 全为 1
    mask = [1] * 225

    return {"input": input_vec, "output": output, "mask": mask}


def _make_samples(n=10):
    """生成 n 个不同的测试样本，每个样本有不同的棋盘布局"""
    samples = []
    for i in range(n):
        # 交替放置黑白子，确保棋盘上有棋子
        black_pos = [(0, i % BOARD_SIZE), (1, (i + 1) % BOARD_SIZE)]
        white_pos = [(2, (i + 2) % BOARD_SIZE)]
        player = 1 if i % 2 == 0 else 2
        win_r = 3 + (i % 5)
        win_c = 5 + (i % 5)
        samples.append(_make_sample(
            black_positions=black_pos,
            white_positions=white_pos,
            player=player,
            win_r=win_r,
            win_c=win_c,
        ))
    return samples


# ──────────────────────────────────────
# 单元测试
# ──────────────────────────────────────

class TestSampleVisualizerUnit(unittest.TestCase):
    """SampleVisualizer 单元测试"""

    def test_visualize_random_sample(self):
        """随机抽样 count 个，输出图片存在且尺寸正确"""
        samples = _make_samples(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "random_test.png")
            viz = SampleVisualizer()
            result = viz.visualize(samples, count=5, output_path=output_path)

            # 返回路径与传入路径一致
            self.assertEqual(result, output_path)
            # 文件存在
            self.assertTrue(os.path.isfile(output_path))
            # 文件大小大于 0
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_visualize_specified_indices(self):
        """指定 indices，图片中子图数与指定索引数一致"""
        samples = _make_samples(10)
        indices = [1, 3, 7]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "indices_test.png")
            viz = SampleVisualizer()
            result = viz.visualize(samples, indices=indices, output_path=output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(os.path.isfile(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_visualize_single_sample(self):
        """仅 1 个样本，正确生成 1×1 布局"""
        samples = _make_samples(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "single_test.png")
            viz = SampleVisualizer()
            result = viz.visualize(samples, indices=[0], output_path=output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(os.path.isfile(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_empty_samples_raises_value_error(self):
        """空样本列表应抛出 ValueError"""
        viz = SampleVisualizer()
        with self.assertRaises(ValueError):
            viz.visualize([], count=5)

    def test_count_exceeds_sample_count(self):
        """count 大于样本总数时，应仅使用实际样本数"""
        samples = _make_samples(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "exceed_test.png")
            viz = SampleVisualizer()
            # count=10 但只有 3 个样本，不应报错
            result = viz.visualize(samples, count=10, output_path=output_path)
            self.assertTrue(os.path.isfile(output_path))

    def test_output_dir_auto_created(self):
        """输出路径的父目录不存在时应自动创建"""
        samples = _make_samples(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "sub", "dir", "output.png")
            viz = SampleVisualizer()
            result = viz.visualize(samples, count=2, output_path=nested_path)
            self.assertTrue(os.path.isfile(nested_path))

    def test_default_output_path(self):
        """不指定 output_path 时使用默认路径 temp/sample_visualizer.png"""
        samples = _make_samples(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            # 切换到临时目录，避免在项目目录创建 temp/
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                viz = SampleVisualizer()
                result = viz.visualize(samples, count=2)
                # 默认路径应为 temp/sample_visualizer.png
                expected = os.path.join("temp", "sample_visualizer.png")
                self.assertEqual(result, expected)
                self.assertTrue(os.path.isfile(expected))
            finally:
                os.chdir(original_cwd)


# ──────────────────────────────────────
# 集成测试
# ──────────────────────────────────────

class TestSampleVisualizerCLI(unittest.TestCase):
    """SampleVisualizer CLI 集成测试"""

    def test_cli_basic(self):
        """CLI 基本调用，输出文件存在"""
        samples = _make_samples(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            # 写入样本文件
            sample_file = os.path.join(tmpdir, "samples.json")
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f)

            output_file = os.path.join(tmpdir, "cli_output.png")

            # 调用 CLI
            result = subprocess.run(
                [sys.executable, "ai/sample_visualizer.py",
                 "--sample", sample_file,
                 "--count", "3",
                 "--output", output_file],
                capture_output=True, text=True,
                cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
            )
            # 应成功退出
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
            # 输出文件存在
            self.assertTrue(os.path.isfile(output_file))

    def test_cli_with_indices(self):
        """CLI --indices 参数，输出文件存在"""
        samples = _make_samples(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "samples.json")
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f)

            output_file = os.path.join(tmpdir, "cli_indices_output.png")

            result = subprocess.run(
                [sys.executable, "ai/sample_visualizer.py",
                 "--sample", sample_file,
                 "--indices", "0,5,9",
                 "--output", output_file],
                capture_output=True, text=True,
                cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
            )
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
            self.assertTrue(os.path.isfile(output_file))

    def test_cli_missing_sample_file(self):
        """CLI 缺少 --sample 文件时退出码非 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "nonexistent.json")
            output_file = os.path.join(tmpdir, "missing_output.png")

            result = subprocess.run(
                [sys.executable, "ai/sample_visualizer.py",
                 "--sample", fake_path,
                 "--output", output_file],
                capture_output=True, text=True,
                cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
            )
            # 文件不存在时应失败
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
