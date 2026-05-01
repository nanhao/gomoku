"""SampleCreator winner_path 模式单元测试

覆盖第三十八次追加的 winner_path 模式（需求 key_points.md 第 643-644 行）。

测试规则（参见 design/sample_creator_design.md "第三十八次追加"）：
- 仅选取获胜方的落子步（黑胜 → history 偶数索引；白胜 → 奇数索引）
- **剔除获胜方的第 1 步**（开局）
- output：落子位=1.0，其余=0
- mask：全 1
- 平局/未结束/空 history：整局跳过
- 强制不做 D4 增强（即使 augment=True 也不增强）
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.sample_creator import SampleCreator


def _make_history(n_steps: int, start_input_value: float = 0.0) -> list:
    """构造长度为 n_steps 的 history，每步 input 的填充值不同，便于测试比对

    每一步的 input 填充 float(start_input_value + i)，position = [i // 15, i % 15]
    """
    history = []
    for i in range(n_steps):
        fill = float(start_input_value + i)
        history.append({
            "input": [fill] * 226,
            "output": [0.0] * 225,
            "position": [i // 15, i % 15],
        })
    return history


def _write_record(tmpdir: str, record: dict, filename: str = "record.json") -> str:
    """把 record 写到临时目录，返回文件绝对路径"""
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f)
    return path


class TestWinnerPathBlackWins(unittest.TestCase):
    """黑胜、history=5 步 → 获胜方 3 步，剔首 → 2 个 sample"""

    def test_black_wins_generates_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 5 步：黑(0) 白(1) 黑(2) 白(3) 黑(4)，黑胜
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # 获胜方步 = [0, 2, 4]，剔首 → [2, 4] → 2 个 sample
            self.assertEqual(len(samples), 2)


class TestWinnerPathWhiteWins(unittest.TestCase):
    """白胜、history=6 步 → 获胜方 3 步，剔首 → 2 个 sample"""

    def test_white_wins_generates_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 6 步：黑(0) 白(1) 黑(2) 白(3) 黑(4) 白(5)，白胜
            record = {"winner": 2, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # 获胜方步 = [1, 3, 5]，剔首 → [3, 5] → 2 个 sample
            self.assertEqual(len(samples), 2)


class TestWinnerPathInputAlignment(unittest.TestCase):
    """每个 sample 的 input 与对应 history[i]["input"] 一致"""

    def test_each_sample_input_is_per_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、6 步：获胜方步 = [0, 2, 4]，剔首 → [2, 4]
            record = {"winner": 1, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(len(samples), 2)
            # _make_history 中 step i 的 input 全为 float(i)
            self.assertEqual(samples[0]["input"], [2.0] * 226)
            self.assertEqual(samples[1]["input"], [4.0] * 226)


class TestWinnerPathOutputOneHot(unittest.TestCase):
    """每个 sample output 落子位=1.0，其它=0"""

    def test_output_one_hot_at_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 手工构造一局黑胜（3 步），让后两步位置可预测
            history = [
                {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},   # 黑，剔除
                {"input": [1.0] * 226, "output": [0.0] * 225, "position": [0, 1]},   # 白
                {"input": [2.0] * 226, "output": [0.0] * 225, "position": [3, 4]},   # 黑，保留
            ]
            record = {"winner": 1, "history": history}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # 获胜方步=[0,2]，剔首 → [2]
            self.assertEqual(len(samples), 1)
            expected_output = [0.0] * 225
            expected_output[3 * 15 + 4] = 1.0  # position [3, 4]
            self.assertEqual(samples[0]["output"], expected_output)


class TestWinnerPathMask(unittest.TestCase):
    """mask 全 1"""

    def test_mask_all_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertGreater(len(samples), 0)
            for s in samples:
                self.assertEqual(s["mask"], [1] * 225)


class TestWinnerPathSkipDraw(unittest.TestCase):
    """winner=3 整局跳过"""

    def test_skip_draw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 3, "history": _make_history(10)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(samples, [])


class TestWinnerPathSkipNoWinner(unittest.TestCase):
    """winner=0 整局跳过"""

    def test_skip_no_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 0, "history": _make_history(10)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(samples, [])


class TestWinnerPathFirstMoveExcluded(unittest.TestCase):
    """显式验证 history[0]（黑胜）不在样本中"""

    def test_first_move_is_excluded_black_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、history=5 步，_make_history 中 step 0 的 input 全为 0.0
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # history[0] 的 input 全为 0.0，不应出现在样本中
            for s in samples:
                self.assertNotEqual(s["input"], [0.0] * 226)

    def test_first_move_is_excluded_white_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 白胜、history=6 步，白方第 1 步是 history[1]，其 input 全为 1.0
            record = {"winner": 2, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # history[1] 的 input 全为 1.0，不应出现在样本中
            for s in samples:
                self.assertNotEqual(s["input"], [1.0] * 226)


class TestWinnerPathShortGame(unittest.TestCase):
    """获胜方仅 1 步时剔首后为空"""

    def test_black_wins_only_one_move_returns_empty(self):
        """history=1 步，黑胜；获胜方仅 1 步，剔首后 0 个 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(1)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(samples, [])

    def test_white_wins_only_one_move_returns_empty(self):
        """history=2 步，白胜；白方仅 1 步，剔首后 0 个 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 2, "history": _make_history(2)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(samples, [])


class TestWinnerPathNoAugmentation(unittest.TestCase):
    """augment=True 时仍不增强"""

    def test_no_augmentation_even_if_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、history=5 步 → 获胜方 3 步，剔首 → 2 个 sample
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path", augment=True)
            samples = creator.create([path])

            # 即使启用 augment，也应只有 2 个（不是 2*8=16）
            self.assertEqual(len(samples), 2)


class TestWinnerPathInvalidPosition(unittest.TestCase):
    """position 越界时不崩溃，对应 sample output 全 0"""

    def test_invalid_position_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、3 步，最后一步（黑方第 2 步）position 越界
            history = [
                {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},      # 黑，剔除
                {"input": [1.0] * 226, "output": [0.0] * 225, "position": [0, 1]},      # 白
                {"input": [2.0] * 226, "output": [0.0] * 225, "position": [20, 20]},    # 黑，保留但越界
            ]
            record = {"winner": 1, "history": history}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["output"], [0.0] * 225)
            self.assertEqual(samples[0]["mask"], [1] * 225)

    def test_negative_position_out_of_bounds(self):
        """负坐标越界时对应 sample output 全 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = [
                {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},     # 白第1步剔除位
                {"input": [1.0] * 226, "output": [0.0] * 225, "position": [0, 1]},     # 白，剔除
                {"input": [2.0] * 226, "output": [0.0] * 225, "position": [0, 2]},     # 黑
                {"input": [3.0] * 226, "output": [0.0] * 225, "position": [-1, 5]},    # 白，保留但越界
            ]
            record = {"winner": 2, "history": history}
            path = _write_record(tmpdir, record)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path])

            # 白方步 = [1, 3]，剔首 → [3]
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["output"], [0.0] * 225)


class TestWinnerPathMultipleFiles(unittest.TestCase):
    """多文件合并数量正确"""

    def test_multiple_files_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 文件 1：黑胜、5 步 → 2 个 sample
            path1 = _write_record(
                tmpdir,
                {"winner": 1, "history": _make_history(5)},
                filename="record1.json",
            )
            # 文件 2：白胜、6 步 → 2 个 sample
            path2 = _write_record(
                tmpdir,
                {"winner": 2, "history": _make_history(6, start_input_value=100.0)},
                filename="record2.json",
            )
            # 文件 3：平局 → 跳过
            path3 = _write_record(
                tmpdir,
                {"winner": 3, "history": _make_history(8)},
                filename="record3.json",
            )

            creator = SampleCreator(mode="winner_path")
            samples = creator.create([path1, path2, path3])

            self.assertEqual(len(samples), 4)


class TestWinnerPathFromRoundDir(unittest.TestCase):
    """从轮次目录自动发现 record.json 并生成 sample"""

    def test_discovers_all_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 3 局黑胜、每局 5 步 → 每局 2 个 sample → 共 6 个
            for i in [1, 2, 3]:
                game_dir = os.path.join(tmpdir, str(i))
                os.makedirs(game_dir, exist_ok=True)
                record = {
                    "winner": 1,
                    "history": _make_history(5, start_input_value=float(i * 10)),
                }
                with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                    json.dump(record, f)

            creator = SampleCreator(mode="winner_path")
            samples = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 6)


if __name__ == '__main__':
    unittest.main()
