"""SampleCreator 单元测试

覆盖 winning_move 模式的各种场景：
- 正常对局生成样本
- 平局跳过
- 未结束跳过
- 空 history 跳过
- 多文件合并
- position 越界处理
- 从轮次目录自动发现
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.sample_creator import SampleCreator


class TestWinningMoveBasic(unittest.TestCase):
    """正常对局生成样本"""

    def test_last_position_output_is_one(self):
        """最后一位置 output=1，其余=0，mask 全 1"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                    {"input": [1.0] * 226, "output": [0.0] * 225, "position": [1, 1]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample["input"], [1.0] * 226)
            self.assertEqual(len(sample["output"]), 225)
            # position [1, 1] -> idx = 1*15 + 1 = 16
            expected_output = [0.0] * 225
            expected_output[16] = 1.0
            self.assertEqual(sample["output"], expected_output)
            self.assertEqual(sample["mask"], [1] * 225)


class TestWinningMoveSkipDraw(unittest.TestCase):
    """winner=3 时跳过"""

    def test_draw_game_skipped(self):
        """平局对局不产生样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 3,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 0)


class TestWinningMoveSkipNoWinner(unittest.TestCase):
    """winner=0 时跳过"""

    def test_no_winner_game_skipped(self):
        """未结束对局不产生样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 0,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 0)


class TestWinningMoveEmptyHistory(unittest.TestCase):
    """history 为空时跳过"""

    def test_empty_history_skipped(self):
        """空 history 不产生样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 0)


class TestWinningMoveMultipleFiles(unittest.TestCase):
    """多文件合并，样本数正确"""

    def test_two_records_generate_two_samples(self):
        """两个有效 record 文件生成两个样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 文件 1：黑胜
            record1 = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            path1 = os.path.join(tmpdir, "1", "record.json")
            os.makedirs(os.path.dirname(path1), exist_ok=True)
            with open(path1, 'w', encoding='utf-8') as f:
                json.dump(record1, f)

            # 文件 2：白胜
            record2 = {
                "winner": 2,
                "history": [
                    {"input": [1.0] * 226, "output": [0.0] * 225, "position": [2, 3]},
                ],
                "end_board": [0.0] * 226
            }
            path2 = os.path.join(tmpdir, "2", "record.json")
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            with open(path2, 'w', encoding='utf-8') as f:
                json.dump(record2, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path1, path2])

            self.assertEqual(len(samples), 2)
            # 第一个样本 input 应为 record1 最后一步
            self.assertEqual(samples[0]["input"], [0.0] * 226)
            # 第二个样本 input 应为 record2 最后一步
            self.assertEqual(samples[1]["input"], [1.0] * 226)

    def test_mixed_valid_and_invalid(self):
        """混合有效和无效记录，只保留有效"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 有效：黑胜
            record1 = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            path1 = os.path.join(tmpdir, "record1.json")
            with open(path1, 'w', encoding='utf-8') as f:
                json.dump(record1, f)

            # 无效：平局
            record2 = {
                "winner": 3,
                "history": [
                    {"input": [1.0] * 226, "output": [0.0] * 225, "position": [1, 1]},
                ],
                "end_board": [0.0] * 226
            }
            path2 = os.path.join(tmpdir, "record2.json")
            with open(path2, 'w', encoding='utf-8') as f:
                json.dump(record2, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path1, path2])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["input"], [0.0] * 226)


class TestWinningMoveInvalidPosition(unittest.TestCase):
    """position 越界时该位置不设置（output 全 0）"""

    def test_position_out_of_bounds_no_crash(self):
        """position 越界时不崩溃，output 全 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [20, 20]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["output"], [0.0] * 225)
            self.assertEqual(samples[0]["mask"], [1] * 225)

    def test_negative_position_out_of_bounds(self):
        """负坐标越界时 output 全 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 2,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [-1, 5]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["output"], [0.0] * 225)


class TestCreateFromRoundDir(unittest.TestCase):
    """从轮次目录自动发现 record.json"""

    def test_discovers_all_records(self):
        """自动发现子目录下的所有 record.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 模拟轮次目录结构：round/1/record.json, round/2/record.json
            for i in [1, 2, 3]:
                game_dir = os.path.join(tmpdir, str(i))
                os.makedirs(game_dir, exist_ok=True)
                record = {
                    "winner": 1,
                    "history": [
                        {"input": [float(i)] * 226, "output": [0.0] * 225, "position": [i - 1, i - 1]},
                    ],
                    "end_board": [0.0] * 226
                }
                with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                    json.dump(record, f)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 3)

    def test_skips_dirs_without_record(self):
        """跳过没有 record.json 的子目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 有效目录
            game_dir1 = os.path.join(tmpdir, "1")
            os.makedirs(game_dir1, exist_ok=True)
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            with open(os.path.join(game_dir1, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record, f)

            # 无效目录（无 record.json）
            game_dir2 = os.path.join(tmpdir, "2")
            os.makedirs(game_dir2, exist_ok=True)

            creator = SampleCreator(mode="winning_move")
            samples = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 1)

    def test_empty_round_dir_returns_empty(self):
        """空轮次目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = SampleCreator(mode="winning_move")
            samples = creator.create_from_round_dir(tmpdir)
            self.assertEqual(samples, [])


class TestUnsupportedMode(unittest.TestCase):
    """不支持的模式"""

    def test_unsupported_mode_raises(self):
        """不支持的模式抛出 ValueError"""
        with self.assertRaises(ValueError):
            SampleCreator(mode="unsupported_mode")


if __name__ == '__main__':
    unittest.main()
