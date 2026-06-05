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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
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

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 1)

    def test_empty_round_dir_returns_empty(self):
        """空轮次目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create_from_round_dir(tmpdir)
            self.assertEqual(samples, [])


class TestUnsupportedMode(unittest.TestCase):
    """不支持的模式"""

    def test_unsupported_mode_raises(self):
        """不支持的模式抛出 ValueError"""
        with self.assertRaises(ValueError):
            SampleCreator(mode="unsupported_mode")


class TestStepIdWinningMove(unittest.TestCase):
    """winning_move 模式 step_id 透传测试"""

    def test_step_id_from_history_to_sample(self):
        """record.json 中含 step_id 时，样本透传该字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"step_id": "aaaa-bbbb-cccc", "input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                    {"step_id": "dddd-eeee-ffff", "input": [1.0] * 226, "output": [0.0] * 225, "position": [1, 1]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            # winning_move 取最后一步，step_id 应为最后一步的
            self.assertEqual(samples[0]["step_id"], "dddd-eeee-ffff")

    def test_no_step_id_in_history_no_step_id_in_sample(self):
        """旧格式 record.json 无 step_id 时，样本也不含该字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)
            self.assertNotIn("step_id", samples[0])


class TestStepIdWithAugmentation(unittest.TestCase):
    """D4 增强时 step_id 保留测试"""

    def test_augment_preserves_step_id(self):
        """D4 增强后的 8 个变体都保留相同的 step_id"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"step_id": "test-uuid-1234", "input": [0.0] * 226, "output": [0.0] * 225, "position": [7, 7]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=True)
            samples = creator.create([path])

            self.assertEqual(len(samples), 8)
            # 所有变体共享同一 step_id
            for s in samples:
                self.assertEqual(s["step_id"], "test-uuid-1234")

    def test_augment_without_step_id_no_step_id(self):
        """旧格式无 step_id 时，增强变体也不含该字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [7, 7]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=True)
            samples = creator.create([path])

            self.assertEqual(len(samples), 8)
            for s in samples:
                self.assertNotIn("step_id", s)


class TestDataAugmentation(unittest.TestCase):
    """D4 群数据增强测试"""

    def test_augment_produces_eight_samples(self):
        """1 个原始 sample 产生 8 个变体"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [7, 7]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=True)
            samples = creator.create([path])

            self.assertEqual(len(samples), 8)

    def test_augment_identity_is_first(self):
        """第 0 个变体（恒等变换）与原始完全一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [float(i) for i in range(226)], "output": [0.0] * 225, "position": [3, 4]},
                ],
                "end_board": [0.0] * 226
            }
            # 手动设置 output，让 position [3,4] 对应位置为 1
            record["history"][0]["output"][3 * 15 + 4] = 1.0
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator_no_aug = SampleCreator(mode="winning_move", augment=False)
            creator_aug = SampleCreator(mode="winning_move", augment=True)

            original_samples = creator_no_aug.create([path])
            augmented_samples = creator_aug.create([path])

            self.assertEqual(len(original_samples), 1)
            self.assertEqual(len(augmented_samples), 8)
            # 恒等变换（tid=0）应与原始完全一致
            self.assertEqual(augmented_samples[0]["input"], original_samples[0]["input"])
            self.assertEqual(augmented_samples[0]["output"], original_samples[0]["output"])
            self.assertEqual(augmented_samples[0]["mask"], original_samples[0]["mask"])

    def test_augment_rotation_90_position(self):
        """验证 R90 后 output 的 1 位置正确移动"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # position [1, 2] -> idx = 1*15 + 2 = 17
            # R90: (r, c) -> (c, 14-r) = (2, 13) -> idx2 = 2*15 + 13 = 43
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [1, 2]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=True)
            samples = creator.create([path])

            self.assertEqual(len(samples), 8)
            # tid=1 是 R90
            r90_output = samples[1]["output"]
            self.assertEqual(r90_output[43], 1.0)
            # 确认原位置不是 1
            self.assertEqual(r90_output[17], 0.0)

    def test_augment_horizontal_flip_position(self):
        """验证左右镜像后 output 的 1 位置正确移动"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # position [3, 5] -> idx = 3*15 + 5 = 50
            # 左右镜像: (r, c) -> (r, 14-c) = (3, 9) -> idx2 = 3*15 + 9 = 54
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [3, 5]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=True)
            samples = creator.create([path])

            self.assertEqual(len(samples), 8)
            # tid=4 是左右镜像
            flip_output = samples[4]["output"]
            self.assertEqual(flip_output[54], 1.0)
            self.assertEqual(flip_output[50], 0.0)

    def test_augment_composition_r90_twice(self):
        """对同一向量应用两次 R90 等价于 R180"""
        creator = SampleCreator(mode="winning_move", augment=False)
        # 构造一个 225 维向量，只有 idx=17 处为 1
        vector = [0.0] * 225
        vector[17] = 1.0

        # 第一次 R90 (tid=1)
        r90_once = creator._apply_transform(vector, 1)
        # 第二次 R90 (tid=1)
        r90_twice = creator._apply_transform(r90_once, 1)
        # R180 (tid=2)
        r180 = creator._apply_transform(vector, 2)

        self.assertEqual(r90_twice, r180)

    def test_augment_disabled_returns_one(self):
        """augment=False 时只返回 1 个原始 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [5, 5]},
                ],
                "end_board": [0.0] * 226
            }
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([path])

            self.assertEqual(len(samples), 1)


class TestRecordFilter(unittest.TestCase):
    """SampleCreator record_filter 参数测试"""

    def _make_record_file(self, tmpdir, game_id, winner, temp_side_this_game=None):
        """辅助：创建一个 record.json 文件并返回路径"""
        record = {
            "winner": winner,
            "history": [
                {
                    "input": [0] * 225 + [1],
                    "output": [0.0] * 225,
                    "position": [7, 7],
                }
            ],
            "end_board": [0] * 226,
        }
        if temp_side_this_game is not None:
            record["temp_side_this_game"] = temp_side_this_game

        game_dir = os.path.join(tmpdir, str(game_id))
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, "record.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        return path

    def test_filter_keeps_sampling_side_win(self):
        """SC-01: should_keep_game 过滤生效——仅采样方获胜局保留"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜 + 采样方=black → 保留
            p1 = self._make_record_file(tmpdir, 1, 1, "black")
            # 白胜 + 采样方=black → 丢弃
            p2 = self._make_record_file(tmpdir, 2, 2, "black")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([p1, p2], record_filter=should_keep_game)
            # 仅黑胜局保留
            self.assertEqual(len(samples), 1)

    def test_filter_both_side_not_filtered(self):
        """SC-02: temp_side_this_game="both" → 不过滤"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "both")
            p2 = self._make_record_file(tmpdir, 2, 2, "both")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([p1, p2], record_filter=should_keep_game)
            self.assertEqual(len(samples), 2)

    def test_filter_none_not_filtered(self):
        """SC-03: temp_side_this_game=None → 不过滤"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, None)

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([p1], record_filter=should_keep_game)
            self.assertEqual(len(samples), 1)

    def test_no_filter_default(self):
        """SC-04: 不传 record_filter → 不过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "black")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([p1])
            self.assertEqual(len(samples), 1)

    def test_custom_filter_all_discard(self):
        """SC-05: 自定义过滤函数全丢弃"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "black")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create([p1], record_filter=lambda r: False)
            self.assertEqual(len(samples), 0)

    def test_create_from_round_dir_with_filter(self):
        """SR-01/SR-02: create_from_round_dir 透传 record_filter"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜 + 采样方=black → 保留
            self._make_record_file(tmpdir, 1, 1, "black")
            # 白胜 + 采样方=black → 丢弃
            self._make_record_file(tmpdir, 2, 2, "black")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create_from_round_dir(tmpdir, record_filter=should_keep_game)
            self.assertEqual(len(samples), 1)

    def test_create_from_round_dir_no_filter(self):
        """SR-04: create_from_round_dir 不传 record_filter → 不过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_record_file(tmpdir, 1, 1, "black")
            self._make_record_file(tmpdir, 2, 2, "black")

            creator = SampleCreator(mode="winning_move", augment=False)
            samples = creator.create_from_round_dir(tmpdir)
            self.assertEqual(len(samples), 2)


if __name__ == '__main__':
    unittest.main()
