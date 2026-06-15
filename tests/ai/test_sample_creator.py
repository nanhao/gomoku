"""SampleCreator 单元测试

覆盖 winner_path 模式的各种场景：
- 变异步过滤 (move_filter)
- 数据增强 (augment)
- 对局级过滤 (record_filter)
- 从轮次目录自动发现
- create() 返回值变更

注意：winning_move 模式已移除（意图 9）。
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.sample_creator import SampleCreator


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_record_file(tmpdir, game_id, winner, temp_side_this_game=None,
                      history_steps=1, outputs=None, positions=None):
    """辅助：创建一个 record.json 文件并返回路径

    Args:
        tmpdir: 临时目录
        game_id: 对局编号
        winner: 胜者
        temp_side_this_game: 采样方
        history_steps: history 步数
        outputs: 可选的 output 列表，长度需等于 history_steps
        positions: 可选的 position 列表，长度需等于 history_steps
    """
    history = []
    for i in range(history_steps):
        step = {
            "input": [float(i)] * 226,
            "output": ([0.0] * 225) if outputs is None else list(outputs[i]),
            "position": ([0, 0]) if positions is None else list(positions[i]),
        }
        history.append(step)

    record = {
        "winner": winner,
        "history": history,
        "end_board": [0.0] * 226,
    }
    if temp_side_this_game is not None:
        record["temp_side_this_game"] = temp_side_this_game

    game_dir = os.path.join(tmpdir, str(game_id))
    os.makedirs(game_dir, exist_ok=True)
    path = os.path.join(game_dir, "record.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f)
    return path


def _make_flat_record_file(tmpdir, filename, winner, history):
    """辅助：在 tmpdir 根目录创建 record.json 文件（非子目录结构）"""
    record = {"winner": winner, "history": history, "end_board": [0.0] * 226}
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f)
    return path


# ---------------------------------------------------------------------------
# 构造函数测试
# ---------------------------------------------------------------------------

class TestConstructor(unittest.TestCase):
    """SampleCreator 构造函数变更测试"""

    def test_default_augment_true(self):
        """SC-01: 不传任何参数，默认 augment=True"""
        creator = SampleCreator()
        self.assertTrue(creator.augment)

    def test_augment_false(self):
        """SC-02: 显式传 augment=False"""
        creator = SampleCreator(augment=False)
        self.assertFalse(creator.augment)

    def test_mode_param_rejected(self):
        """SC-03: 不再接受 mode 参数"""
        with self.assertRaises(TypeError):
            SampleCreator(mode="winner_path")


class TestCreateFromRoundDir(unittest.TestCase):
    """从轮次目录自动发现 record.json"""

    def test_discovers_all_records(self):
        """自动发现子目录下的所有 record.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in [1, 2, 3]:
                game_dir = os.path.join(tmpdir, str(i))
                os.makedirs(game_dir, exist_ok=True)
                record = {
                    "winner": 1,
                    "history": [
                        {"input": [float(i)] * 226, "output": [0.0] * 225,
                         "position": [i - 1, i - 1]},
                    ],
                    "end_board": [0.0] * 226
                }
                with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                    json.dump(record, f)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 3)

    def test_skips_dirs_without_record(self):
        """跳过没有 record.json 的子目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
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

            game_dir2 = os.path.join(tmpdir, "2")
            os.makedirs(game_dir2, exist_ok=True)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 1)

    def test_empty_round_dir_returns_empty(self):
        """空轮次目录返回空列表和零值 stats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(tmpdir)
            self.assertEqual(samples, [])
            self.assertEqual(stats["total_moves"], 0)


# ---------------------------------------------------------------------------
# D4 增强测试
# ---------------------------------------------------------------------------

class TestDataAugmentation(unittest.TestCase):
    """D4 群数据增强测试"""

    def _make_record(self, tmpdir, winner=1, position=None):
        """创建测试用 record 文件"""
        pos = position or [7, 7]
        history = [
            {"input": [0.0] * 226, "output": [0.0] * 225, "position": pos},
        ]
        record = {"winner": winner, "history": history, "end_board": [0.0] * 226}
        path = os.path.join(tmpdir, "record.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        return path

    def test_augment_produces_eight_samples(self):
        """1 个原始 sample 产生 8 个变体"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_record(tmpdir)
            creator = SampleCreator(augment=True)
            samples, stats = creator.create([path])
            self.assertEqual(len(samples), 8)

    def test_augment_identity_is_first(self):
        """第 0 个变体（恒等变换）与原始完全一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                "winner": 1,
                "history": [
                    {"input": [float(i) for i in range(226)],
                     "output": [0.0] * 225, "position": [3, 4]},
                ],
                "end_board": [0.0] * 226
            }
            record["history"][0]["output"][3 * 15 + 4] = 1.0
            path = os.path.join(tmpdir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator_no_aug = SampleCreator(augment=False)
            creator_aug = SampleCreator(augment=True)

            original_samples, _ = creator_no_aug.create([path])
            augmented_samples, _ = creator_aug.create([path])

            self.assertEqual(len(original_samples), 1)
            self.assertEqual(len(augmented_samples), 8)
            self.assertEqual(augmented_samples[0]["input"], original_samples[0]["input"])
            self.assertEqual(augmented_samples[0]["output"], original_samples[0]["output"])
            self.assertEqual(augmented_samples[0]["mask"], original_samples[0]["mask"])

    def test_augment_rotation_90_position(self):
        """验证 R90 后 output 的 1 位置正确移动"""
        with tempfile.TemporaryDirectory() as tmpdir:
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

            creator = SampleCreator(augment=True)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 8)
            r90_output = samples[1]["output"]
            self.assertEqual(r90_output[43], 1.0)
            self.assertEqual(r90_output[17], 0.0)

    def test_augment_horizontal_flip_position(self):
        """验证左右镜像后 output 的 1 位置正确移动"""
        with tempfile.TemporaryDirectory() as tmpdir:
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

            creator = SampleCreator(augment=True)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 8)
            flip_output = samples[4]["output"]
            self.assertEqual(flip_output[54], 1.0)
            self.assertEqual(flip_output[50], 0.0)

    def test_augment_composition_r90_twice(self):
        """对同一向量应用两次 R90 等价于 R180"""
        creator = SampleCreator(augment=False)
        vector = [0.0] * 225
        vector[17] = 1.0

        r90_once = creator._apply_transform(vector, 1)
        r90_twice = creator._apply_transform(r90_once, 1)
        r180 = creator._apply_transform(vector, 2)

        self.assertEqual(r90_twice, r180)

    def test_augment_disabled_returns_one(self):
        """augment=False 时只返回 1 个原始 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_record(tmpdir)
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])
            self.assertEqual(len(samples), 1)


# ---------------------------------------------------------------------------
# 对局级过滤测试
# ---------------------------------------------------------------------------

class TestRecordFilter(unittest.TestCase):
    """SampleCreator record_filter 参数测试"""

    def _make_record_file(self, tmpdir, game_id, winner, temp_side_this_game=None,
                          history_steps=1):
        """辅助：创建一个 record.json 文件并返回路径"""
        history = []
        for i in range(history_steps):
            history.append({
                "input": [float(i)] * 226,
                "output": [0.0] * 225,
                "position": [i // 15, i % 15],
            })
        record = {
            "winner": winner,
            "history": history,
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
            p1 = self._make_record_file(tmpdir, 1, 1, "black")
            p2 = self._make_record_file(tmpdir, 2, 2, "black")

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1, p2], record_filter=should_keep_game)
            self.assertEqual(len(samples), 1)

    def test_filter_both_side_not_filtered(self):
        """SC-02: temp_side_this_game="both" → 不过滤"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "both")
            # 白胜需要 ≥2 步 history 才能有白方步
            p2 = self._make_record_file(tmpdir, 2, 2, "both", history_steps=2)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1, p2], record_filter=should_keep_game)
            # 黑胜=1步，白胜=1步（2步history中白方在索引1）
            self.assertEqual(len(samples), 2)

    def test_filter_none_not_filtered(self):
        """SC-03: temp_side_this_game=None → 不过滤"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, None)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1], record_filter=should_keep_game)
            self.assertEqual(len(samples), 1)

    def test_no_filter_default(self):
        """SC-04: 不传 record_filter → 不过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "black")

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1])
            self.assertEqual(len(samples), 1)

    def test_custom_filter_all_discard(self):
        """SC-05: 自定义过滤函数全丢弃"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 1, "black")

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1], record_filter=lambda r: False)
            self.assertEqual(len(samples), 0)

    def test_create_from_round_dir_with_filter(self):
        """SR-01: create_from_round_dir 透传 record_filter"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_record_file(tmpdir, 1, 1, "black")
            self._make_record_file(tmpdir, 2, 2, "black")

            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(
                tmpdir, record_filter=should_keep_game)
            self.assertEqual(len(samples), 1)

    def test_create_from_round_dir_no_filter(self):
        """SR-04: create_from_round_dir 不传 record_filter → 不过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_record_file(tmpdir, 1, 1, "black")
            # 白胜需要 ≥2 步 history 才能有白方步
            self._make_record_file(tmpdir, 2, 2, "black", history_steps=2)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(tmpdir)
            # 黑胜=1步，白胜=1步（2步history中白方在索引1）
            self.assertEqual(len(samples), 2)

    def test_filter_draw_game_skipped(self):
        """SC-06: 平局（winner=3）+ record_filter → 0 样本"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_record_file(tmpdir, 1, 3, "black")

            creator = SampleCreator(augment=False)
            samples, stats = creator.create([p1], record_filter=should_keep_game)
            self.assertEqual(len(samples), 0)

    def test_create_from_round_dir_both_side_not_filtered(self):
        """SR-03: create_from_round_dir + temp_side_this_game="both" + 白胜 → 不过滤"""
        from ai.game_filter import should_keep_game
        with tempfile.TemporaryDirectory() as tmpdir:
            # 白胜需要 ≥2 步 history 才能有白方步
            self._make_record_file(tmpdir, 1, 2, "both", history_steps=2)

            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(
                tmpdir, record_filter=should_keep_game)
            # 白胜=1步（2步history中白方在索引1）
            self.assertEqual(len(samples), 1)


if __name__ == '__main__':
    unittest.main()
