"""SampleCreator 变异步过滤单元测试

覆盖意图 9 新增的步级过滤（变异步过滤）功能：
- 单方采样场景的变异步判定
- 双方采样场景的变异步过滤
- move_filter=None 时不过滤
- create() 返回值变更适配
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

def _is_variant_move(step: dict) -> bool:
    """测试用变异步判定函数（与 create_samples.is_variant_move 同语义）"""
    output = step["output"]
    pos = step["position"]
    idx = pos[0] * 15 + pos[1]
    if not (0 <= pos[0] < 15 and 0 <= pos[1] < 15):
        return False
    if len(output) != 225:
        return False
    max_val = max(output)
    return output[idx] != max_val


def _make_record_file(tmpdir, game_id, winner, temp_side_this_game=None):
    """创建基础 record 文件"""
    record = {
        "winner": winner,
        "history": [
            {"input": [0] * 225 + [1], "output": [0.0] * 225, "position": [7, 7]}
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


# ---------------------------------------------------------------------------
# 变异步过滤 - 单方采样场景
# ---------------------------------------------------------------------------

class TestVariantMoveFilter(unittest.TestCase):
    """变异步过滤：单方采样场景（MV-01 ~ MV-07）"""

    def _make_record_with_outputs(self, tmpdir, game_id, winner,
                                   outputs, positions, temp_side="black"):
        """创建含指定 output 和 position 的 record 文件"""
        history = []
        for i in range(len(outputs)):
            history.append({
                "input": [float(i)] * 226,
                "output": list(outputs[i]),
                "position": list(positions[i]),
            })
        record = {
            "winner": winner,
            "history": history,
            "end_board": [0.0] * 226,
            "temp_side_this_game": temp_side,
        }
        game_dir = os.path.join(tmpdir, str(game_id))
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, "record.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        return path

    def test_single_variant_move_kept(self):
        """MV-01: 单步变异步 → 保留"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = [0.9] + [0.0] * 224
            output[1] = 0.1  # 落子位 [0, 1] → idx=1，概率 0.1
            path = self._make_record_with_outputs(
                tmpdir, 1, 1, [output], [[0, 1]])

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 8)  # augment=True，1 步 × 8
            self.assertEqual(stats["variant_moves"], 1)
            self.assertEqual(stats["skipped_moves"], 0)
            self.assertEqual(stats["total_moves"], 1)
            self.assertTrue(stats["move_filter_active"])

    def test_single_non_variant_move_skipped(self):
        """MV-02: 单步非变异步 → 跳过"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = [0.0] * 225
            output[0] = 0.9  # 落子位 [0, 0] → idx=0，概率=0.9（唯一最大值）
            remaining = 0.1 / 224
            for i in range(1, 225):
                output[i] = remaining
            path = self._make_record_with_outputs(
                tmpdir, 1, 1, [output], [[0, 0]])

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 0)
            self.assertEqual(stats["variant_moves"], 0)
            self.assertEqual(stats["skipped_moves"], 1)
            self.assertEqual(stats["total_moves"], 1)

    def test_tied_argmax_skipped(self):
        """MV-03: 落子位与另一个位置并列 argmax → 非变异步，跳过"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = [0.0] * 225
            output[0] = 0.5  # 落子位
            output[15] = 0.5  # 另一个位置也 0.5
            path = self._make_record_with_outputs(
                tmpdir, 1, 1, [output], [[0, 0]])

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 0)
            self.assertEqual(stats["skipped_moves"], 1)

    def test_all_positions_equal_skipped(self):
        """MV-04: 所有位置概率相等 → 非变异步，跳过"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = [1.0 / 225] * 225
            path = self._make_record_with_outputs(
                tmpdir, 1, 1, [output], [[7, 7]])

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 0)

    def test_mixed_steps(self):
        """MV-05: 多步混合：步1变异、步2非变异、步3变异"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 5 步黑胜 history（黑方步=索引 0,2,4）
            outputs = []
            positions = []
            for i in range(5):
                o = [0.0] * 225
                pos = [i // 15, i % 15]
                idx = pos[0] * 15 + pos[1]
                if i in (0,):
                    # 步 0（黑）：落子位=0.9（唯一最大值），非变异
                    o[idx] = 0.9
                    o[(idx + 1) % 225] = 0.1
                elif i in (2, 4):
                    # 步 2, 4（黑）：落子位=0.1，其他=0.9 → 变异
                    o[(idx + 1) % 225] = 0.9
                    o[idx] = 0.1
                else:
                    # 白方步：随便
                    o[idx] = 0.5
                    o[(idx + 1) % 225] = 0.5
                outputs.append(o)
                positions.append(pos)

            path = self._make_record_with_outputs(
                tmpdir, 1, 1, outputs, positions)

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            # 黑方步=3（索引 0,2,4），步0非变异跳过，步2和4变异保留 → 2 步 × 8 = 16
            self.assertEqual(len(samples), 16)
            self.assertEqual(stats["variant_moves"], 2)
            self.assertEqual(stats["skipped_moves"], 1)
            self.assertEqual(stats["total_moves"], 3)

    def test_all_variant(self):
        """MV-06: 全部变异步（3步全变异）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 5 步黑胜，黑方 3 步全部变异
            outputs = []
            positions = []
            for i in range(5):
                o = [0.0] * 225
                pos = [i // 15, i % 15]
                idx = pos[0] * 15 + pos[1]
                o[(idx + 1) % 225] = 0.9
                o[idx] = 0.1
                outputs.append(o)
                positions.append(pos)

            path = self._make_record_with_outputs(
                tmpdir, 1, 1, outputs, positions)

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 24)  # 3 × 8
            self.assertEqual(stats["variant_moves"], 3)
            self.assertEqual(stats["skipped_moves"], 0)

    def test_all_non_variant(self):
        """MV-07: 全部非变异步 → 0 样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = []
            positions = []
            for i in range(5):
                o = [0.0] * 225
                pos = [i // 15, i % 15]
                idx = pos[0] * 15 + pos[1]
                o[idx] = 0.9
                o[(idx + 1) % 225] = 0.1
                outputs.append(o)
                positions.append(pos)

            path = self._make_record_with_outputs(
                tmpdir, 1, 1, outputs, positions)

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 0)
            self.assertEqual(stats["variant_moves"], 0)
            self.assertEqual(stats["skipped_moves"], 3)


# ---------------------------------------------------------------------------
# 变异步过滤 - 双方采样场景
# ---------------------------------------------------------------------------

class TestVariantMoveBothSide(unittest.TestCase):
    """变异步过滤：双方采样场景（MV-08 ~ MV-09）"""

    def test_both_side_mixed(self):
        """MV-08: 双方采样 + 混合步"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 3 步黑胜，黑方 2 步：1 变异 + 1 非变异
            outputs = []
            positions = []
            for i in range(3):
                o = [0.0] * 225
                pos = [i, i]
                idx = pos[0] * 15 + pos[1]
                if i == 0:
                    # 步 0（黑）：变异
                    o[(idx + 1) % 225] = 0.9
                    o[idx] = 0.1
                elif i == 2:
                    # 步 2（黑）：非变异
                    o[idx] = 0.9
                    o[(idx + 1) % 225] = 0.1
                else:
                    # 白方步
                    o[idx] = 0.5
                outputs.append(o)
                positions.append(pos)

            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": list(outputs[0]),
                     "position": positions[0]},
                    {"input": [1.0] * 226, "output": list(outputs[1]),
                     "position": positions[1]},
                    {"input": [2.0] * 226, "output": list(outputs[2]),
                     "position": positions[2]},
                ],
                "end_board": [0.0] * 226,
                "temp_side_this_game": "both",
            }
            game_dir = os.path.join(tmpdir, "1")
            os.makedirs(game_dir, exist_ok=True)
            path = os.path.join(game_dir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            # 黑方 2 步：步0变异保留，步2非变异跳过 → 1 步 × 8 = 8
            self.assertEqual(len(samples), 8)
            self.assertEqual(stats["variant_moves"], 1)
            self.assertEqual(stats["skipped_moves"], 1)
            self.assertTrue(stats["move_filter_active"])

    def test_both_side_all_non_variant(self):
        """MV-09: 双方采样 + 全部非变异步 → 0 样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = []
            positions = []
            for i in range(3):
                o = [0.0] * 225
                pos = [i, i]
                idx = pos[0] * 15 + pos[1]
                o[idx] = 0.9
                o[(idx + 1) % 225] = 0.1
                outputs.append(o)
                positions.append(pos)

            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": list(outputs[0]),
                     "position": positions[0]},
                    {"input": [1.0] * 226, "output": list(outputs[1]),
                     "position": positions[1]},
                    {"input": [2.0] * 226, "output": list(outputs[2]),
                     "position": positions[2]},
                ],
                "end_board": [0.0] * 226,
                "temp_side_this_game": "both",
            }
            game_dir = os.path.join(tmpdir, "1")
            os.makedirs(game_dir, exist_ok=True)
            path = os.path.join(game_dir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)

            self.assertEqual(len(samples), 0)
            self.assertTrue(stats["move_filter_active"])


# ---------------------------------------------------------------------------
# move_filter=None 场景
# ---------------------------------------------------------------------------

class TestMoveFilterNone(unittest.TestCase):
    """变异步过滤：move_filter 为 None（MV-10）"""

    def test_no_move_filter_keeps_all(self):
        """MV-10: 不传 move_filter → 全部获胜方步生成样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = []
            positions = []
            for i in range(5):
                o = [0.0] * 225
                pos = [i // 15, i % 15]
                idx = pos[0] * 15 + pos[1]
                o[idx] = 0.9
                o[(idx + 1) % 225] = 0.1
                outputs.append(o)
                positions.append(pos)

            history = []
            for i in range(5):
                history.append({
                    "input": [float(i)] * 226,
                    "output": list(outputs[i]),
                    "position": positions[i],
                })
            record = {
                "winner": 1, "history": history, "end_board": [0.0] * 226,
                "temp_side_this_game": "black",
            }
            game_dir = os.path.join(tmpdir, "1")
            os.makedirs(game_dir, exist_ok=True)
            path = os.path.join(game_dir, "record.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record, f)

            creator = SampleCreator()
            samples, stats = creator.create([path])  # 不传 move_filter

            # 黑方 3 步全部保留（不过滤）× 8 = 24
            self.assertEqual(len(samples), 24)
            self.assertEqual(stats["total_moves"], 3)
            self.assertEqual(stats["variant_moves"], 3)
            self.assertFalse(stats["move_filter_active"])


# ---------------------------------------------------------------------------
# create() 返回值测试
# ---------------------------------------------------------------------------

class TestCreateReturnValue(unittest.TestCase):
    """create() 返回值变更测试（RET-01 ~ RET-05）"""

    def test_returns_tuple(self):
        """RET-01: 返回二元组"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_record_file(tmpdir, 1, 1, "black")
            creator = SampleCreator()
            result = creator.create([path], move_filter=_is_variant_move)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            samples, stats = result
            self.assertIsInstance(samples, list)
            self.assertIsInstance(stats, dict)

    def test_stats_fields_complete(self):
        """RET-02: stats 字段完整"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_record_file(tmpdir, 1, 1, "black")
            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)
            self.assertIn("total_moves", stats)
            self.assertIn("variant_moves", stats)
            self.assertIn("skipped_moves", stats)
            self.assertIn("move_filter_active", stats)

    def test_move_filter_active_true(self):
        """RET-03: 传 move_filter → move_filter_active=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_record_file(tmpdir, 1, 1, "black")
            creator = SampleCreator()
            samples, stats = creator.create([path], move_filter=_is_variant_move)
            self.assertTrue(stats["move_filter_active"])

    def test_move_filter_active_false_no_filter(self):
        """RET-05: 不传 move_filter → move_filter_active=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_record_file(tmpdir, 1, 1, "black")
            creator = SampleCreator()
            samples, stats = creator.create([path])
            self.assertFalse(stats["move_filter_active"])


if __name__ == '__main__':
    unittest.main()
