"""SampleCreator winner_path 模式单元测试

覆盖 winner_path 模式（需求 key_points.md 第 643-644 行）。

测试规则（参见 design/sample_creator_design.md "第三十八次追加"）：
- 仅选取获胜方的落子步（黑胜 → history 偶数索引；白胜 → 奇数索引）
- mask：全 1
- 平局/未结束/空 history：整局跳过
- augment 默认启用 D4 增强（意图 9 恢复）

注意：第三十九次追加已将 output 构造由硬标签替换为软标签，相关 output 精确值
校验移至 tests/ai/test_sample_creator_winner_path_soft_label.py；
本文件只保留与 output 算法无关的结构性测试。
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
    """黑胜、history=5 步 → 获胜方 3 步，不再剔首 → 3 个 sample"""

    def test_black_wins_generates_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 5 步：黑(0) 白(1) 黑(2) 白(3) 黑(4)，黑胜
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试 winner_path 步数逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            # 获胜方步 = [0, 2, 4]，不再剔首 → 3 个 sample
            self.assertEqual(len(samples), 3)


class TestWinnerPathWhiteWins(unittest.TestCase):
    """白胜、history=6 步 → 获胜方 3 步，不再剔首 → 3 个 sample"""

    def test_white_wins_generates_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 6 步：黑(0) 白(1) 黑(2) 白(3) 黑(4) 白(5)，白胜
            record = {"winner": 2, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试 winner_path 步数逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            # 获胜方步 = [1, 3, 5]，不再剔首 → 3 个 sample
            self.assertEqual(len(samples), 3)


class TestWinnerPathInputAlignment(unittest.TestCase):
    """每个 sample 的 input 与对应 history[i]["input"] 一致"""

    def test_each_sample_input_is_per_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、6 步：获胜方步 = [0, 2, 4]，不再剔首 → 3 个 sample
            record = {"winner": 1, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试 input 对齐逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 3)
            # _make_history 中 step i 的 input 全为 float(i)
            self.assertEqual(samples[0]["input"], [0.0] * 226)
            self.assertEqual(samples[1]["input"], [2.0] * 226)
            self.assertEqual(samples[2]["input"], [4.0] * 226)


class TestWinnerPathOutputOneHot(unittest.TestCase):
    """[已移除] 原硬标签 one-hot 语义用例已由第三十九次追加替换。

    软标签下的 output 精确值校验统一在
    tests/ai/test_sample_creator_winner_path_soft_label.py 中覆盖。
    """


class TestWinnerPathMask(unittest.TestCase):
    """mask 全 1"""

    def test_mask_all_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator()
            samples, stats = creator.create([path])

            self.assertGreater(len(samples), 0)
            for s in samples:
                self.assertEqual(s["mask"], [1] * 225)


class TestWinnerPathSkipDraw(unittest.TestCase):
    """winner=3 整局跳过"""

    def test_skip_draw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 3, "history": _make_history(10)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator()
            samples, stats = creator.create([path])

            self.assertEqual(samples, [])


class TestWinnerPathSkipNoWinner(unittest.TestCase):
    """winner=0 整局跳过"""

    def test_skip_no_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 0, "history": _make_history(10)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator()
            samples, stats = creator.create([path])

            self.assertEqual(samples, [])


class TestWinnerPathFirstMoveIncluded(unittest.TestCase):
    """显式验证 history[0]（黑胜）或 history[1]（白胜）出现在样本中"""

    def test_first_move_is_included_black_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、history=5 步，_make_history 中 step 0 的 input 全为 0.0
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator()
            samples, stats = creator.create([path])

            # history[0] 的 input 全为 0.0，应出现在样本中（不再剔首）
            included = any(s["input"] == [0.0] * 226 for s in samples)
            self.assertTrue(included)

    def test_first_move_is_included_white_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 白胜、history=6 步，白方第 1 步是 history[1]，其 input 全为 1.0
            record = {"winner": 2, "history": _make_history(6)}
            path = _write_record(tmpdir, record)

            creator = SampleCreator()
            samples, stats = creator.create([path])

            # history[1] 的 input 全为 1.0，应出现在样本中（不再剔首）
            included = any(s["input"] == [1.0] * 226 for s in samples)
            self.assertTrue(included)


class TestWinnerPathShortGame(unittest.TestCase):
    """获胜方仅 1 步时不再剔首，期望 1 个 sample"""

    def test_black_wins_only_one_move_returns_one_sample(self):
        """history=1 步，黑胜；获胜方仅 1 步，不再剔首 → 1 个 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(1)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试短局逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 1)

    def test_white_wins_only_one_move_returns_one_sample(self):
        """history=2 步，白胜；白方仅 1 步，不再剔首 → 1 个 sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 2, "history": _make_history(2)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试短局逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 1)


class TestWinnerPathNoAugmentation(unittest.TestCase):
    """augment=True 时启用增强（意图 9 恢复）"""

    def test_augment_enabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、history=5 步 → 获胜方 3 步，不再剔首
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            # 默认 augment=True，应产出 3*8 = 24 个样本
            creator = SampleCreator()
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 24)
            self.assertTrue(creator.augment)

    def test_augment_disabled_explicitly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(5)}
            path = _write_record(tmpdir, record)

            # 显式禁用增强，应只有 3 个样本
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            self.assertEqual(len(samples), 3)
            self.assertFalse(creator.augment)
            # 每个 sample 的 output 的 1 位置应落在原始保留步的 position 上
            self.assertEqual(samples[0]["output"][0], 1.0)
            self.assertEqual(samples[1]["output"][2], 1.0)
            self.assertEqual(samples[2]["output"][4], 1.0)


class TestWinnerPathInvalidPosition(unittest.TestCase):
    """[已移除] 原"越界 → output 全 0"语义已由第三十九次追加替换为
    "越界 → output = 模型原分布"。相关用例移至
    tests/ai/test_sample_creator_winner_path_soft_label.py 中
    TestSoftLabelInvalidPositionReturnsModelOutput。
    """


class TestWinnerPathMultipleFiles(unittest.TestCase):
    """多文件合并数量正确"""

    def test_multiple_files_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 文件 1：黑胜、5 步 → 3 个 sample
            path1 = _write_record(
                tmpdir,
                {"winner": 1, "history": _make_history(5)},
                filename="record1.json",
            )
            # 文件 2：白胜、6 步 → 3 个 sample
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

            # augment=False 用于隔离测试多文件合并逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path1, path2, path3])

            self.assertEqual(len(samples), 6)


class TestWinnerPathFromRoundDir(unittest.TestCase):
    """从轮次目录自动发现 record.json 并生成 sample"""

    def test_discovers_all_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 3 局黑胜、每局 5 步 → 每局 3 个 sample → 共 9 个
            for i in [1, 2, 3]:
                game_dir = os.path.join(tmpdir, str(i))
                os.makedirs(game_dir, exist_ok=True)
                record = {
                    "winner": 1,
                    "history": _make_history(5, start_input_value=float(i * 10)),
                }
                with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                    json.dump(record, f)

            # augment=False 用于隔离测试轮次目录发现逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create_from_round_dir(tmpdir)

            self.assertEqual(len(samples), 9)


class TestWinnerPathStepId(unittest.TestCase):
    """winner_path 模式 step_id 透传测试"""

    def test_step_id_passthrough(self):
        """record.json 中含 step_id 时，winner_path 样本透传该字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 黑胜、3 步：获胜方步 = [0, 2]
            history = [
                {"step_id": "sid-0", "input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                {"step_id": "sid-1", "input": [1.0] * 226, "output": [0.0] * 225, "position": [0, 1]},
                {"step_id": "sid-2", "input": [2.0] * 226, "output": [0.0] * 225, "position": [0, 2]},
            ]
            record = {"winner": 1, "history": history}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试 step_id 透传逻辑
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            # 获胜方步 = [0, 2]
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["step_id"], "sid-0")
            self.assertEqual(samples[1]["step_id"], "sid-2")

    def test_no_step_id_no_field_in_sample(self):
        """旧格式 record.json 无 step_id 时，winner_path 样本也不含该字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {"winner": 1, "history": _make_history(3)}
            path = _write_record(tmpdir, record)

            # augment=False 用于隔离测试
            creator = SampleCreator(augment=False)
            samples, stats = creator.create([path])

            self.assertGreater(len(samples), 0)
            for s in samples:
                self.assertNotIn("step_id", s)


if __name__ == '__main__':
    unittest.main()
