"""ai.sample_generator 单元/集成测试

按 design/sample_generator_design.md 的 T1~T13 用例编写。
严格 TDD：本文件先行存在，实现尚未完成时应全部失败（ImportError 或 AssertionError）。
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

from ai.sample_generator import (
    SUPPORTED_SHAPES,
    generate_shape_board,
    create_sample,
    verify_samples,
)
from ai.shape_detector import detect_shapes
from ai.reward_calculator import RewardCalculator, RewardConfig


# ---------------------------------------------------------------------------
# 工具：重复调用 generate_shape_board 直到凑够 n 个成功样本
# ---------------------------------------------------------------------------

def _collect_samples(shape, n, seed=42, max_attempts=500, **kwargs):
    rg = random.Random(seed)
    results = []
    for _ in range(max_attempts):
        r = generate_shape_board(shape, rg, **kwargs)
        if r is not None:
            results.append(r)
            if len(results) >= n:
                break
    return results


# ---------------------------------------------------------------------------
# T1~T4: 进攻棋型生成合法性
# ---------------------------------------------------------------------------

class TestGenerateAttackShapes(unittest.TestCase):

    def test_T1_chong4(self):
        """T1: chong4 每个 attack_positions 长度=1，该位置落子 detect 得'连五'"""
        samples = _collect_samples("chong4", 10)
        self.assertEqual(len(samples), 10)
        for board, attack, player in samples:
            self.assertEqual(len(attack), 1)
            pos = attack[0]
            self.assertEqual(board[pos], 0, "attack 位必须为空")
            x, y = pos // 15, pos % 15
            shape_names = [s["shape"] for s in detect_shapes(board, (x, y), player)]
            self.assertIn("连五", shape_names,
                          f"chong4 进攻位落子应形成连五, got {shape_names}")

    def test_T2_huo4(self):
        """T2: huo4 每个 attack_positions 长度=2，两位置落子 detect 都得'连五'"""
        samples = _collect_samples("huo4", 10)
        self.assertEqual(len(samples), 10)
        for board, attack, player in samples:
            self.assertEqual(len(attack), 2)
            for pos in attack:
                self.assertEqual(board[pos], 0)
                x, y = pos // 15, pos % 15
                shape_names = [s["shape"] for s in detect_shapes(board, (x, y), player)]
                self.assertIn("连五", shape_names,
                              f"huo4 任一端落子应形成连五, got {shape_names}")

    def test_T3_huo3(self):
        """T3: huo3 长度=2，两端落子 detect 都含'活四'"""
        samples = _collect_samples("huo3", 10)
        self.assertEqual(len(samples), 10)
        for board, attack, player in samples:
            self.assertEqual(len(attack), 2)
            for pos in attack:
                self.assertEqual(board[pos], 0)
                x, y = pos // 15, pos % 15
                shape_names = [s["shape"] for s in detect_shapes(board, (x, y), player)]
                # 活三两端任一落子形成 4 连 + 两端开放 = 活四
                self.assertIn("活四", shape_names,
                              f"huo3 任一端落子应形成活四, got {shape_names}")

    def test_T4_huo2(self):
        """T4: huo2 长度=2，两端落子 detect 都含'活三'或'多活三'"""
        samples = _collect_samples("huo2", 10)
        self.assertEqual(len(samples), 10)
        for board, attack, player in samples:
            self.assertEqual(len(attack), 2)
            for pos in attack:
                self.assertEqual(board[pos], 0)
                x, y = pos // 15, pos % 15
                shape_names = [s["shape"] for s in detect_shapes(board, (x, y), player)]
                self.assertTrue(
                    "活三" in shape_names or "多活三" in shape_names,
                    f"huo2 任一端落子应形成活三/多活三, got {shape_names}")


# ---------------------------------------------------------------------------
# T5~T6: 样本格式正确性
# ---------------------------------------------------------------------------

class TestCreateSampleFormat(unittest.TestCase):

    def setUp(self):
        self.rc = RewardCalculator(RewardConfig())

    def _first_sample(self, shape):
        samples = _collect_samples(shape, 1)
        self.assertEqual(len(samples), 1, f"{shape} 未能生成测试用样本")
        return samples[0]

    def test_T5_mask_correctness(self):
        """T5: mask[i]==0 当且仅当 board[i]!=0"""
        board, attack, player = self._first_sample("chong4")
        sample = create_sample(board, attack, player, "chong4", self.rc)
        self.assertEqual(len(sample["mask"]), 225)
        for i in range(225):
            expected = 0 if board[i] != 0 else 1
            self.assertEqual(sample["mask"][i], expected, f"mask[{i}] 不符")

    def test_T6_output_normalization(self):
        """T6: output 总和约 1.0，且 mask=0 位 output=0"""
        board, attack, player = self._first_sample("chong4")
        sample = create_sample(board, attack, player, "chong4", self.rc)
        self.assertAlmostEqual(sum(sample["output"]), 1.0, places=5)
        for i in range(225):
            if sample["mask"][i] == 0:
                self.assertEqual(sample["output"][i], 0.0)

    def test_sample_has_metadata(self):
        """样本必须包含 _attack_positions 和 _shape 元数据"""
        board, attack, player = self._first_sample("huo3")
        sample = create_sample(board, attack, player, "huo3", self.rc)
        self.assertEqual(sample["_attack_positions"], attack)
        self.assertEqual(sample["_shape"], "huo3")
        # input 格式：225 板面 + 1 current_player
        self.assertEqual(len(sample["input"]), 226)
        self.assertEqual(int(sample["input"][225]), player)


# ---------------------------------------------------------------------------
# T7: 未知 shape
# ---------------------------------------------------------------------------

class TestShapeKeyValidation(unittest.TestCase):

    def test_T7_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            generate_shape_board("xxx", random.Random(42))

    def test_supported_shapes_contains_six(self):
        """SUPPORTED_SHAPES 应覆盖 4 进攻 + 2 防守"""
        self.assertEqual(set(SUPPORTED_SHAPES), {
            "chong4", "huo4", "huo3", "huo2",
            "defend_chong4", "defend_huo3",
        })


# ---------------------------------------------------------------------------
# T8: verify_samples 主路径
# ---------------------------------------------------------------------------

class TestVerifySamples(unittest.TestCase):

    def _make_fake_sample(self, attack, top_indices):
        """构造一个 output 分布：top_indices 按顺序占据最高的几个概率槽"""
        output = [1e-6] * 225  # 极小底噪
        # 给 top_indices 按降序分配明显更大的概率
        for rank, idx in enumerate(top_indices):
            output[idx] = 1.0 / (rank + 1)
        s = sum(output)
        output = [v / s for v in output]
        return {
            "input": [0.0] * 226,
            "output": output,
            "mask": [1] * 225,
            "_attack_positions": attack,
            "_shape": "chong4",
        }

    def test_T8_verify_hit_and_miss(self):
        """两个必中 + 一个必漏 → hit_at_N==2, failed_indices 含漏的"""
        hit_a = self._make_fake_sample(attack=[100], top_indices=[100, 50, 20])
        hit_b = self._make_fake_sample(attack=[50, 51], top_indices=[50, 51, 30])
        miss = self._make_fake_sample(attack=[10], top_indices=[20, 10, 5])

        report = verify_samples([hit_a, hit_b, miss])
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["hit_at_N"], 2)
        self.assertIn(2, report["failed_indices"])  # miss 的样本索引

    def test_verify_report_has_aux_metrics(self):
        """辅助指标必须存在：hit_at_N_plus_1/2, attack_prob_sum, prob_gap"""
        hit = self._make_fake_sample(attack=[100], top_indices=[100])
        report = verify_samples([hit])
        for key in ("hit_at_N_plus_1", "hit_at_N_plus_2",
                    "attack_prob_sum", "prob_gap", "failed_indices"):
            self.assertIn(key, report)


# ---------------------------------------------------------------------------
# T9: max_extra=0
# ---------------------------------------------------------------------------

class TestMaxExtraZero(unittest.TestCase):

    def test_T9_max_extra_zero_strict(self):
        """max_extra=0: 棋盘上只有 4 连珠 + 0~1 阻挡；attack 落子仍满足目标棋型"""
        samples = _collect_samples("chong4", 10, max_extra=0)
        self.assertGreaterEqual(len(samples), 10)
        for board, attack, player in samples:
            stone_count = sum(1 for v in board if v != 0)
            self.assertLessEqual(stone_count, 5,
                                 "max_extra=0 时棋子数 <= 5 (4 连珠 + 至多 1 阻挡)")
            pos = attack[0]
            x, y = pos // 15, pos % 15
            shape_names = [s["shape"] for s in detect_shapes(board, (x, y), player)]
            self.assertIn("连五", shape_names)


# ---------------------------------------------------------------------------
# T10: CLI smoke test
# ---------------------------------------------------------------------------

class TestCLISmoke(unittest.TestCase):

    def test_T10_cli_generate_huo3(self):
        """CLI 生成 20 个 huo3 样本写到临时文件，结构合格"""
        proj_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            result = subprocess.run(
                [sys.executable, "-m", "ai.sample_generator",
                 "--shape", "huo3",
                 "--num-samples", "20",
                 "--output", out,
                 "--print-failures", "0"],
                cwd=proj_root,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0,
                             f"exit != 0\nstdout:{result.stdout}\nstderr:{result.stderr}")
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as f:
                samples = json.load(f)
            self.assertEqual(len(samples), 20)
            for s in samples:
                self.assertIn("_attack_positions", s)
                self.assertIn("_shape", s)
                self.assertEqual(s["_shape"], "huo3")


# ---------------------------------------------------------------------------
# T11~T12: 防守棋型生成
# ---------------------------------------------------------------------------

class TestGenerateDefendShapes(unittest.TestCase):

    def test_T11_defend_chong4(self):
        """T11: defend_chong4: attacker（对手）落 attack_positions 形成连五"""
        samples = _collect_samples("defend_chong4", 5)
        self.assertEqual(len(samples), 5)
        for board, attack, defender in samples:
            self.assertEqual(len(attack), 1)
            attacker = 3 - defender
            pos = attack[0]
            self.assertEqual(board[pos], 0)
            x, y = pos // 15, pos % 15
            shape_names = [s["shape"] for s in detect_shapes(board, (x, y), attacker)]
            self.assertIn("连五", shape_names,
                          "defend_chong4: 对手落 attack 位应形成连五")

    def test_T12_defend_huo3(self):
        """T12: defend_huo3: attacker（对手）落两端形成活四"""
        samples = _collect_samples("defend_huo3", 5)
        self.assertEqual(len(samples), 5)
        for board, attack, defender in samples:
            self.assertEqual(len(attack), 2)
            attacker = 3 - defender
            for pos in attack:
                self.assertEqual(board[pos], 0)
                x, y = pos // 15, pos % 15
                shape_names = [s["shape"] for s in detect_shapes(board, (x, y), attacker)]
                self.assertIn("活四", shape_names,
                              "defend_huo3: 对手落两端都应形成活四")


# ---------------------------------------------------------------------------
# T13: 防守样本 label 尖锐度
# ---------------------------------------------------------------------------

class TestDefenseLabelSharpness(unittest.TestCase):

    def test_T13_defend_chong4_label_sharp(self):
        """T13: defend_chong4 样本 output top1 应命中 attack 位，attack_prob_sum > 0.5"""
        rc = RewardCalculator(RewardConfig())
        samples = _collect_samples("defend_chong4", 5)
        self.assertEqual(len(samples), 5)
        for board, attack, defender in samples:
            sample = create_sample(board, attack, defender, "defend_chong4", rc)
            output = sample["output"]
            top1 = max(range(225), key=lambda i: output[i])
            self.assertEqual(top1, attack[0], "top1 应正好是防守位")
            attack_prob_sum = sum(output[i] for i in attack)
            self.assertGreater(attack_prob_sum, 0.5,
                               f"防守位总概率应 > 0.5, got {attack_prob_sum}")


if __name__ == "__main__":
    unittest.main()
