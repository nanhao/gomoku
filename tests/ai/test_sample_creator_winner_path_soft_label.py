"""SampleCreator winner_path 软标签化单元测试

覆盖第三十九次追加（需求 key_points.md 第 646-647 行）：
winner_path 模式下 output 的构造由硬标签（落子位=1、其余=0）替换为
**基于模型自身输出的软标签 + 最小增强到严格 top1 + clip + 归一化**。

算法契约（详见 design/sample_creator_design.md "第三十九次追加"）：
    p        = list(history[i]["output"])          # 225 维，sum≈1
    idx      = pos[0] * 15 + pos[1]
    若 idx 越界 -> output = p（原分布）
    否则若落子位已经是严格 top1 -> output = p（原样）
    否则：
        delta     = max(p) - p[idx] + EPS           # EPS = 1e-3
        per_share = delta / 224
        q[idx] = p[idx] + delta
        q[j]   = p[j] - per_share   (j != idx)
        q      = max(q, 0.0)                         # clip
        output = q / sum(q)                          # 归一化
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.sample_creator import SampleCreator, WINNER_PATH_TOP1_EPS


# ----------------------------------------------------------------------
# 测试辅助：构造带指定 output 分布 / position 的 winner_path 场景
# ----------------------------------------------------------------------

def _make_black_wins_record(keep_output: list, keep_position: list) -> dict:
    """构造一局黑胜、3 步 history 的 record。

    history[0]（黑，保留）、history[1]（白）、history[2]（黑，保留）。
    history[2] 的 output = keep_output，position = keep_position；
    history[0] 使用占位 output/position。

    Args:
        keep_output: 长度为 225 的浮点数列表（模拟该步模型 softmax 输出）
        keep_position: [row, col]
    """
    history = [
        {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},    # 黑第 1 步（保留）
        {"input": [1.0] * 226, "output": [0.0] * 225, "position": [1, 1]},    # 白第 1 步
        {"input": [2.0] * 226, "output": list(keep_output),
         "position": list(keep_position)},                                    # 黑第 2 步（保留）
    ]
    return {"winner": 1, "history": history}


def _write_record(tmpdir: str, record: dict, filename: str = "record.json") -> str:
    """把 record 写入临时目录并返回路径"""
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f)
    return path


def _uniform(prob_at_idx: float, idx: int) -> list:
    """构造 225 维分布：idx 位置概率 = prob_at_idx，其余 224 位平分剩余概率"""
    p = [(1.0 - prob_at_idx) / 224] * 225
    p[idx] = prob_at_idx
    return p


# ----------------------------------------------------------------------
# 用例 1：落子位已经是严格 top1 → 原样保留
# ----------------------------------------------------------------------
class TestSoftLabelAlreadyStrictTop1(unittest.TestCase):
    def test_soft_label_when_move_already_strict_top1(self):
        idx_pos = [3, 4]
        idx = idx_pos[0] * 15 + idx_pos[1]
        # 落子位 0.5，其他 224 位平分 0.5 → 落子位严格最大
        p = _uniform(prob_at_idx=0.5, idx=idx)
        record = _make_black_wins_record(keep_output=p, keep_position=idx_pos)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        self.assertEqual(len(samples), 2)
        # 原样保留，精确等于 history[2] 中记录的 output
        self.assertEqual(samples[1]["output"], p)


# ----------------------------------------------------------------------
# 用例 2：落子位非 top1 → 增强为严格 top1
# ----------------------------------------------------------------------
class TestSoftLabelBoostsToStrictTop1(unittest.TestCase):
    def test_soft_label_boosts_move_to_strict_top1(self):
        # 棋盘 [0,0]=0.9、[1,1]=0.05、其他≈0；落子位设为 [1,1]
        idx_move = 1 * 15 + 1
        idx_peak = 0 * 15 + 0
        p = [0.05 / 223] * 225
        p[idx_peak] = 0.9
        p[idx_move] = 0.05
        record = _make_black_wins_record(keep_output=p, keep_position=[1, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        out = samples[1]["output"]
        # 落子位严格大于其他所有位置
        other_max = max(v for j, v in enumerate(out) if j != idx_move)
        self.assertGreater(out[idx_move], other_max)


# ----------------------------------------------------------------------
# 用例 3：归一化后 sum == 1
# ----------------------------------------------------------------------
class TestSoftLabelSumIsOne(unittest.TestCase):
    def test_soft_label_sum_is_one(self):
        idx_move = 5 * 15 + 7
        p = _uniform(prob_at_idx=0.001, idx=idx_move)   # 落子位很小 → 必然触发增强
        record = _make_black_wins_record(keep_output=p, keep_position=[5, 7])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        self.assertAlmostEqual(sum(samples[1]["output"]), 1.0, places=9)


# ----------------------------------------------------------------------
# 用例 4：所有位置 >= 0
# ----------------------------------------------------------------------
class TestSoftLabelNoNegative(unittest.TestCase):
    def test_soft_label_no_negative(self):
        # 极不均分布：某位置 0.95、其余接近 0；落子位选在"其余"中
        idx_move = 7 * 15 + 8
        idx_peak = 0 * 15 + 0
        p = [0.05 / 223] * 225
        p[idx_peak] = 0.95
        p[idx_move] = 0.05 / 223
        record = _make_black_wins_record(keep_output=p, keep_position=[7, 8])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        for v in samples[1]["output"]:
            self.assertGreaterEqual(v, 0.0)


# ----------------------------------------------------------------------
# 用例 5：软标签必须读取 record 中的 output 字段
# ----------------------------------------------------------------------
class TestSoftLabelUsesRecordedOutput(unittest.TestCase):
    def test_soft_label_uses_recorded_output(self):
        """两个不同的 record.output 下，相同的 position 产生不同的 output。

        若实现忽略 history[i]["output"] 而退化为硬标签，则两个样本必然相同，
        此测试失败。
        """
        # 用相同的 keep_position，但两个不同的 record.output
        pos = [6, 6]
        idx_move = pos[0] * 15 + pos[1]

        # 分布 A：落子位附近有一个强峰
        p_a = _uniform(0.6, idx=idx_move + 1)
        p_a[idx_move] = 0.01
        # 归一：把多出来的总量平摊补偿，保持 sum≈1
        diff = 1.0 - sum(p_a)
        p_a[0] += diff

        # 分布 B：强峰放在棋盘另一边
        p_b = _uniform(0.6, idx=0)
        p_b[idx_move] = 0.01
        diff_b = 1.0 - sum(p_b)
        p_b[1] += diff_b

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = _write_record(
                tmpdir,
                _make_black_wins_record(p_a, pos),
                filename="a.json",
            )
            path_b = _write_record(
                tmpdir,
                _make_black_wins_record(p_b, pos),
                filename="b.json",
            )
            samples_a = SampleCreator(mode="winner_path").create([path_a])
            samples_b = SampleCreator(mode="winner_path").create([path_b])

        # 同一 position、不同 p → 生成的 output 必然不同
        self.assertNotEqual(samples_a[1]["output"], samples_b[1]["output"])


# ----------------------------------------------------------------------
# 用例 6：落子位与其他位置并列 top1 → 仍要抬升为严格 top1
# ----------------------------------------------------------------------
class TestSoftLabelTieTop1IsBoosted(unittest.TestCase):
    def test_soft_label_tie_top1_is_boosted(self):
        pos = [2, 3]
        idx_move = pos[0] * 15 + pos[1]
        idx_peer = 10 * 15 + 10
        # 落子位和另一个位置并列 0.3；剩余位置平分 0.4
        p = [0.4 / 223] * 225
        p[idx_move] = 0.3
        p[idx_peer] = 0.3
        record = _make_black_wins_record(keep_output=p, keep_position=pos)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        out = samples[1]["output"]
        # 落子位必须严格大于并列的那个对手位置
        self.assertGreater(out[idx_move], out[idx_peer])
        # 且大于所有其他位置
        other_max = max(v for j, v in enumerate(out) if j != idx_move)
        self.assertGreater(out[idx_move], other_max)


# ----------------------------------------------------------------------
# 用例 7：极不均分布下 clip + 归一化仍然成立
# ----------------------------------------------------------------------
class TestSoftLabelClipExtremeDistribution(unittest.TestCase):
    def test_soft_label_clip_extreme_distribution(self):
        # 某位置 0.95、其余接近 0；落子位设为其余之一（必然触发 clip）
        pos = [5, 5]
        idx_move = pos[0] * 15 + pos[1]
        idx_peak = 0 * 15 + 0
        p = [0.05 / 223] * 225
        p[idx_peak] = 0.95
        p[idx_move] = 0.05 / 223
        record = _make_black_wins_record(keep_output=p, keep_position=pos)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        out = samples[1]["output"]
        # 非负 + sum=1 + 落子位严格 top1
        self.assertTrue(all(v >= 0.0 for v in out))
        self.assertAlmostEqual(sum(out), 1.0, places=9)
        other_max = max(v for j, v in enumerate(out) if j != idx_move)
        self.assertGreater(out[idx_move], other_max)


# ----------------------------------------------------------------------
# 用例 8：EPS 语义校验（具体数值精度）
# ----------------------------------------------------------------------
class TestSoftLabelEpsRespected(unittest.TestCase):
    def test_soft_label_eps_respected(self):
        """验证抬升阈值 EPS 被严格遵循。

        构造"落子位与单一对手位并列 top1"的场景，使 delta = EPS；
        其余 223 位平均填充但远小于 top1，使 clip 不触发。
        此时归一化前 q[idx_move] - q[idx_top] = EPS + delta/224 ≈ EPS（差 ≤ 1e-5），
        归一化后 s 精确等于 1，故 out[idx_move] - out[idx_top] ≈ EPS。
        """
        pos = [4, 4]
        idx_move = pos[0] * 15 + pos[1]
        idx_top = 9 * 15 + 9

        # 落子位与 idx_top 并列 top1 = 0.1；其余 223 位平分剩余 0.8
        p = [0.8 / 223] * 225
        p[idx_move] = 0.1
        p[idx_top] = 0.1
        # 浮点微调，保证 sum 精确为 1.0
        p[0] += (1.0 - sum(p))

        record = _make_black_wins_record(keep_output=p, keep_position=pos)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        out = samples[1]["output"]

        # delta = EPS（因并列 top1 → p_max = p_move）
        # 归一化前差 = EPS + EPS/224 ≈ EPS * (1 + 1/224)
        # clip 不触发（0.8/223 ≈ 3.6e-3 >> EPS/224）→ s = 1
        expected_diff = WINNER_PATH_TOP1_EPS * (1.0 + 1.0 / 224)
        self.assertAlmostEqual(
            out[idx_move] - out[idx_top],
            expected_diff,
            places=6,
        )


# ----------------------------------------------------------------------
# 用例 9：回归 — output 不再是 one-hot
# ----------------------------------------------------------------------
class TestSoftLabelNotOneHot(unittest.TestCase):
    def test_winner_path_no_longer_one_hot(self):
        """只要 history.output 非退化（存在非零分布），生成 output 就不应是 one-hot。"""
        pos = [2, 2]
        idx_move = pos[0] * 15 + pos[1]
        # 落子位非 top1，分布充分非退化
        p = _uniform(0.6, idx=10 * 15 + 10)
        p[idx_move] = 0.05
        diff = 1.0 - sum(p)
        p[0] += diff

        record = _make_black_wins_record(keep_output=p, keep_position=pos)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        out = samples[1]["output"]
        # 落子位绝不会等于 1.0
        self.assertNotEqual(out[idx_move], 1.0)
        # 至少存在两个 > 0 的元素（即非 one-hot）
        positive_count = sum(1 for v in out if v > 0.0)
        self.assertGreater(positive_count, 1)


# ----------------------------------------------------------------------
# 用例 10：position 越界 → output = 模型原分布
# ----------------------------------------------------------------------
class TestSoftLabelInvalidPositionReturnsModelOutput(unittest.TestCase):
    def test_invalid_position_returns_model_output(self):
        # position 越界（列 > 14）
        pos = [0, 20]
        p = _uniform(prob_at_idx=0.5, idx=7 * 15 + 7)
        record = _make_black_wins_record(keep_output=p, keep_position=pos)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_record(tmpdir, record)
            samples = SampleCreator(mode="winner_path").create([path])

        self.assertEqual(len(samples), 2)
        # 越界兜底：output 直接等于模型原分布
        self.assertEqual(samples[1]["output"], p)
        # mask 仍全 1
        self.assertEqual(samples[1]["mask"], [1] * 225)


if __name__ == '__main__':
    unittest.main()
