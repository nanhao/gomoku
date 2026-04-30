"""sample_visualizer 测试（T11-T18）

覆盖 design/sample_visualizer_redesign.md 第 9 节测试用例 T11-T18：
- 新格式样本可视化 / 旧格式兼容 / --no-heatmap / --indices / --seed
- 空输入 / 非法参数 / CLI smoke

所有输出都写入 tmp_path，测试结束后由 pytest 自动清理。
"""

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

from ai.sample_visualizer import SampleVisualizer


# ---------------------------------------------------------------------------
# 构造样本工具
# ---------------------------------------------------------------------------

BOARD_SIZE = 15
NUM_CELLS = BOARD_SIZE * BOARD_SIZE


def _make_new_format_sample(board, attack_positions, shape="chong4",
                            current_player=1):
    """构造新格式样本（含 _attack_positions / _shape / mask）。

    output 用一个简单的概率分布近似 masked_softmax 的输出：
    - attack_positions 上较高概率；其他空位较小概率；已落子位置为 0。
    """
    mask = [1 if v == 0 else 0 for v in board]
    # 非均匀 output：attack 位概率高，其余空位给一个小正值
    output = [0.0] * NUM_CELLS
    empty_indices = [i for i, v in enumerate(board) if v == 0]
    # 每个空位一个基础小概率
    base = 0.001
    for i in empty_indices:
        output[i] = base
    # attack 位抬高
    for p in attack_positions:
        output[p] = 0.5 / max(len(attack_positions), 1)
    # 归一化
    s = sum(output)
    output = [v / s for v in output]
    return {
        "input": [float(v) for v in board] + [float(current_player)],
        "output": output,
        "mask": mask,
        "_attack_positions": list(attack_positions),
        "_shape": shape,
    }


def _make_old_format_sample(board, win_index, current_player=1):
    """构造旧 winning_move 格式样本（output 是 one-hot，没有元数据）。"""
    output = [0.0] * NUM_CELLS
    output[win_index] = 1.0
    return {
        "input": [float(v) for v in board] + [float(current_player)],
        "output": output,
    }


def _empty_board():
    return [0] * NUM_CELLS


def _board_with_stones(moves):
    """moves: [(idx, stone), ...] stone: 1=黑 2=白"""
    board = _empty_board()
    for idx, stone in moves:
        board[idx] = stone
    return board


# ---------------------------------------------------------------------------
# T11: 新格式样本可视化生成文件
# ---------------------------------------------------------------------------
class TestNewFormatVisualization:
    def test_new_format_produces_non_empty_png(self, tmp_path):
        # 构造一个包含黑白子 + 两个 attack 位的棋盘
        board = _board_with_stones([
            (112, 1), (113, 1), (114, 1),   # 中央三连
            (97, 2),                        # 一个白子
        ])
        sample = _make_new_format_sample(board, [111, 115], shape="huo3")

        out = tmp_path / "new_fmt.png"
        viz = SampleVisualizer()
        result_path = viz.visualize(
            [sample], count=1, output_path=str(out)
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
        # 应该是 PNG（前 8 字节签名）
        with open(result_path, "rb") as f:
            head = f.read(8)
        assert head[:8] == b"\x89PNG\r\n\x1a\n"

    def test_new_format_multiple_samples(self, tmp_path):
        samples = []
        for i in range(6):
            board = _board_with_stones([(112 + i, 1)])
            samples.append(_make_new_format_sample(
                board, [100 + i, 120 + i], shape="chong4"
            ))
        out = tmp_path / "multi.png"
        viz = SampleVisualizer()
        viz.visualize(samples, count=6, output_path=str(out), seed=42)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# T12: 旧格式样本可视化兼容
# ---------------------------------------------------------------------------
class TestOldFormatCompatibility:
    def test_old_format_uses_argmax_as_attack(self, tmp_path):
        # 旧样本只含 input / output（one-hot）；不含 _attack_positions / mask / _shape
        board = _board_with_stones([(112, 1), (113, 2)])
        sample = _make_old_format_sample(board, win_index=50)
        out = tmp_path / "old_fmt.png"
        viz = SampleVisualizer()
        viz.visualize([sample], count=1, output_path=str(out))
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# T13: --no-heatmap 开关
# ---------------------------------------------------------------------------
class TestNoHeatmap:
    def test_show_heatmap_false(self, tmp_path):
        board = _board_with_stones([(112, 1)])
        sample = _make_new_format_sample(board, [111])
        out = tmp_path / "no_heatmap.png"
        viz = SampleVisualizer()
        viz.visualize([sample], count=1, output_path=str(out),
                      show_heatmap=False)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# T14: --indices 精确抽取
# ---------------------------------------------------------------------------
class TestIndicesPriority:
    def test_indices_selects_exact_samples(self, tmp_path):
        # 构造 20 个不同的样本
        samples = []
        for i in range(20):
            board = _board_with_stones([(i, 1)])
            samples.append(_make_new_format_sample(board, [i + 1]))

        # 指定只画 3 个索引
        out = tmp_path / "pick_three.png"
        viz = SampleVisualizer()
        viz.visualize(samples, indices=[0, 5, 10],
                      count=6,  # 应被忽略
                      output_path=str(out))
        assert os.path.exists(out)

    def test_invalid_indices_raises_or_skipped(self, tmp_path):
        # 越界索引应抛 IndexError / ValueError 或被优雅跳过。
        # 这里主要确保不会静默输出错误图（把该断言写为"不沉默"风格）。
        board = _empty_board()
        sample = _make_new_format_sample(board, [0])
        out = tmp_path / "bad_idx.png"
        viz = SampleVisualizer()
        with pytest.raises((IndexError, ValueError)):
            viz.visualize([sample], indices=[99], output_path=str(out))


# ---------------------------------------------------------------------------
# T15: --seed 可复现
# ---------------------------------------------------------------------------
class TestSeedReproducibility:
    def test_same_seed_selects_same_subset(self, tmp_path):
        # 构造 30 个样本
        samples = []
        for i in range(30):
            board = _board_with_stones([(i, 1 if i % 2 == 0 else 2)])
            samples.append(_make_new_format_sample(board, [i]))

        # 两次相同 seed + count 抽样，输出文件应字节相同
        # （matplotlib 的输出对相同 figure 稳定）
        import random
        r1 = random.Random(1234)
        pick1 = r1.sample(range(30), 5)
        r2 = random.Random(1234)
        pick2 = r2.sample(range(30), 5)
        # 先断言 random 库的前置条件（否则测试本身失效）
        assert pick1 == pick2

        # 现在经由 SampleVisualizer 两次调用，画出的 PNG 应相同
        out1 = tmp_path / "seed_a.png"
        out2 = tmp_path / "seed_b.png"
        viz = SampleVisualizer()
        viz.visualize(samples, count=5, seed=1234, output_path=str(out1))
        viz.visualize(samples, count=5, seed=1234, output_path=str(out2))
        assert out1.read_bytes() == out2.read_bytes()

    def test_different_seed_selects_different_subset(self, tmp_path):
        samples = []
        for i in range(30):
            board = _board_with_stones([(i, 1)])
            samples.append(_make_new_format_sample(board, [i]))
        out1 = tmp_path / "sA.png"
        out2 = tmp_path / "sB.png"
        viz = SampleVisualizer()
        viz.visualize(samples, count=5, seed=1, output_path=str(out1))
        viz.visualize(samples, count=5, seed=99999, output_path=str(out2))
        # 不同 seed 应抽到不同样本 → 输出字节应不同（极小概率相同，可接受）
        assert out1.read_bytes() != out2.read_bytes()


# ---------------------------------------------------------------------------
# T16: 空样本列表抛异常
# ---------------------------------------------------------------------------
class TestEmptySamples:
    def test_empty_list_raises_value_error(self, tmp_path):
        viz = SampleVisualizer()
        with pytest.raises(ValueError):
            viz.visualize([], count=1, output_path=str(tmp_path / "x.png"))


# ---------------------------------------------------------------------------
# T17: 非法 heatmap_alpha 抛异常
# ---------------------------------------------------------------------------
class TestInvalidHeatmapAlpha:
    def test_alpha_above_one_raises(self, tmp_path):
        board = _empty_board()
        sample = _make_new_format_sample(board, [0])
        viz = SampleVisualizer()
        with pytest.raises(ValueError):
            viz.visualize([sample], count=1,
                          heatmap_alpha=1.5,
                          output_path=str(tmp_path / "x.png"))

    def test_alpha_negative_raises(self, tmp_path):
        board = _empty_board()
        sample = _make_new_format_sample(board, [0])
        viz = SampleVisualizer()
        with pytest.raises(ValueError):
            viz.visualize([sample], count=1,
                          heatmap_alpha=-0.1,
                          output_path=str(tmp_path / "x.png"))

    def test_invalid_heatmap_mode_raises(self, tmp_path):
        board = _empty_board()
        sample = _make_new_format_sample(board, [0])
        viz = SampleVisualizer()
        with pytest.raises(ValueError):
            viz.visualize([sample], count=1,
                          heatmap_mode="bogus",
                          output_path=str(tmp_path / "x.png"))

    def test_attack_position_out_of_range_raises(self, tmp_path):
        # 脏数据：_attack_positions 含越界索引（>= 225）
        # 必须显式抛错，避免静默绘制到棋盘外
        board = _empty_board()
        sample = _make_new_format_sample(board, [0])
        sample["_attack_positions"] = [225]  # 越界
        viz = SampleVisualizer()
        with pytest.raises(ValueError):
            viz.visualize([sample], count=1,
                          output_path=str(tmp_path / "x.png"))


# ---------------------------------------------------------------------------
# T18: CLI smoke
# ---------------------------------------------------------------------------
class TestCliSmoke:
    def test_cli_runs_and_produces_png(self, tmp_path):
        # 准备样本 JSON 文件
        samples = []
        for i in range(6):
            board = _board_with_stones([(112 + i, 1)])
            samples.append(_make_new_format_sample(
                board, [100 + i], shape="huo2"
            ))
        sample_file = tmp_path / "samples.json"
        sample_file.write_text(json.dumps(samples), encoding="utf-8")

        out_png = tmp_path / "cli_out.png"

        # 通过 subprocess 跑 CLI，避免污染当前 pytest 的 matplotlib 状态
        proj_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = proj_root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "ai.sample_visualizer",
             "--sample", str(sample_file),
             "--count", "3",
             "--seed", "7",
             "--output", str(out_png)],
            capture_output=True,
            text=True,
            cwd=proj_root,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert out_png.exists()
        assert out_png.stat().st_size > 0

    def test_cli_help_exits_zero(self, tmp_path):
        proj_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = proj_root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "ai.sample_visualizer", "--help"],
            capture_output=True,
            text=True,
            cwd=proj_root,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0
        # --help 应含关键字
        assert "--sample" in result.stdout
        assert "--heatmap" in result.stdout.lower() or \
               "--no-heatmap" in result.stdout
