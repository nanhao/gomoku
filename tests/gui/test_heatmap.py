"""Heatmap 组件测试

本测试文件在"GUI 热力图复用 visualization 模块"重构后同步调整：

- 原 HeatmapCanvas 的 9 色板 + 无归一化 + _get_reward_color(单值) 入口已废弃。
- 新实现改用 visualization.heatmap_color 的 normalize_rewards + value_to_rgb 组合，
  通过 heatmap_mode / color_stops 参数暴露可配置性。
- 对应的硬编码色值断言（0.0→黑 / 0.5→白 / 1.0→红）已删除，改为新 4 色语义。

测试清单见设计文档 design/gui_heatmap_reuse_design.md §7.1。
"""

import os
import sys
import tkinter as tk

import pytest

# 确保项目根目录在路径中
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.heatmap import (   # noqa: E402
    HeatmapCanvas,
    HeatmapLegend,
    _rgb_float_to_hex,
)
from visualization.heatmap_color import DEFAULT_COLOR_STOPS   # noqa: E402


# ---------------------------------------------------------------------------
# TestHeatmapCanvas
# ---------------------------------------------------------------------------


class TestHeatmapCanvas:
    """测试热力图画布组件。"""

    def setup_method(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ----- 基础初始化与布局（保留） -----

    def test_init(self):
        heatmap = HeatmapCanvas(self.root, title="测试热力图")
        assert heatmap is not None
        assert heatmap.title == "测试热力图"
        assert heatmap.width == 280
        assert heatmap.height == 280

    def test_init_custom_size(self):
        heatmap = HeatmapCanvas(self.root, title="自定义", width=300, height=300)
        assert heatmap.width == 300
        assert heatmap.height == 300

    def test_clear(self):
        heatmap = HeatmapCanvas(self.root)
        heatmap.clear()
        assert True  # 正常执行即可

    # ----- 基础绘制（保留） -----

    def test_draw_heatmap_valid(self):
        """225 个零值应能正常绘制。"""
        heatmap = HeatmapCanvas(self.root)
        rewards = [0.0] * 225
        heatmap.draw_heatmap(rewards)

    def test_draw_heatmap_invalid_length(self):
        """长度不等于 225 应当静默返回（向后兼容）。"""
        heatmap = HeatmapCanvas(self.root)
        heatmap.draw_heatmap([0] * 100)

    def test_draw_heatmap_with_values(self):
        heatmap = HeatmapCanvas(self.root)
        rewards = [i / 224 for i in range(225)]
        heatmap.draw_heatmap(rewards)

    def test_draw_heatmap_with_softmax_values(self):
        """[primary] Softmax 分布下能正常绘制。"""
        import random
        random.seed(42)
        rewards = [random.random() for _ in range(225)]
        total = sum(rewards)
        rewards = [r / total for r in rewards]

        heatmap = HeatmapCanvas(self.root)
        heatmap.draw_heatmap(rewards)

    # ----- Tooltip 相关（保留） -----

    def test_create_tooltip(self):
        heatmap = HeatmapCanvas(self.root)
        heatmap.create_tooltip(100, 100, "reward: 0.5")

        assert heatmap.tooltip_window is not None
        assert heatmap.tooltip_window.winfo_exists()
        heatmap._hide_tooltip()

    def test_hide_tooltip(self):
        heatmap = HeatmapCanvas(self.root)
        heatmap.create_tooltip(100, 100, "test")
        heatmap._hide_tooltip()
        assert heatmap.tooltip_window is None

    def test_hide_tooltip_when_none(self):
        heatmap = HeatmapCanvas(self.root)
        heatmap._hide_tooltip()
        assert heatmap.tooltip_window is None

    def test_show_tooltip_creates_window(self):
        """回归：悬停应创建 tooltip 并展示原始 reward 值。

        注：新设计下 tooltip 显示**原始 reward**（不是归一化后的），
        便于用户看到真实概率。
        """
        heatmap = HeatmapCanvas(self.root)
        heatmap.draw_heatmap([0.5] * 225)

        class FakeEvent:
            x_root = 200
            y_root = 200

        heatmap._show_tooltip(FakeEvent(), 0.42, 3, 5, None)

        assert heatmap.tooltip_window is not None
        assert heatmap.tooltip_window.winfo_exists()
        label = heatmap.tooltip_window.winfo_children()[0]
        text = label.cget("text")
        assert "reward: 0.42" in text
        assert "3" in text   # row
        assert "5" in text   # col
        heatmap._hide_tooltip()

    # ----- G1 ~ G8：新契约 -----

    def test_g1_default_mode_is_log(self):
        """G1：默认归一化模式应为 log。"""
        heatmap = HeatmapCanvas(self.root)
        assert heatmap.heatmap_mode == "log"

    def test_g2_custom_mode_accepted(self):
        """G2：构造函数接受 heatmap_mode 参数并保留。"""
        heatmap = HeatmapCanvas(self.root, heatmap_mode="max")
        assert heatmap.heatmap_mode == "max"
        # 绘制不抛异常
        heatmap.draw_heatmap([0.1] * 225)

    def test_g3_invalid_mode_raises_on_draw(self):
        """G3：非法 mode 在 draw_heatmap 触发归一化时抛 ValueError。"""
        heatmap = HeatmapCanvas(self.root, heatmap_mode="invalid_mode")
        with pytest.raises(ValueError):
            heatmap.draw_heatmap([0.1] * 225)

    def test_g4_default_color_stops_is_4_colors(self):
        """G4：默认 color_stops 应为 4 色（白→黄→橙→红）。"""
        heatmap = HeatmapCanvas(self.root)
        assert len(heatmap.color_stops) == 4
        # 起点是白、终点是红
        assert heatmap.color_stops[0] == (1.0, 1.0, 1.0)
        assert heatmap.color_stops[-1] == (1.0, 0.0, 0.0)

    def test_g5_custom_color_stops_accepted(self):
        """G5：支持自定义色板（例如沿用旧 9 色也能工作）。"""
        custom = (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.0, 1.0),
            (0.0, 0.5, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.65, 0.0),
            (1.0, 0.75, 0.80),
            (1.0, 0.0, 0.0),
        )
        heatmap = HeatmapCanvas(self.root, color_stops=custom)
        assert len(heatmap.color_stops) == 9
        # 绘制不抛异常
        heatmap.draw_heatmap([i / 224 for i in range(225)])

    def test_g6_rgb_float_to_hex_format(self):
        """G6：浮点 RGB → hex 格式正确。"""
        assert _rgb_float_to_hex(1.0, 0.0, 0.0) == "#ff0000"
        assert _rgb_float_to_hex(0.0, 0.0, 0.0) == "#000000"
        assert _rgb_float_to_hex(1.0, 1.0, 1.0) == "#ffffff"

    def test_g7_rgb_float_to_hex_any_01_value(self):
        """G7：任何 [0, 1] 输入都能输出 7 字符 hex。"""
        samples = [0.0, 0.123, 0.5, 0.7654, 1.0]
        for r in samples:
            for g in samples:
                for b in samples:
                    s = _rgb_float_to_hex(r, g, b)
                    assert s.startswith("#")
                    assert len(s) == 7
                    # 纯十六进制字符
                    int(s[1:], 16)  # 不抛异常即合法

    def test_g8_draw_heatmap_225_softmax_log_mode(self):
        """G8：log 模式下 softmax 极端分布能绘制成功（smoke）。

        复现 sample_visualizer_redesign §1.1 提到的"一个高峰 + 大片接近 0"
        场景；旧 9 色无归一化下画面全黑，新 log 模式下应能正常绘制。
        """
        rewards = [1e-5] * 225
        rewards[42] = 0.95
        heatmap = HeatmapCanvas(self.root, heatmap_mode="log")
        heatmap.draw_heatmap(rewards)


# ---------------------------------------------------------------------------
# TestHeatmapLegend
# ---------------------------------------------------------------------------


class TestHeatmapLegend:
    """测试热力图颜色标尺组件。"""

    def setup_method(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ----- 基础初始化（保留） -----

    def test_init(self):
        legend = HeatmapLegend(self.root)
        assert legend is not None

    # ----- G9 ~ G12：新契约 -----

    def test_g9_default_bands_is_24(self):
        """G9：默认 n_bands 为 24（3 段 × 8 档，替代原 40 档）。"""
        legend = HeatmapLegend(self.root)
        colors = legend._generate_gradient_colors()
        assert len(colors) == 24
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7

    def test_g10_custom_bands(self):
        """G10：支持自定义档数。"""
        legend = HeatmapLegend(self.root, n_bands=16)
        colors = legend._generate_gradient_colors()
        assert len(colors) == 16

    def test_g11_first_color_is_white(self):
        """G11：默认色板首档 == 白（#ffffff）。"""
        legend = HeatmapLegend(self.root)
        colors = legend._generate_gradient_colors()
        assert colors[0].lower() == "#ffffff"

    def test_g12_last_color_is_red(self):
        """G12：默认色板末档 == 红（#ff0000）。"""
        legend = HeatmapLegend(self.root)
        colors = legend._generate_gradient_colors()
        assert colors[-1].lower() == "#ff0000"

    def test_g13_uses_default_color_stops(self):
        """附加：未传 color_stops 时默认使用 DEFAULT_COLOR_STOPS。"""
        legend = HeatmapLegend(self.root)
        assert legend.color_stops == DEFAULT_COLOR_STOPS

    def test_legend_n_bands_edge_case_1(self):
        """P1-1 边界：n_bands=1 时只返回 1 个颜色（首档色）。

        review 指出 `_generate_gradient_colors` 中存在 `range(n_bands - 1)` 类
        的除法/步长计算，极端 n_bands=1 需走独立分支，避免除零或空列表。
        """
        legend = HeatmapLegend(self.root, n_bands=1)
        colors = legend._generate_gradient_colors()
        assert len(colors) == 1
        # 首档应为色板起点（默认白色）
        assert colors[0].lower() == "#ffffff"


# ---------------------------------------------------------------------------
# P1-3 附加测试：tooltip reward 精度格式化
# ---------------------------------------------------------------------------


class TestTooltipPrecision:
    """验证 tooltip 用 {:.6g} 格式化 reward，避免长尾浮点显示。"""

    def setup_method(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_long_float_is_truncated_to_6_significant_digits(self):
        """长尾浮点应被截断到 6 位有效数字。"""
        heatmap = HeatmapCanvas(self.root)

        class FakeEvent:
            x_root = 10
            y_root = 10

        # 0.00012345678901234567 → "0.000123457"（{:.6g}，6 位有效数字 + 四舍五入）
        heatmap._show_tooltip(FakeEvent(), 0.00012345678901234567, 0, 0, None)
        label = heatmap.tooltip_window.winfo_children()[0]
        text = label.cget("text")
        # 关键断言：整串不含 "0.00012345678901234567" 这种长尾
        assert "0.00012345678901234567" not in text
        # 6 位有效数字下应出现 "0.000123457"
        assert "0.000123457" in text
        heatmap._hide_tooltip()

    def test_non_float_reward_falls_back_to_str(self):
        """reward 不是数字（如 None）时 fallback 到 str()。"""
        heatmap = HeatmapCanvas(self.root)

        class FakeEvent:
            x_root = 10
            y_root = 10

        heatmap._show_tooltip(FakeEvent(), None, 0, 0, None)
        label = heatmap.tooltip_window.winfo_children()[0]
        text = label.cget("text")
        assert "None" in text
        heatmap._hide_tooltip()
