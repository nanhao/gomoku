"""Heatmap 组件测试 - TestHeatmapCanvas, TestHeatmapLegend"""

import sys
import os
import tkinter as tk
import unittest

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.heatmap import HeatmapCanvas, HeatmapLegend


class TestHeatmapCanvas:
    """测试热力图画布组件"""

    def setup_method(self):
        """测试前准备"""
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏窗口

    def teardown_method(self):
        """测试后清理"""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略

    def test_init(self):
        """测试初始化"""
        heatmap = HeatmapCanvas(self.root, title="测试热力图")
        assert heatmap is not None
        assert heatmap.title == "测试热力图"
        assert heatmap.width == 280
        assert heatmap.height == 280

    def test_init_custom_size(self):
        """测试自定义尺寸初始化"""
        heatmap = HeatmapCanvas(self.root, title="自定义", width=300, height=300)
        assert heatmap.width == 300
        assert heatmap.height == 300

    def test_clear(self):
        """测试清空热力图"""
        heatmap = HeatmapCanvas(self.root)
        heatmap.clear()
        assert True  # 应该正常执行不报错

    def test_draw_heatmap_valid(self):
        """测试绘制有效热力图"""
        heatmap = HeatmapCanvas(self.root)
        rewards = [0] * 225  # 225个位置

        heatmap.draw_heatmap(rewards)
        assert True

    def test_draw_heatmap_invalid_length(self):
        """测试绘制无效长度热力图"""
        heatmap = HeatmapCanvas(self.root)
        rewards = [0] * 100  # 错误的长度

        heatmap.draw_heatmap(rewards)
        # 应该正常处理不报错
        assert True

    def test_draw_heatmap_with_values(self):
        """测试绘制带数值的热力图"""
        heatmap = HeatmapCanvas(self.root)
        rewards = [i / 224 for i in range(225)]  # [0, 1] 范围内的不同数值

        heatmap.draw_heatmap(rewards)
        assert True

    def test_draw_heatmap_with_softmax_values(self):
        """[primary] 测试绘制 Softmax 输出的热力图 [0, 1]"""
        heatmap = HeatmapCanvas(self.root)
        # 模拟 Softmax 输出：概率分布，和为 1
        import random
        random.seed(42)
        rewards = [random.random() for _ in range(225)]
        # 归一化使和接近 1（模拟 Softmax 特性）
        total = sum(rewards)
        rewards = [r / total for r in rewards]

        heatmap.draw_heatmap(rewards)
        assert True

    def test_get_reward_color_zero(self):
        """测试获取零值颜色"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(0)

        assert color is not None
        assert isinstance(color, str)
        assert color.startswith('#')

    def test_get_reward_color_high(self):
        """测试获取高值颜色（接近 1）"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(0.75)

        assert color is not None
        assert isinstance(color, str)
        assert color.startswith('#')

    def test_get_reward_color_low(self):
        """测试获取低值颜色（接近 0）"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(0.25)

        assert color is not None
        assert isinstance(color, str)
        assert color.startswith('#')

    def test_get_reward_color_max(self):
        """测试获取最大值颜色 (1.0 -> 红色)"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(1.0)

        assert color is not None
        # 1.0 对应红色（忽略大小写）
        assert color.lower() == "#ff0000"

    def test_get_reward_color_min(self):
        """测试获取最小值颜色 (0.0 -> 黑色)"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(0.0)

        assert color is not None
        # 0.0 对应黑色（忽略大小写）
        assert color.lower() == "#000000"

    def test_get_reward_color_middle(self):
        """测试获取中间值颜色 (0.5 -> 白色)"""
        heatmap = HeatmapCanvas(self.root)
        color = heatmap._get_reward_color(0.5)

        assert color is not None
        # 0.5 对应白色（忽略大小写）
        assert color.lower() == "#ffffff"

    def test_get_reward_color_out_of_range(self):
        """测试超出范围的值被正确截断"""
        heatmap = HeatmapCanvas(self.root)
        
        # 大于 1 的值应被截断为 1
        color_high = heatmap._get_reward_color(2.0)
        color_max = heatmap._get_reward_color(1.0)
        assert color_high == color_max
        
        # 小于 0 的值应被截断为 0
        color_low = heatmap._get_reward_color(-1.0)
        color_min = heatmap._get_reward_color(0.0)
        assert color_low == color_min

    def test_primary_colors_defined(self):
        """测试正色定义"""
        heatmap = HeatmapCanvas(self.root)

        assert len(heatmap.PRIMARY_COLORS) == 9
        # 验证是 RGB 元组
        for color in heatmap.PRIMARY_COLORS:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_reward_range_constants(self):
        """测试 reward 值域常量 [0, 1]（与 Softmax 输出匹配）"""
        heatmap = HeatmapCanvas(self.root)

        # 验证值域常量存在且正确
        assert hasattr(heatmap, 'REWARD_MIN')
        assert hasattr(heatmap, 'REWARD_MAX')
        assert heatmap.REWARD_MIN == 0.0
        assert heatmap.REWARD_MAX == 1.0

    def test_create_tooltip(self):
        """测试创建 tooltip 窗口"""
        heatmap = HeatmapCanvas(self.root)
        heatmap.create_tooltip(100, 100, "reward: 500")

        assert heatmap.tooltip_window is not None
        assert heatmap.tooltip_window.winfo_exists()
        heatmap._hide_tooltip()

    def test_hide_tooltip(self):
        """测试隐藏 tooltip"""
        heatmap = HeatmapCanvas(self.root)
        heatmap.create_tooltip(100, 100, "test")
        heatmap._hide_tooltip()

        assert heatmap.tooltip_window is None

    def test_hide_tooltip_when_none(self):
        """测试 tooltip 为空时隐藏不抛异常"""
        heatmap = HeatmapCanvas(self.root)
        # 未创建过 tooltip，直接隐藏不应报错
        heatmap._hide_tooltip()
        assert heatmap.tooltip_window is None

    def test_show_tooltip_creates_window(self):
        """回归：鼠标悬停应创建 tooltip 窗口显示 reward 值

        复现 Bug：_show_tooltip 只调用回调而不创建 tooltip 窗口，
        导致鼠标悬停时无任何提示显示。
        """
        heatmap = HeatmapCanvas(self.root)
        rewards = [12345] * 225
        heatmap.draw_heatmap(rewards)

        # 模拟 _show_tooltip 直接调用
        class FakeEvent:
            x_root = 200
            y_root = 200

        heatmap._show_tooltip(FakeEvent(), 12345, 3, 5, None)

        assert heatmap.tooltip_window is not None
        assert heatmap.tooltip_window.winfo_exists()
        # tooltip 文本应包含坐标和 reward
        label = heatmap.tooltip_window.winfo_children()[0]
        assert "reward: 12345" in label.cget("text")
        assert "3" in label.cget("text")  # row
        assert "5" in label.cget("text")  # col
        heatmap._hide_tooltip()


class TestHeatmapLegend:
    """测试热力图颜色标尺组件"""

    def setup_method(self):
        """测试前准备"""
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        """测试后清理"""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略

    def test_init(self):
        """测试初始化"""
        legend = HeatmapLegend(self.root)
        assert legend is not None

    def test_colors_full_defined(self):
        """测试完整颜色定义"""
        legend = HeatmapLegend(self.root)

        assert len(legend.COLORS_FULL) == 9
        # 验证格式 (名称, 颜色值)
        for name, color in legend.COLORS_FULL:
            assert isinstance(name, str)
            assert color.startswith('#')

    def test_base_colors_defined(self):
        """测试基础颜色定义"""
        legend = HeatmapLegend(self.root)

        assert len(legend.BASE_COLORS) == 9
        # 验证是 RGB 元组
        for color in legend.BASE_COLORS:
            assert len(color) == 3

    def test_generate_gradient_colors(self):
        """测试生成渐变颜色"""
        legend = HeatmapLegend(self.root)
        colors = legend._generate_gradient_colors()

        # 9种颜色，8个区间，每个区间 5 档 = 40种颜色
        assert len(colors) == 40
        # 验证都是有效的颜色值
        for color in colors:
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB
