"""heatmap_color 模块单元测试（T1-T10）

覆盖 design/sample_visualizer_redesign.md 第 9 节测试用例 T1-T10：
- value_to_rgb:      端点 / 中段插值 / 越界截断 / 自定义色板
- normalize_rewards: max / log / linear / 全 0 / 非法 mode
- rewards_to_rgba:   alpha 附加
"""

import math
import pytest

from visualization.heatmap_color import (
    DEFAULT_COLOR_STOPS,
    normalize_rewards,
    rewards_to_rgba,
    value_to_rgb,
)


# 浮点比较容差
EPS = 1e-6


def _assert_rgb_close(actual, expected, tol=EPS):
    """断言两个 RGB 三元组各通道接近。"""
    assert len(actual) == 3
    assert len(expected) == 3
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"channel mismatch: got {actual}, expected {expected}"


# =============================================================================
# T1: value_to_rgb 端点正确
# =============================================================================
class TestValueToRgbEndpoints:
    """默认色板为白→黄→橙→红，首尾色节点应精确命中。"""

    def test_value_zero_returns_first_stop(self):
        # v=0 时应完全等于第一个色节点（白色）
        rgb = value_to_rgb(0.0)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[0])

    def test_value_one_returns_last_stop(self):
        # v=1 时应完全等于最后一个色节点（红色）
        rgb = value_to_rgb(1.0)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[-1])


# =============================================================================
# T2: value_to_rgb 中段插值
# =============================================================================
class TestValueToRgbMidpoints:
    """默认 4 色（白/黄/橙/红），等距节点 v=1/3→黄、v=2/3→橙。"""

    def test_one_third_hits_yellow(self):
        rgb = value_to_rgb(1.0 / 3.0)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[1], tol=1e-5)

    def test_two_thirds_hits_orange(self):
        rgb = value_to_rgb(2.0 / 3.0)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[2], tol=1e-5)

    def test_midpoint_between_first_two_stops(self):
        # v=1/6 处于白与黄之间的中点
        rgb = value_to_rgb(1.0 / 6.0)
        c1 = DEFAULT_COLOR_STOPS[0]
        c2 = DEFAULT_COLOR_STOPS[1]
        expected = tuple(c1[i] + (c2[i] - c1[i]) * 0.5 for i in range(3))
        _assert_rgb_close(rgb, expected, tol=1e-5)


# =============================================================================
# T3: value_to_rgb 越界截断
# =============================================================================
class TestValueToRgbClamp:
    """超出 [0, 1] 区间的值应被截断到端点。"""

    def test_negative_value_clamped_to_zero(self):
        rgb_neg = value_to_rgb(-0.5)
        rgb_zero = value_to_rgb(0.0)
        _assert_rgb_close(rgb_neg, rgb_zero)

    def test_value_above_one_clamped_to_one(self):
        rgb_high = value_to_rgb(1.5)
        rgb_one = value_to_rgb(1.0)
        _assert_rgb_close(rgb_high, rgb_one)

    def test_extreme_negative(self):
        rgb = value_to_rgb(-1000.0)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[0])

    def test_extreme_positive(self):
        rgb = value_to_rgb(1e9)
        _assert_rgb_close(rgb, DEFAULT_COLOR_STOPS[-1])


# =============================================================================
# T4: value_to_rgb 支持自定义色板（2/3/9 色）
# =============================================================================
class TestValueToRgbCustomStops:
    """至少 2 个色节点即合法，节点数改变时端点和插值都正确。"""

    def test_two_stops_endpoints(self):
        stops = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        _assert_rgb_close(value_to_rgb(0.0, stops), (0.0, 0.0, 0.0))
        _assert_rgb_close(value_to_rgb(1.0, stops), (1.0, 1.0, 1.0))

    def test_two_stops_midpoint(self):
        # 2 色等距插值，中点应为各通道均值
        stops = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        _assert_rgb_close(value_to_rgb(0.5, stops), (0.5, 0.5, 0.5))

    def test_three_stops_midpoints(self):
        # 3 色在 v=0.5 处正好是中间节点
        stops = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        _assert_rgb_close(value_to_rgb(0.5, stops), (0.5, 0.5, 0.5), tol=1e-5)

    def test_nine_stops_hit_each_node(self):
        # 9 个等距色节点：v = i / 8 应精确命中第 i 个节点（i=0..8）
        stops = tuple((i / 8.0, 0.0, 0.0) for i in range(9))
        for i in range(9):
            _assert_rgb_close(
                value_to_rgb(i / 8.0, stops),
                stops[i],
                tol=1e-5,
            )


# =============================================================================
# T5: normalize_rewards max 模式
# =============================================================================
class TestNormalizeMax:
    def test_max_mode_scales_max_to_one(self):
        result = normalize_rewards([0.2, 0.5, 1.0, 0.1], mode="max")
        assert abs(result[2] - 1.0) < EPS
        assert abs(result[0] - 0.2) < EPS
        assert abs(result[1] - 0.5) < EPS
        assert abs(result[3] - 0.1) < EPS

    def test_max_mode_preserves_ratios(self):
        result = normalize_rewards([1.0, 2.0, 4.0], mode="max")
        # 每个值 / 4.0
        assert abs(result[0] - 0.25) < EPS
        assert abs(result[1] - 0.5) < EPS
        assert abs(result[2] - 1.0) < EPS


# =============================================================================
# T6: normalize_rewards log 模式单调性
# =============================================================================
class TestNormalizeLog:
    """log 模式专为"一个高峰+大片接近0"的 softmax 分布设计，
    必须保持严格单调：r1 < r2 ⇒ norm(r1) < norm(r2)。"""

    def test_log_mode_monotonic(self):
        rewards = [0.0001, 0.001, 0.01, 0.1, 0.9]
        result = normalize_rewards(rewards, mode="log")
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1], (
                f"log mode not monotonic at idx {i}: {result}"
            )

    def test_log_mode_max_is_one(self):
        # log 模式下最大值应被归一化到 1
        rewards = [0.01, 0.1, 1.0]
        result = normalize_rewards(rewards, mode="log")
        assert abs(result[-1] - 1.0) < EPS

    def test_log_mode_boosts_small_values(self):
        # 对于极度不均的输入（1 高峰 + 小值），log 后小值应严格大于 max 归一化
        # （K=1/max 设定下 log1p 仍近似线性，所以提升是温和的；只做严格大于断言）
        rewards = [0.001, 0.002, 0.95]
        log_out = normalize_rewards(rewards, mode="log")
        max_out = normalize_rewards(rewards, mode="max")
        # 每个非最大元素，log 模式都应严格不低于 max 模式（最大值两者都是 1）
        for i in (0, 1):
            assert log_out[i] > max_out[i], (
                f"log mode should lift small values: "
                f"log_out[{i}]={log_out[i]}, max_out[{i}]={max_out[i]}"
            )


# =============================================================================
# T7: normalize_rewards linear 模式
# =============================================================================
class TestNormalizeLinear:
    def test_linear_mode_endpoints(self):
        result = normalize_rewards([2.0, 3.0, 4.0, 5.0], mode="linear")
        assert abs(result[0] - 0.0) < EPS  # min
        assert abs(result[-1] - 1.0) < EPS  # max

    def test_linear_mode_midpoint(self):
        # [0, 10] 映射时 5.0 应该落在 0.5
        result = normalize_rewards([0.0, 5.0, 10.0], mode="linear")
        assert abs(result[1] - 0.5) < EPS


# =============================================================================
# T8: normalize_rewards 全 0 输入
# =============================================================================
class TestNormalizeAllZeros:
    @pytest.mark.parametrize("mode", ["log", "max", "linear"])
    def test_all_zeros_returns_all_zeros(self, mode):
        # 全 0 输入在任何 mode 下都不应抛异常，且返回全 0
        result = normalize_rewards([0.0, 0.0, 0.0, 0.0], mode=mode)
        assert len(result) == 4
        assert all(abs(r - 0.0) < EPS for r in result)

    def test_empty_input(self):
        # 空输入返回空列表（边界情况）
        result = normalize_rewards([], mode="log")
        assert result == []


# =============================================================================
# T9: normalize_rewards 非法 mode
# =============================================================================
class TestNormalizeInvalidMode:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_rewards([1.0, 2.0], mode="bogus")

    def test_empty_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_rewards([1.0, 2.0], mode="")


# =============================================================================
# T10: rewards_to_rgba alpha 正确附加
# =============================================================================
class TestRewardsToRgba:
    def test_rgba_length_matches_input(self):
        rgba = rewards_to_rgba([0.1, 0.5, 0.9], alpha=0.75)
        assert len(rgba) == 3

    def test_rgba_has_four_components(self):
        rgba = rewards_to_rgba([0.1, 0.5, 0.9], alpha=0.75)
        for entry in rgba:
            assert len(entry) == 4

    def test_rgba_alpha_channel_preserved(self):
        rgba = rewards_to_rgba([0.1, 0.5, 0.9], alpha=0.42)
        for entry in rgba:
            assert abs(entry[3] - 0.42) < EPS

    def test_rgba_end_colors_correct(self):
        # 最大值应被归一化到 1 → 映射到最后一个色节点（默认红）
        rgba = rewards_to_rgba([0.0, 1.0], mode="max", alpha=1.0)
        last_rgb = rgba[-1][:3]
        _assert_rgb_close(last_rgb, DEFAULT_COLOR_STOPS[-1], tol=1e-5)

    def test_rgba_with_custom_color_stops(self):
        stops = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        rgba = rewards_to_rgba([0.0, 1.0], mode="max", color_stops=stops, alpha=0.5)
        _assert_rgb_close(rgba[0][:3], (0.0, 0.0, 0.0))
        _assert_rgb_close(rgba[1][:3], (1.0, 1.0, 1.0))
