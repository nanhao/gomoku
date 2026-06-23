"""热力图颜色映射核心算法（纯函数、无 UI 依赖）

本模块是 visualization 包的核心，提供三个用于把任意"奖励值序列"
转换为 matplotlib 可用颜色的工具函数：

  - normalize_rewards(rewards, mode)  : 归一化到 [0, 1]
  - value_to_rgb(value, color_stops)  : 单值 → RGB 三元组（色节点等距线性插值）
  - rewards_to_rgba(rewards, ...)     : 组合 API，直接得到 RGBA 列表

设计要点：
  1) 无 UI 依赖。只使用 Python 标准库 + 数学运算，不 import matplotlib / tkinter。
     方便在任意场景（脚本 / 静态图 / GUI Canvas）复用。
  2) 色板通过 `color_stops` 参数可配置：
     默认白→黄→橙→红（4 色顺序渐变，信息密度高于 2 色且直观符合"热度"语义）。
     支持 2、3、9 或任意 n (n>=2) 个色节点；节点等距分布在 [0, 1]。
  3) 归一化默认使用 log 模式：针对 masked_softmax 这类"1 个高峰 + 一大片
     接近 0"的极端分布，log 能显著提升次高位的可辨识度。

详见 design/sample_visualizer_redesign.md 第 3 / 8 节。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

# 默认色板：白 → 黄 → 橙 → 红（4 色顺序渐变）
# 每通道为 [0.0, 1.0] 浮点。
DEFAULT_COLOR_STOPS: Tuple[Tuple[float, float, float], ...] = (
    (1.00, 1.00, 1.00),   # 白
    (1.00, 0.94, 0.00),   # 黄
    (1.00, 0.55, 0.00),   # 橙
    (1.00, 0.00, 0.00),   # 红
)


# 支持的归一化模式集合
_VALID_MODES = frozenset({"log", "max", "linear"})


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    """把 value 截断到 [0.0, 1.0] 区间。"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize_rewards(rewards: Sequence[float],
                      mode: str = "log",
                      eps: float = 1e-9) -> List[float]:
    """把任意 reward 列表归一化到 [0, 1]。

    Args:
        rewards: 原始值列表（通常是 masked_softmax 输出；本实现不假设全非负，
                 但 log 模式要求非负输入，对负值会按 0 处理以避免数学域错误）。
        mode:
            - "log":    log(1 + r*K) / log(1 + max*K)，K=1/max_r（自适应）。
                        专为"1 个高峰 + 一大片接近 0"的分布设计，
                        能让次高位在视觉上可辨。默认。
            - "max":    r / max(r)。最简单直观。
            - "linear": (r - min) / (max - min)。适合已均匀分布的信号。
        eps: 数值稳定性常量（避免除零 / log(0)）。

    Returns:
        长度与 rewards 一致的 [0, 1] 列表。
        rewards 全 0 或 rewards 为空时返回与输入等长的全 0 列表。

    Raises:
        ValueError: mode 不在 {"log", "max", "linear"} 中。
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"unknown mode: {mode!r}, expected one of {sorted(_VALID_MODES)}"
        )

    if not rewards:
        return []

    n = len(rewards)

    # 检测"全零"快速路径：避免后续分母为 0
    max_r = max(rewards)
    if max_r <= eps and min(rewards) >= -eps:
        return [0.0] * n

    if mode == "max":
        # 标准 max 归一化；负值也按同一尺度缩放（一般不会进到这里）
        denom = max(max_r, eps)
        return [r / denom for r in rewards]

    if mode == "linear":
        lo = min(rewards)
        hi = max_r
        rng = max(hi - lo, eps)
        return [(r - lo) / rng for r in rewards]

    # mode == "log"
    # 自适应 K：把 max_r 映射到 1.0，再对所有值做 log(1 + r*K)
    # log 的单调性使小值也能获得可见的非零归一化值
    K = 1.0 / max(max_r, eps)
    # log(1 + r*K) 对负数无定义，故负值统一当作 0 处理
    scaled = [math.log1p(max(r, 0.0) * K) for r in rewards]
    m = max(scaled)
    if m <= eps:
        return [0.0] * n
    return [s / m for s in scaled]


# ---------------------------------------------------------------------------
# 颜色映射
# ---------------------------------------------------------------------------

def value_to_rgb(value: float,
                 color_stops: Sequence[Tuple[float, float, float]]
                             = DEFAULT_COLOR_STOPS
                 ) -> Tuple[float, float, float]:
    """把单个 [0, 1] 值映射到 RGB 三元组。

    色节点等距分布在 [0, 1]：n 个节点把区间分成 (n-1) 段，
    相邻节点之间在 RGB 空间做线性插值。

    Args:
        value: 标量；越界值会被自动截断到 [0, 1]。
        color_stops: 有序色节点列表，至少 2 个，每个 (r, g, b) 三通道 0~1。

    Returns:
        (r, g, b) 三元组，每通道 0.0 ~ 1.0 浮点。

    Raises:
        ValueError: color_stops 少于 2 个色节点。
    """
    n = len(color_stops)
    if n < 2:
        raise ValueError(f"color_stops must have at least 2 entries, got {n}")

    v = _clamp01(float(value))

    # 只有 1 段时（2 色节点）直接线性插值
    seg_len = 1.0 / (n - 1)
    idx = int(v / seg_len)
    # v == 1.0 时 idx == n - 1，需要归并到最后一段
    if idx >= n - 1:
        idx = n - 2

    # t 是本段内的归一化位置 [0, 1]
    t = (v - idx * seg_len) / seg_len

    c1 = color_stops[idx]
    c2 = color_stops[idx + 1]
    # 对插值结果的每通道再做一次 clamp，消除浮点舍入误差
    # （matplotlib 对 RGBA 做严格 [0, 1] 校验，-1e-16 也会报错）
    return (
        _clamp01(c1[0] + (c2[0] - c1[0]) * t),
        _clamp01(c1[1] + (c2[1] - c1[1]) * t),
        _clamp01(c1[2] + (c2[2] - c1[2]) * t),
    )


# ---------------------------------------------------------------------------
# 组合 API
# ---------------------------------------------------------------------------

def rewards_to_rgba(rewards: Sequence[float],
                    mode: str = "log",
                    color_stops: Sequence[Tuple[float, float, float]]
                                = DEFAULT_COLOR_STOPS,
                    alpha: float = 0.75
                    ) -> List[Tuple[float, float, float, float]]:
    """一次性把 rewards 列表转换为 matplotlib 可用的 RGBA 列表。

    等价于 `[value_to_rgb(v, color_stops) + (alpha,)
             for v in normalize_rewards(rewards, mode)]`。

    Args:
        rewards:     原始值列表。
        mode:        归一化模式，见 normalize_rewards。
        color_stops: 色板，见 value_to_rgb。
        alpha:       统一透明度（0~1）。

    Returns:
        长度等于 rewards 的 (r, g, b, alpha) 列表。
    """
    normalized = normalize_rewards(rewards, mode=mode)
    result: List[Tuple[float, float, float, float]] = []
    for v in normalized:
        r, g, b = value_to_rgb(v, color_stops)
        result.append((r, g, b, alpha))
    return result
