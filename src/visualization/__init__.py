"""visualization 包

跨 UI、跨场景的通用可视化工具集。
当前仅包含热力图颜色映射核心算法（无 UI 依赖），
未来 GUI 模块重构时可直接复用。

参见: design/sample_visualizer_redesign.md
"""

from visualization.heatmap_color import (
    DEFAULT_COLOR_STOPS,
    normalize_rewards,
    rewards_to_rgba,
    value_to_rgb,
)

__all__ = [
    "DEFAULT_COLOR_STOPS",
    "normalize_rewards",
    "rewards_to_rgba",
    "value_to_rgb",
]
