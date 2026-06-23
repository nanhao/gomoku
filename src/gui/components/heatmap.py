"""热力图 GUI 组件（基于 visualization.heatmap_color）。

本模块提供两个 tkinter 公开组件：

  - HeatmapCanvas:  225 格热力图画布 + 悬停 tooltip
  - HeatmapLegend:  颜色标尺（渐变条 + 色名标签）

### 重构说明（对应 design/gui_heatmap_reuse_design.md）

- 颜色映射核心算法复用 visualization.heatmap_color：
  * normalize_rewards(rewards, mode) 做归一化
  * value_to_rgb(v, color_stops) 得到 (r, g, b) 浮点
  * _rgb_float_to_hex(...) 转为 tkinter Canvas 所需的 "#RRGGBB"
- 默认色板：白 → 黄 → 橙 → 红（DEFAULT_COLOR_STOPS，4 色顺序渐变）
- 默认归一化：log（专为 softmax 极端分布设计，次高位也能看出层次）
- tk Canvas 不支持 alpha 通道，因此只用 value_to_rgb 取 RGB，不走 rewards_to_rgba

### 废弃符号（旧 9 色实现遗物，已移除）

- HeatmapCanvas.PRIMARY_COLORS / REWARD_MIN / REWARD_MAX
- HeatmapCanvas._get_reward_color(单值入口)
- HeatmapLegend.COLORS_FULL / BASE_COLORS
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Sequence, Tuple

from visualization.heatmap_color import (
    DEFAULT_COLOR_STOPS,
    normalize_rewards,
    value_to_rgb,
)


# ---------------------------------------------------------------------------
# 模块私有：浮点 RGB → tkinter hex 字符串
# ---------------------------------------------------------------------------

def _rgb_float_to_hex(r: float, g: float, b: float) -> str:
    """把 [0, 1] 浮点 RGB 三元组转成 tkinter 的 "#RRGGBB" 字符串。

    value_to_rgb 的输出已在内部做过 clamp，这里再乘 255 四舍五入。
    即使输入略越界（如 -1e-16、1.0+1e-16）也能产出合法 hex。

    Args:
        r, g, b: 每通道 0.0 ~ 1.0（允许轻微越界，内部兜底）。

    Returns:
        形如 "#ff8c00" 的 7 字符小写 hex 串。
    """
    def _to_byte(v: float) -> int:
        # 双保险截断到 [0, 1]，再乘 255
        v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        return int(round(v * 255))

    return f"#{_to_byte(r):02x}{_to_byte(g):02x}{_to_byte(b):02x}"


# ---------------------------------------------------------------------------
# HeatmapCanvas
# ---------------------------------------------------------------------------

_ColorStops = Sequence[Tuple[float, float, float]]


class HeatmapCanvas:
    """热力图画布组件（15×15 方格 + 悬停 tooltip）。"""

    def __init__(self,
                 parent,
                 title: str = "热力图",
                 width: int = 280,
                 height: int = 280,
                 heatmap_mode: str = "log",
                 color_stops: Optional[_ColorStops] = None,
                 ) -> None:
        """
        Args:
            parent:        tk 父容器。
            title:         标题文字（对外备用，目前 canvas 内未渲染）。
            width/height:  画布像素尺寸。
            heatmap_mode:  归一化模式 "log" / "max" / "linear"（默认 "log"）。
                           具体语义见 visualization.heatmap_color.normalize_rewards。
            color_stops:   自定义色节点序列；None = DEFAULT_COLOR_STOPS
                           （白→黄→橙→红，4 色顺序渐变）。
        """
        self.parent = parent
        self.title = title
        self.width = width
        self.height = height

        # 归一化/色板配置（延迟到 draw_heatmap 才使用，构造时不校验 mode
        # 合法性：G3 要求非法 mode 在 draw_heatmap 阶段抛 ValueError）
        self.heatmap_mode = heatmap_mode
        self.color_stops = (
            color_stops if color_stops is not None else DEFAULT_COLOR_STOPS
        )

        # 格子布局参数
        self.cell_size = 16
        self.margin = 20
        self.board_size = 15

        # 框架 + 画布
        self.frame = tk.Frame(parent, bg="#F5F5DC")
        self.canvas = tk.Canvas(
            self.frame,
            width=width,
            height=height,
            bg="#E3C16F",
            highlightthickness=1,
            highlightbackground="#8B4513",
        )
        self.canvas.pack()

        # tooltip 懒创建
        self.tooltip_window: Optional[tk.Toplevel] = None

    # ------- 布局 -------

    def pack(self, **kwargs) -> None:
        """包装布局。"""
        self.frame.pack(**kwargs)

    def pack_side(self, side=tk.LEFT, **kwargs) -> None:
        """侧边布局。"""
        self.frame.pack(side=side, **kwargs)

    def clear(self) -> None:
        """清空画布。"""
        self.canvas.delete("all")

    # ------- 绘制 -------

    def draw_heatmap(self, rewards, on_cell_hover=None) -> None:
        """绘制 225 格热力图。

        流程：
          1) 先对整列 rewards 做 normalize_rewards(mode=self.heatmap_mode)
          2) 逐格 value_to_rgb(normalized[i], self.color_stops) 得到 (r,g,b)
          3) _rgb_float_to_hex 转为 tkinter fill 颜色

        注意：tooltip 显示**原始 reward**，不显示归一化值，便于用户看真实概率。

        Args:
            rewards:       长度 225 的数值列表；长度不符时静默返回（向后兼容）。
            on_cell_hover: 鼠标悬停回调 (reward, row, col, x_root, y_root)。

        Raises:
            ValueError: self.heatmap_mode 非 "log"/"max"/"linear"。
                        （由 normalize_rewards 内部抛出。）
        """
        self.clear()

        if len(rewards) != 225:
            # 兼容旧行为：长度不符直接忽略，不抛异常
            return

        # 批量归一化：只计算一次全局统计量
        normalized = normalize_rewards(rewards, mode=self.heatmap_mode)

        for i in range(225):
            row = i // self.board_size
            col = i % self.board_size
            reward_raw = rewards[i]   # 原始值供 tooltip 展示

            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size

            # 颜色：归一化 → value_to_rgb → hex
            r, g, b = value_to_rgb(normalized[i], self.color_stops)
            color = _rgb_float_to_hex(r, g, b)

            rect = self.canvas.create_rectangle(
                x - 7, y - 7, x + 8, y + 8,
                fill=color, outline="#D2691E", width=1,
            )

            # 绑定悬停事件。
            # 注意：lambda 通过**默认参数值**捕获当前循环的 reward_raw/row/col，
            # 而非在调用时才解析闭包（那样会全部取到循环末尾的最后一组值）。
            # 这是 Python 闭包中处理循环变量的标准做法。
            if on_cell_hover:
                self.canvas.tag_bind(
                    rect, "<Enter>",
                    lambda e, rv=reward_raw, rr=row, cc=col:
                        self._show_tooltip(e, rv, rr, cc, on_cell_hover),
                )
                self.canvas.tag_bind(
                    rect, "<Leave>",
                    lambda e: self._hide_tooltip(),
                )

    # ------- Tooltip -------

    def _show_tooltip(self, event, reward, row, col, callback) -> None:
        """显示 tooltip，并调用外部 hover 回调。

        tooltip 展示**原始 reward**（D7 决策）：
          - float 类型用 "{:.6g}" 格式化，最多 6 位有效数字，避免
            "0.00012345678901234567" 这种长尾数字；
          - 其他类型退化为 str()。
        """
        self._hide_tooltip()
        try:
            reward_str = f"{float(reward):.6g}"
        except (TypeError, ValueError):
            reward_str = str(reward)
        text = f"({row}, {col})  reward: {reward_str}"
        self.create_tooltip(event.x_root, event.y_root, text)
        if callback:
            callback(reward, row, col, event.x_root, event.y_root)

    def _hide_tooltip(self) -> None:
        """隐藏 tooltip（若存在）。"""
        if self.tooltip_window and self.tooltip_window.winfo_exists():
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def create_tooltip(self, x: int, y: int, text: str) -> None:
        """创建或复用 tooltip 窗口。"""
        if self.tooltip_window and self.tooltip_window.winfo_exists():
            # 复用已有窗口：只更新位置和内容
            self.tooltip_window.wm_geometry(f"+{x + 15}+{y + 15}")
            for child in self.tooltip_window.winfo_children():
                child.configure(text=text)
        else:
            # 新建 tooltip
            self.tooltip_window = tk.Toplevel(self.parent)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x + 15}+{y + 15}")
            self.tooltip_window.configure(bg="#FFF8DC")

            tk.Label(
                self.tooltip_window,
                text=text,
                font=("微软雅黑", 9),
                bg="#FFF8DC",
                fg="#5D4E37",
                relief=tk.SOLID,
                bd=1,
                padx=8,
                pady=5,
            ).pack()


# ---------------------------------------------------------------------------
# HeatmapLegend
# ---------------------------------------------------------------------------

# 默认档数：3 段 × 8 档（4 色 3 段下已足够平滑；原 9 色 40 档过密）
_DEFAULT_LEGEND_BANDS = 24


class HeatmapLegend:
    """热力图颜色标尺组件（渐变条 + 色名标签）。"""

    def __init__(self,
                 parent,
                 color_stops: Optional[_ColorStops] = None,
                 n_bands: int = _DEFAULT_LEGEND_BANDS,
                 ) -> None:
        """
        Args:
            parent:      tk 父容器。
            color_stops: 色节点；None = DEFAULT_COLOR_STOPS（4 色）。
            n_bands:     渐变条总档数（默认 24）。
        """
        self.parent = parent
        self.color_stops = (
            color_stops if color_stops is not None else DEFAULT_COLOR_STOPS
        )
        self.n_bands = n_bands

        # 框架
        self.frame = tk.Frame(parent, bg="#F5F5DC")

        # 标题
        tk.Label(
            self.frame,
            text="颜色标尺 (min → max):",
            font=("微软雅黑", 9, "bold"),
            bg="#F5F5DC",
            fg="#5D4E37",
        ).pack(anchor=tk.W, pady=(0, 5))

        # 渐变条
        gradient_bar = tk.Frame(self.frame, bg="#F5F5DC")
        gradient_bar.pack(fill=tk.X)
        for color in self._generate_gradient_colors():
            tk.Frame(
                gradient_bar,
                bg=color,
                width=10,
                height=20,
            ).pack(side=tk.LEFT)

        # 色板名称标签（显示各色节点对应颜色块 + 名称）
        labels_frame = tk.Frame(self.frame, bg="#F5F5DC")
        labels_frame.pack(fill=tk.X, pady=(2, 0))

        for stop_rgb, name in zip(self.color_stops,
                                  self._stop_labels(self.color_stops)):
            cell = tk.Frame(labels_frame, bg="#F5F5DC")
            cell.pack(side=tk.LEFT, expand=True)

            tk.Frame(
                cell,
                bg=_rgb_float_to_hex(*stop_rgb),
                width=12,
                height=12,
            ).pack()

            tk.Label(
                cell,
                text=name,
                font=("微软雅黑", 7),
                bg="#F5F5DC",
                fg="#5D4E37",
            ).pack()

    def pack(self, **kwargs) -> None:
        """包装布局。"""
        self.frame.pack(**kwargs)

    def _generate_gradient_colors(self):
        """沿 color_stops 在 [0, 1] 上等距取 n_bands 档颜色。

        实现：统一通过 visualization.value_to_rgb 逐档取色，再转 hex。
        替代原"相邻 BASE_COLORS 之间手工插值"的双层循环。

        Returns:
            长度 == n_bands 的 hex 字符串列表。
        """
        if self.n_bands <= 1:
            # 边界：只有 1 档时直接返回起点色
            r, g, b = value_to_rgb(0.0, self.color_stops)
            return [_rgb_float_to_hex(r, g, b)]

        colors = []
        for i in range(self.n_bands):
            v = i / (self.n_bands - 1)
            r, g, b = value_to_rgb(v, self.color_stops)
            colors.append(_rgb_float_to_hex(r, g, b))
        return colors

    @staticmethod
    def _stop_labels(color_stops: _ColorStops):
        """为每个色节点生成中文标签（默认 4 色板对应"白/黄/橙/红"）。

        对于非默认色板，退化为 "C1 / C2 / ..."。
        """
        if tuple(color_stops) == tuple(DEFAULT_COLOR_STOPS):
            return ("白", "黄", "橙", "红")
        return tuple(f"C{i + 1}" for i in range(len(color_stops)))
