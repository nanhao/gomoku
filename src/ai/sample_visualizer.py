"""SampleVisualizer — 样本可视化工具（重设计版）

按 design/sample_visualizer_redesign.md 实现：

- 每个样本绘制 15×15 棋盘：榧木色底 → 热力图（仅空位）→ 网格/星位 → 棋子 → attack 标记 → 标题
- 热力图基于 visualization.heatmap_color（可配置色板、默认 log 归一化、白→黄→橙→红）
- 兼容新旧样本格式：
    新格式（sample_generator）含 ``_attack_positions`` / ``_shape`` / ``mask``
    旧格式（winning_move）缺元数据，走 ``argmax(output)`` 兜底
- 主要通过 ``SampleVisualizer.visualize`` 调用；CLI 参数见 __main__ 块

用法::

    python -m ai.sample_visualizer --sample <样本文件> [选项]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# 必须在 import matplotlib.pyplot 之前设置后端，避免 tkinter 初始化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

import numpy as np

from visualization.heatmap_color import (
    DEFAULT_COLOR_STOPS,
    rewards_to_rgba,
)


# ---------- 常量 ----------

BOARD_SIZE: int = 15
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 225

# 棋盘底色（榧木色）
BOARD_BG_COLOR: str = "#DEB887"

# 9 个星位点（列, 行），原点左下
STAR_POINTS: Tuple[Tuple[int, int], ...] = (
    (3, 3), (3, 7), (3, 11),
    (7, 3), (7, 7), (7, 11),
    (11, 3), (11, 7), (11, 11),
)

# 默认输出路径（相对项目根）
DEFAULT_OUTPUT_PATH: str = "temp/sample_visualizer.png"

# 支持的归一化模式（在 SampleVisualizer 内做入参校验时引用）
_VALID_HEATMAP_MODES = frozenset({"log", "max", "linear"})


# ---------- CJK 字体探测 ----------

def _get_cjk_font() -> Optional[str]:
    """探测系统中可用的 CJK 字体。

    依次尝试 Microsoft YaHei / SimHei / SimSun，
    均不可用时返回 None（标题中文字可能显示为方框）。
    """
    for name in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            path = fm.findfont(
                fm.FontProperties(family=name), fallback_to_default=False
            )
            if path and os.path.exists(path):
                return name
        except Exception:
            continue
    return None


# ---------- 元数据抽取（兼容新旧格式） ----------

def _extract_metadata(sample: Dict) -> Dict:
    """统一抽取 board / attack / shape / mask / output / current_player。

    - 新格式（sample_generator）含 ``_attack_positions`` / ``_shape`` / ``mask``
    - 旧格式（winning_move）走 ``argmax(output)`` 作为 attack 兜底；
      ``mask`` 从 board==0 推导；``_shape`` 为 None
    """
    input_vec = sample["input"]
    output = list(sample["output"])

    # input 末尾是 current_player（1=黑, 2=白）；旧 sample 也遵守此约定
    board = [int(v) for v in input_vec[:NUM_CELLS]]
    current_player = int(input_vec[NUM_CELLS]) if len(input_vec) > NUM_CELLS else 1

    # attack_positions：优先用元数据；否则 argmax 兜底
    attack = sample.get("_attack_positions")
    if attack is None:
        attack = [int(np.argmax(output))]
    else:
        attack = [int(i) for i in attack]
    # 防御脏数据：attack 索引必须在 [0, 225) 内，否则 divmod 会落到棋盘外
    for idx in attack:
        if not (0 <= idx < NUM_CELLS):
            raise ValueError(
                f"attack_position {idx} out of range [0, {NUM_CELLS})"
            )

    # shape（仅新格式有）
    shape = sample.get("_shape")

    # mask
    mask = sample.get("mask")
    if mask is None:
        mask = [1 if v == 0 else 0 for v in board]
    else:
        mask = [int(v) for v in mask]

    return {
        "board": board,
        "attack": attack,
        "shape": shape,
        "mask": mask,
        "output": output,
        "current_player": current_player,
    }


# ---------- 样本选择（indices 优先 / count 随机抽样） ----------

def _select_samples(samples: List[Dict],
                    indices: Optional[Sequence[int]],
                    count: int,
                    seed: Optional[int]) -> List[Tuple[int, Dict]]:
    """按 indices 精确取样或按 seed+count 随机取样。

    Returns:
        [(原始索引, 样本), ...] 有序列表。
    """
    n = len(samples)

    if indices is not None:
        # 显式指定：严格边界检查，越界抛 IndexError
        picked: List[Tuple[int, Dict]] = []
        for i in indices:
            if i < 0 or i >= n:
                raise IndexError(
                    f"sample index {i} out of range [0, {n})"
                )
            picked.append((int(i), samples[i]))
        return picked

    # 随机抽样：count 超过可用量则截断
    k = min(count, n)
    rng = random.Random(seed) if seed is not None else random.Random()
    chosen_idx = sorted(rng.sample(range(n), k))
    return [(i, samples[i]) for i in chosen_idx]


# ---------- 绘制辅助 ----------

def _draw_grid_and_stars(ax) -> None:
    """画 15×15 网格线 + 9 个星位圆点。"""
    # 网格：每条线跨 0~14（col 或 row）
    for i in range(BOARD_SIZE):
        ax.plot([i, i], [0, BOARD_SIZE - 1], color="#333333", linewidth=0.6,
                zorder=3)
        ax.plot([0, BOARD_SIZE - 1], [i, i], color="#333333", linewidth=0.6,
                zorder=3)
    # 星位
    for (c, r) in STAR_POINTS:
        # y 坐标翻转：row=3 对应视觉上部偏下 → y=14-r
        ax.scatter([c], [BOARD_SIZE - 1 - r], s=18, color="#333333", zorder=4)


def _draw_heatmap_layer(ax,
                        output: Sequence[float],
                        mask: Sequence[int],
                        mode: str,
                        alpha: float) -> None:
    """仅在空位（mask==1）上画半透明热力色块。

    每个色块中心对齐格点 (col, 14-row)，边长 1，比格点稍大也可；
    此处用 1.0×1.0 矩形覆盖格点所在单元格。
    """
    rgba_list = rewards_to_rgba(output, mode=mode,
                                color_stops=DEFAULT_COLOR_STOPS,
                                alpha=alpha)
    for i in range(NUM_CELLS):
        if mask[i] != 1:
            continue
        row, col = divmod(i, BOARD_SIZE)
        x = col - 0.5
        y = (BOARD_SIZE - 1 - row) - 0.5
        ax.add_patch(Rectangle(
            (x, y), 1.0, 1.0,
            facecolor=rgba_list[i],
            edgecolor="none",
            zorder=2,
        ))


def _draw_stones(ax, board: Sequence[int]) -> None:
    """根据 board 数组绘制黑白棋子（实心圆）。"""
    blacks_x: List[int] = []
    blacks_y: List[int] = []
    whites_x: List[int] = []
    whites_y: List[int] = []
    for i, v in enumerate(board):
        if v == 0:
            continue
        row, col = divmod(i, BOARD_SIZE)
        y = BOARD_SIZE - 1 - row  # 上下翻转
        if v == 1:
            blacks_x.append(col)
            blacks_y.append(y)
        elif v == 2:
            whites_x.append(col)
            whites_y.append(y)
    if blacks_x:
        ax.scatter(blacks_x, blacks_y, s=220, color="black", edgecolors="black",
                   linewidths=0.5, zorder=5)
    if whites_x:
        ax.scatter(whites_x, whites_y, s=220, color="white", edgecolors="black",
                   linewidths=0.8, zorder=5)


def _draw_attack_markers(ax, attack: Sequence[int]) -> None:
    """在 attack_positions 上画红色星号。"""
    if not attack:
        return
    xs: List[int] = []
    ys: List[int] = []
    for idx in attack:
        row, col = divmod(int(idx), BOARD_SIZE)
        xs.append(col)
        ys.append(BOARD_SIZE - 1 - row)
    ax.scatter(xs, ys, marker="*", color="red", s=220, zorder=10,
               edgecolors="black", linewidths=0.4)


def _format_title(sample_idx: int, shape: Optional[str],
                  current_player: int, n_attack: int) -> str:
    """构造单样本子图标题。"""
    player_label = "Black" if current_player == 1 else "White"
    shape_label = shape if shape is not None else "unknown"
    return (f"#{sample_idx} │ shape={shape_label} │ "
            f"P={player_label} │ attacks={n_attack}")


def _draw_one_sample(ax, sample_idx: int, sample: Dict,
                     show_heatmap: bool,
                     heatmap_mode: str,
                     heatmap_alpha: float) -> None:
    """绘制单个样本到指定 axes。"""
    meta = _extract_metadata(sample)

    # 1) 底色
    ax.set_facecolor(BOARD_BG_COLOR)

    # 2) 热力图层（仅空位）
    if show_heatmap:
        _draw_heatmap_layer(ax, meta["output"], meta["mask"],
                            heatmap_mode, heatmap_alpha)

    # 3) 网格 + 星位
    _draw_grid_and_stars(ax)

    # 4) 棋子
    _draw_stones(ax, meta["board"])

    # 5) attack 标记
    _draw_attack_markers(ax, meta["attack"])

    # 6) 标题
    ax.set_title(
        _format_title(sample_idx, meta["shape"],
                      meta["current_player"], len(meta["attack"])),
        fontsize=8,
    )

    # 坐标系：原点左下，范围 -0.5 ~ 14.5，保持正方形
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------- SampleVisualizer 类 ----------

class SampleVisualizer:
    """样本可视化器（棋盘 + 棋子 + attack 标记 + 可选热力图叠加）。"""

    def __init__(self, cjk_font: Optional[str] = None) -> None:
        """初始化。

        Args:
            cjk_font: 手工指定 CJK 字体名；None 时自动探测。
                      只影响标题；未设定时 matplotlib 用默认字体。
        """
        self._cjk_font = cjk_font if cjk_font is not None else _get_cjk_font()
        if self._cjk_font:
            plt.rcParams["font.sans-serif"] = [self._cjk_font] + \
                list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["axes.unicode_minus"] = False

    def visualize(self,
                  samples: List[Dict],
                  indices: Optional[Sequence[int]] = None,
                  count: int = 6,
                  seed: Optional[int] = None,
                  output_path: Optional[str] = None,
                  show_heatmap: bool = True,
                  heatmap_mode: str = "log",
                  heatmap_alpha: float = 0.75,
                  ) -> str:
        """绘制样本图到 output_path。

        Args:
            samples:        样本列表（新格式或旧 winning_move 格式均可）。
            indices:        指定索引列表（优先于 count）；越界会抛 IndexError。
            count:          随机抽样数量（默认 6），indices 指定时忽略。
            seed:           抽样随机种子（None=每次随机）。
            output_path:    输出 PNG 路径（默认 temp/sample_visualizer.png）。
            show_heatmap:   是否叠加热力图（默认 True）。
            heatmap_mode:   归一化模式 "log" / "max" / "linear"（默认 "log"）。
            heatmap_alpha:  热力图透明度 0~1（默认 0.75）。

        Returns:
            绝对输出路径。

        Raises:
            ValueError: samples 为空；heatmap_mode 不合法；heatmap_alpha 越界。
            IndexError: indices 中存在越界。
        """
        # 入参校验
        if not samples:
            raise ValueError("samples must be a non-empty list")
        if heatmap_mode not in _VALID_HEATMAP_MODES:
            raise ValueError(
                f"invalid heatmap_mode: {heatmap_mode!r}, "
                f"expected one of {sorted(_VALID_HEATMAP_MODES)}"
            )
        if not (0.0 <= heatmap_alpha <= 1.0):
            raise ValueError(
                f"heatmap_alpha must be in [0, 1], got {heatmap_alpha}"
            )

        output_path = output_path or DEFAULT_OUTPUT_PATH

        # 选择要画的样本
        selected = _select_samples(samples, indices, count, seed)

        # 布局：尽量接近正方形
        n = len(selected)
        ncols = max(1, math.ceil(math.sqrt(n)))
        nrows = max(1, math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.5, nrows * 3.5),
        )

        # 规整 axes 成 2D 数组（兼容 n=1 / 单行 / 单列）
        axes_arr = np.atleast_2d(axes).reshape(nrows, ncols)

        for plot_idx, (sample_idx, sample) in enumerate(selected):
            r = plot_idx // ncols
            c = plot_idx % ncols
            _draw_one_sample(
                axes_arr[r][c],
                sample_idx, sample,
                show_heatmap, heatmap_mode, heatmap_alpha,
            )

        # 隐藏多余的子图
        for plot_idx in range(n, nrows * ncols):
            r = plot_idx // ncols
            c = plot_idx % ncols
            axes_arr[r][c].axis("off")

        plt.tight_layout()

        # 确保输出目录存在
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # 保存：显式抑制 PNG 时间戳，保证同输入 → 同字节（便于复现测试）
        fig.savefig(
            output_path,
            dpi=130,
            bbox_inches="tight",
            metadata={"Creation Time": None, "CreationDate": None,
                      "Software": None},
        )
        plt.close(fig)

        return os.path.abspath(output_path)


# ---------- CLI ----------

def _parse_indices(raw: Optional[str]) -> Optional[List[int]]:
    """把 "0,5,10" 解析为 [0, 5, 10]。"""
    if raw is None or raw.strip() == "":
        return None
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    try:
        return [int(s) for s in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--indices must be comma-separated integers, got {raw!r}"
        ) from e


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai.sample_visualizer",
        description="样本可视化工具（含新版热力图）。"
                    "输入样本 JSON 文件，输出含棋盘/棋子/attack/热力图的 PNG。",
    )
    parser.add_argument(
        "--sample", required=True,
        help="样本 JSON 文件路径（由 ai.sample_generator 等生成）",
    )
    parser.add_argument(
        "--count", type=int, default=6,
        help="随机抽样数量（默认 6）；--indices 指定时此参数被忽略",
    )
    parser.add_argument(
        "--indices", type=str, default=None,
        help="指定要画的样本索引（逗号分隔，如 '0,5,10'），优先于 --count",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="抽样随机种子（相同 seed 可复现抽样；默认每次随机）",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"输出 PNG 路径（默认 {DEFAULT_OUTPUT_PATH}）",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="关闭热力图叠加（默认开启）",
    )
    parser.add_argument(
        "--heatmap-mode", type=str, default="log",
        choices=sorted(_VALID_HEATMAP_MODES),
        help="热力图归一化模式（默认 log；适配 masked_softmax 极端分布）",
    )
    parser.add_argument(
        "--heatmap-alpha", type=float, default=0.75,
        help="热力图透明度 0~1（默认 0.75）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口。"""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # 读样本文件
    with open(args.sample, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list) or not samples:
        print(f"error: sample file {args.sample} must contain a non-empty list",
              file=sys.stderr)
        return 1

    indices = _parse_indices(args.indices)

    viz = SampleVisualizer()
    out_path = viz.visualize(
        samples,
        indices=indices,
        count=args.count,
        seed=args.seed,
        output_path=args.output,
        show_heatmap=not args.no_heatmap,
        heatmap_mode=args.heatmap_mode,
        heatmap_alpha=args.heatmap_alpha,
    )
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
