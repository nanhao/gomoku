"""SampleVisualizer 样本可视化工具

读取 winning_move 模式生成的 .json 样本文件，抽样可视化：
- 每个样本绘制 15×15 棋盘（黑白子）
- 取胜位置以红色星号标记
- 支持 CLI 参数控制抽样数量、指定索引、输出路径

用法:
    python ai/sample_visualizer.py --sample <样本文件> [--count 6] [--indices 0,5,10] [--output temp/sample_visualizer.png]
"""

import json
import math
import os
import random
import sys

# 必须在 import matplotlib.pyplot 之前设置后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ──────────────────────────────────────
# 常量
# ──────────────────────────────────────

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE  # 225

# 棋盘底色（榧木色）
BOARD_BG_COLOR = '#DEB887'

# 星位点坐标（标准 15×15 棋盘的 9 个星位）
STAR_POINTS = [
    (3, 3), (3, 7), (3, 11),
    (7, 3), (7, 7), (7, 11),
    (11, 3), (11, 7), (11, 11),
]


# ──────────────────────────────────────
# 中文字体检测
# ──────────────────────────────────────

def _get_cjk_font():
    """检测系统中可用的 CJK 字体

    依次尝试 Microsoft YaHei、SimHei、SimSun，
    若均不可用则返回 None（标题中中文可能显示为方框）。

    Returns:
        可用的 CJK 字体名称，或 None
    """
    for name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        try:
            font_path = fm.findfont(fm.FontProperties(family=name))
            # findfont 在找不到时会回退到 DejaVu Sans，需要排除
            if 'DejaVu' not in font_path:
                return name
        except Exception:
            pass
    return None


# ──────────────────────────────────────
# SampleVisualizer 类
# ──────────────────────────────────────

class SampleVisualizer:
    """样本可视化器

    将 winning_move 模式的样本列表绘制为棋盘图片，
    包含黑白子和取胜位置标记。
    """

    def __init__(self):
        """初始化可视化器（配置中文字体、matplotlib 后端）"""
        # 检测并设置 CJK 字体
        cjk_font = _get_cjk_font()
        if cjk_font:
            plt.rcParams['font.family'] = cjk_font
        # 避免负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

    def visualize(self, samples, indices=None, count=6, output_path=None):
        """可视化样本列表

        Args:
            samples: 样本列表，每个元素含 input/output/mask
            indices: 指定样本索引列表（与 count 互斥，优先于 count）
            count: 随机抽样数量，默认 6
            output_path: 输出图片路径，默认 temp/sample_visualizer.png

        Returns:
            输出图片路径

        Raises:
            ValueError: 样本列表为空
        """
        # 参数校验：空样本列表
        if not samples:
            raise ValueError("样本列表不能为空")

        # 默认输出路径
        if output_path is None:
            output_path = os.path.join("temp", "sample_visualizer.png")

        # 确定要展示的样本
        selected = self._select_samples(samples, indices, count)

        # 计算 grid 布局：尽量接近正方形
        n = len(selected)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        # 创建 figure
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.5, nrows * 3.5),
        )
        # 统一为二维数组方便迭代
        if n == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        # 逐个绘制子图
        for idx_in_grid, (sample_idx, sample) in enumerate(selected):
            row = idx_in_grid // ncols
            col = idx_in_grid % ncols
            ax = axes[row][col]
            self._draw_sample(ax, sample_idx, sample)

        # 隐藏多余的空子图
        for idx_extra in range(n, nrows * ncols):
            row = idx_extra // ncols
            col = idx_extra % ncols
            axes[row][col].set_visible(False)

        # 保存图片
        plt.tight_layout()
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=130, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _select_samples(self, samples, indices, count):
        """根据索引或随机方式选取样本

        Args:
            samples: 完整样本列表
            indices: 指定索引列表（优先），或 None
            count: 随机抽样数量

        Returns:
            列表 [(原始索引, 样本), ...]
        """
        if indices is not None:
            # 指定索引模式
            result = []
            for idx in indices:
                if 0 <= idx < len(samples):
                    result.append((idx, samples[idx]))
            return result

        # 随机抽样模式：count 不超过样本总数
        actual_count = min(count, len(samples))
        chosen = random.sample(range(len(samples)), actual_count)
        # 按索引排序，保持视觉有序
        chosen.sort()
        return [(idx, samples[idx]) for idx in chosen]

    def _draw_sample(self, ax, sample_idx, sample):
        """在单个子图上绘制棋盘和取胜位置标记

        Args:
            ax: matplotlib Axes 对象
            sample_idx: 样本在原始列表中的索引
            sample: 样本字典 {"input": [...], "output": [...], "mask": [...]}
        """
        # 解析 input → 棋盘状态 + 当前玩家
        input_vec = sample["input"]
        board = np.array(input_vec[:BOARD_CELLS]).reshape(BOARD_SIZE, BOARD_SIZE)
        player = input_vec[BOARD_CELLS]  # 第 226 维：当前玩家

        # 解析 output → 取胜位置
        output_vec = np.array(sample["output"])
        win_flat = int(np.argmax(output_vec))  # 唯一为 1.0 的位置
        win_r, win_c = divmod(win_flat, BOARD_SIZE)

        # ── 绘制棋盘底色 ──
        ax.set_facecolor(BOARD_BG_COLOR)
        ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
        ax.set_ylim(-0.5, BOARD_SIZE - 0.5)

        # ── 绘制网格线 ──
        for i in range(BOARD_SIZE):
            ax.axhline(i, color='#555', lw=0.5)
            ax.axvline(i, color='#555', lw=0.5)

        # ── 绘制星位点 ──
        for sr, sc in STAR_POINTS:
            ax.plot(sc, BOARD_SIZE - 1 - sr, 'o', color='#333', ms=3, zorder=4)

        # ── 绘制黑白子 ──
        occupied_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = board[r, c]
                # 注意：y 轴需要翻转（棋盘行 0 在上方）
                y = BOARD_SIZE - 1 - r
                if val == 1:
                    # 黑子：黑色实心圆
                    ax.add_patch(plt.Circle((c, y), 0.42, color='#111', zorder=5))
                    occupied_count += 1
                elif val == 2:
                    # 白子：白色实心圆 + 黑色描边
                    ax.add_patch(plt.Circle(
                        (c, y), 0.42, color='#eee', ec='#333', lw=1.2, zorder=5,
                    ))
                    occupied_count += 1

        # ── 绘制取胜位置标记（红色星号）──
        win_y = BOARD_SIZE - 1 - win_r
        ax.scatter([win_c], [win_y], marker='*', color='red', s=200, zorder=10)

        # ── 设置标题 ──
        player_str = 'Black' if player == 1 else 'White'
        ax.set_title(
            f"Sample #{sample_idx} | P={player_str} | "
            f"{occupied_count} stones | win=({win_r},{win_c})",
            fontsize=8,
        )

        # ── 设置坐标轴 ──
        ax.set_xticks(range(BOARD_SIZE))
        ax.set_yticks(range(BOARD_SIZE))
        ax.tick_params(labelsize=5)
        ax.set_aspect('equal')


# ──────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────

def main():
    """命令行入口：解析参数、加载样本、可视化并保存"""
    import argparse

    parser = argparse.ArgumentParser(
        description='SampleVisualizer - 样本可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ai/sample_visualizer.py --sample ai/train_data/win_samples.json
  python ai/sample_visualizer.py --sample data.json --count 3
  python ai/sample_visualizer.py --sample data.json --indices 0,5,10
  python ai/sample_visualizer.py --sample data.json --output temp/my_viz.png
        """,
    )

    parser.add_argument(
        '--sample', type=str, required=True,
        help='样本文件路径（.json）',
    )
    parser.add_argument(
        '--count', type=int, default=6,
        help='随机抽样数量，默认 6',
    )
    parser.add_argument(
        '--indices', type=str, default=None,
        help='指定样本索引（逗号分隔，如 0,5,10），与 --count 互斥，优先于 --count',
    )
    parser.add_argument(
        '--output', type=str, default='temp/sample_visualizer.png',
        help='输出图片路径，默认 temp/sample_visualizer.png',
    )

    args = parser.parse_args()

    # 解析 --indices 参数
    indices = None
    if args.indices is not None:
        try:
            indices = [int(x.strip()) for x in args.indices.split(',')]
        except ValueError:
            print(f"错误: --indices 参数格式无效，应为逗号分隔的整数，如 '0,5,10'")
            sys.exit(1)

    # 加载样本文件
    if not os.path.isfile(args.sample):
        print(f"错误: 样本文件不存在: {args.sample}")
        sys.exit(1)

    print(f"加载样本文件: {args.sample}")
    with open(args.sample, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"样本数量: {len(samples)}")

    # 可视化
    visualizer = SampleVisualizer()
    output = visualizer.visualize(
        samples, indices=indices, count=args.count, output_path=args.output,
    )
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
