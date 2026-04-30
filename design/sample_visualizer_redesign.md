# Sample Visualizer 重设计（含全新热力图方案）

## 1. 背景与目标

### 1.1 现状痛点
- 旧 `ai/sample_visualizer.py` 仅适配 `winning_move` 样本：把 `output`
  视为 one-hot，用 `argmax` 找取胜位，丢失概率分布信息。
- 新 `ai/sample_generator.py` 输出 `masked_softmax` 概率 + `_attack_positions`
  + `_shape` 元数据，旧可视化完全没利用这些信息，多进攻位只能画出 1 个。
- GUI 的 `gui/components/heatmap.py` 存在值域语义（假设 reward ∈ [0,1]
  线性映射，softmax 分布下画面大面积黑色）、40 档颜色边界重复等问题；
  但它**本次不在改造范围**，保持原样。

### 1.2 目标
1. 为 `sample_visualizer` 重写一套全新热力图方案，支持 `masked_softmax`
   分布的高质量可视化（log 归一化 + 可配置顺序色板）。
2. 把热力图的**核心算法**抽成一个**无 UI 依赖**的独立可复用模块，
   放在项目根级新建的 `visualization/` 包下；未来 GUI 那套热力图
   重构时可直接 import 复用。
3. 重写 `ai/sample_visualizer.py`：兼容新旧样本格式，支持 attack 标记
   + 热力图叠加 + shape 元数据标题显示。

### 1.3 约束
- 不动 `gui/` 包任何代码。
- 遵循 `code_rule.md`（TDD、单文件 < 500 行、丰富注释、立即 git 提交）。
- 遵循 `test_rule.md`（单元测试放 `unit/`，集成测试放 `integration/`，
  临时输出进临时目录）。

---

## 2. 模块划分

```
visualization/                            ← 新建根级包
├── __init__.py                           # 对外导出少量 API
└── heatmap_color.py                      # 纯算法，无 UI 依赖

ai/sample_visualizer.py                   ← 重写（覆盖旧文件）

tests/unit/test_heatmap_color.py          ← heatmap_color 单元测试
tests/ai/test_sample_visualizer.py        ← sample_visualizer 测试（覆盖旧）
```

### 2.1 依赖关系

```
sample_visualizer.py  ──(import)──>  visualization.heatmap_color
                     ──(import)──>  matplotlib

gui/components/heatmap.py  (不变)    （未来重构时可 import heatmap_color）
```

### 2.2 为什么放根级 `visualization/` 而非 `ai/` 或 `gui/`
- 不隶属业务模块（AI 训练）也不绑 UI 框架（tkinter/GUI）。
- 作为"跨 UI、跨场景"的通用可视化工具包，根级最中性。

---

## 3. heatmap_color 模块设计

模块内全部为**纯函数**（无类、无状态、无副作用）。

### 3.1 归一化函数

```python
def normalize_rewards(rewards: Sequence[float],
                      mode: str = "log",
                      eps: float = 1e-9) -> List[float]:
    """把任意 reward 列表归一化到 [0, 1]，用于后续颜色映射。

    Args:
        rewards: 原始值列表（通常是 masked_softmax 输出；全正值）。
        mode:
            - "log":    log(1 + r*K) / log(1 + max*K)，K 自适应。
                        专为"1 个高峰 + 一大片接近 0"的分布设计，
                        能让次高位在视觉上可辨。默认。
            - "max":    r / max(r)。最简单直观。
            - "linear": (r - min) / (max - min)。适合已均匀分布的信号。
        eps: 数值稳定性（避免除零）。

    Returns:
        长度与 rewards 一致的 [0, 1] 列表。
        rewards 全 0 时返回全 0。

    Raises:
        ValueError: mode 不合法。
    """
```

### 3.2 颜色映射函数

```python
# 默认色板：白 → 黄 → 橙 → 红（4 色顺序渐变）
DEFAULT_COLOR_STOPS: Tuple[Tuple[float, float, float], ...] = (
    (1.00, 1.00, 1.00),   # 白
    (1.00, 0.94, 0.00),   # 黄
    (1.00, 0.55, 0.00),   # 橙
    (1.00, 0.00, 0.00),   # 红
)


def value_to_rgb(value: float,
                 color_stops: Sequence[Tuple[float, float, float]]
                            = DEFAULT_COLOR_STOPS
                 ) -> Tuple[float, float, float]:
    """单个 [0, 1] 值 → RGB 三元组（每通道 0.0 ~ 1.0 float）。

    色节点等距分布在 [0, 1]（n 个节点把区间分成 n-1 段），
    相邻色节点之间做线性 RGB 插值。

    value 越界会被截断到 [0, 1]。

    Args:
        value: 标量。
        color_stops: 有序色节点列表，至少 2 个。默认白→黄→橙→红。
    """


def rewards_to_rgba(rewards: Sequence[float],
                    mode: str = "log",
                    color_stops: Sequence[Tuple[float, float, float]]
                                = DEFAULT_COLOR_STOPS,
                    alpha: float = 0.75
                    ) -> List[Tuple[float, float, float, float]]:
    """组合 API：归一化 + 颜色映射 + 附加 alpha。

    便于一次性把 rewards 转换成 matplotlib 可用的 RGBA 列表。

    Returns:
        长度同 rewards 的 (r, g, b, alpha) 列表。
    """
```

### 3.3 色节点约定
- `color_stops` 每个元素是 `(r, g, b)`，每通道 0.0~1.0 浮点。
- 至少 2 个色节点，多于 2 个均可（2/3/4/9 色都合法）。
- 节点等距：n 个色节点把 [0, 1] 分成 n-1 段。
- 相邻段之间在 RGB 空间做线性插值（不走 HSL/HSV）。

---

## 4. SampleVisualizer 类设计

只定义公开接口；私有辅助方法在实现阶段按需决定。

```python
class SampleVisualizer:
    """样本可视化器（棋盘 + 棋子 + attack 标记 + 可选热力图叠加）。"""

    def __init__(self, cjk_font: Optional[str] = None) -> None:
        """初始化。

        Args:
            cjk_font: 手工指定 CJK 字体名；None 时自动探测
                      (Microsoft YaHei / SimHei / SimSun)。
        """

    def visualize(self,
                  samples: List[Dict],
                  indices: Optional[List[int]] = None,
                  count: int = 6,
                  seed: Optional[int] = None,
                  output_path: Optional[str] = None,
                  show_heatmap: bool = True,
                  heatmap_mode: str = "log",
                  heatmap_alpha: float = 0.75,
                  ) -> str:
        """绘制样本图到 output_path。

        Args:
            samples:        样本列表（create_sample 返回的格式，或旧
                            winning_move 格式）。
            indices:        指定索引列表（优先于 count）。
            count:          随机抽样数量（默认 6）。
            seed:           抽样随机种子（None=非复现）。
            output_path:    输出 PNG 路径（默认 temp/sample_visualizer.png）。
            show_heatmap:   是否叠加热力图（默认 True）。
            heatmap_mode:   归一化模式 "log" / "max" / "linear"（默认 "log"）。
            heatmap_alpha:  热力图透明度（0~1，默认 0.75）。

        Returns:
            绝对输出路径。

        Raises:
            ValueError: samples 为空；heatmap_mode 不合法；heatmap_alpha 越界。
        """
```

---

## 5. 视觉元素与分层

每个子图从下至上分层：

| 层 | 内容 | 说明 |
|----|------|------|
| 1  | 棋盘底色（榧木色 `#DEB887`） | `ax.set_facecolor` |
| 2  | **热力图色块**（仅空位） | 半透明 RGBA，`heatmap_alpha` 控制；已落子位**不画**，避免遮挡棋子 |
| 3  | 网格线 + 9 个星位点 | 细灰线 |
| 4  | 黑白棋子 | 实心圆 |
| 5  | **attack_positions 标记** | 红色星号，`s=200`，`zorder=10` |
| 6  | 标题 | `#N │ shape=X │ P=Black │ attacks=2` |

说明：
- 热力图仅覆盖空位，既保留棋子清晰可见，又突出"哪些空位 reward 高"。
- `_attack_positions` 存在时画红星；不存在时 fallback 到 `argmax(output)`
  作为标记位（兼容旧 winning_move 样本）。

---

## 6. 样本格式兼容

```python
def _extract_metadata(sample):
    """统一抽取 attack_positions / shape / mask，兼容新旧格式。"""
    output = sample["output"]          # 两种格式都有
    input_vec = sample["input"]        # 两种格式都有
    board = input_vec[:225]

    # attack_positions：优先用元数据；否则 argmax 兜底
    attack = sample.get("_attack_positions")
    if attack is None:
        attack = [int(argmax(output))]

    # shape：新格式才有
    shape = sample.get("_shape")      # 可能为 None

    # mask：新格式有；旧格式用 board==0 推出
    mask = sample.get("mask")
    if mask is None:
        mask = [1 if v == 0 else 0 for v in board]

    return board, attack, shape, mask, output
```

---

## 7. CLI 参数

```bash
python -m ai.sample_visualizer --sample <文件> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | 样本 JSON 文件路径 | 必填 |
| `--count` | 随机抽样数量 | 6 |
| `--indices` | 指定样本索引（逗号分隔，如 `0,5,10`），优先于 `--count` | 无 |
| `--seed` | 抽样随机种子（相同 seed 可复现抽样） | 无（每次随机） |
| `--output` | 输出图片路径 | `temp/sample_visualizer.png` |
| `--no-heatmap` | 关闭热力图叠加（只画棋盘+棋子+attack 标记） | 默认叠加 |
| `--heatmap-mode` | 归一化模式 `log` / `max` / `linear` | `log` |
| `--heatmap-alpha` | 热力图透明度（0~1） | 0.75 |

---

## 8. 主流程伪代码

### 8.1 heatmap_color.normalize_rewards

```
function normalize_rewards(rewards, mode):
    if all(r == 0 for r in rewards):
        return [0] * len(rewards)

    if mode == "log":
        max_r = max(rewards)
        K = 1.0 / max(max_r, eps)      # 自适应：把最大值拉到 1
        scaled = [log(1 + r * K) for r in rewards]
        m = max(scaled)
        return [s / m for s in scaled] if m > eps else [0] * len(rewards)

    elif mode == "max":
        m = max(rewards)
        return [r / max(m, eps) for r in rewards]

    elif mode == "linear":
        lo, hi = min(rewards), max(rewards)
        rng = max(hi - lo, eps)
        return [(r - lo) / rng for r in rewards]

    else:
        raise ValueError(f"unknown mode: {mode}")
```

### 8.2 heatmap_color.value_to_rgb

```
function value_to_rgb(v, color_stops):
    v = clamp(v, 0.0, 1.0)
    n = len(color_stops)
    seg_len = 1.0 / (n - 1)

    idx = int(v / seg_len)
    if idx >= n - 1:
        idx = n - 2                      # 末端归到最后一段
    t = (v - idx * seg_len) / seg_len

    c1 = color_stops[idx]
    c2 = color_stops[idx + 1]
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )
```

### 8.3 SampleVisualizer.visualize

```
function visualize(samples, indices, count, seed, output_path, show_heatmap, mode, alpha):
    if not samples: raise ValueError
    if heatmap_alpha ∉ [0, 1]: raise ValueError
    validate mode ∈ {"log", "max", "linear"}

    output_path = output_path or "temp/sample_visualizer.png"

    # 选样本（指定索引优先；否则按 seed 随机抽）
    selected = select_samples(samples, indices, count, seed)

    # grid 布局：尽量接近正方形
    n = len(selected)
    ncols = ceil(sqrt(n));  nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3.5))
    normalize axes to 2D array

    for (sample_idx, sample) in selected:
        ax = 对应格子
        board, attack, shape, mask, output = _extract_metadata(sample)

        # 1) 棋盘底色
        ax.set_facecolor(BOARD_BG)

        # 2) 热力图层（仅空位）
        if show_heatmap:
            rgba_list = rewards_to_rgba(output, mode, DEFAULT_COLOR_STOPS, alpha)
            for i in 空位 (mask[i] == 1):
                row, col = divmod(i, 15)
                ax.add_patch(Rectangle, 位置, color=rgba_list[i])

        # 3) 网格 + 星位
        draw_grid(ax);  draw_stars(ax)

        # 4) 棋子
        for (r, c) in 每个非空位:
            draw_stone(ax, r, c, board[r*15+c])

        # 5) attack 标记（红星）
        for idx in attack:
            row, col = divmod(idx, 15)
            ax.scatter([col], [14-row], marker='*', color='red', s=200, zorder=10)

        # 6) 标题
        title = format_title(sample_idx, shape, current_player, len(attack))
        ax.set_title(title, fontsize=8)

    hide 多余子图
    plt.tight_layout()
    makedirs(output_path 的父目录)
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return abspath(output_path)
```

---

## 9. 测试用例表

| # | 测试点 | 所属模块 | 用例说明 |
|---|--------|---------|---------|
| T1 | `value_to_rgb` 端点正确 | heatmap_color | v=0 → 第一色节点；v=1 → 最后色节点（默认白/红） |
| T2 | `value_to_rgb` 中段插值 | heatmap_color | v=1/3 ≈ 第二色节点（黄）；v=2/3 ≈ 第三色节点（橙） |
| T3 | `value_to_rgb` 越界截断 | heatmap_color | v=-0.5 同 v=0；v=1.5 同 v=1 |
| T4 | `value_to_rgb` 自定义色板 | heatmap_color | 支持 2/3/9 色节点，端点与插值都正确 |
| T5 | `normalize_rewards` max 模式 | heatmap_color | max 归一化为 1.0，其他按比例 |
| T6 | `normalize_rewards` log 模式单调性 | heatmap_color | r1 < r2 ⇒ norm(r1) < norm(r2) |
| T7 | `normalize_rewards` linear 模式 | heatmap_color | [min, max] 精确映射到 [0, 1] |
| T8 | `normalize_rewards` 全 0 输入 | heatmap_color | 返回全 0，不抛异常 |
| T9 | `normalize_rewards` 非法 mode | heatmap_color | 抛 ValueError |
| T10 | `rewards_to_rgba` alpha 正确附加 | heatmap_color | 第 4 分量 == 传入的 alpha |
| T11 | 新格式样本可视化生成文件 | sample_visualizer | 含 `_attack_positions` 的样本，输出 PNG 存在且非空 |
| T12 | 旧格式样本可视化兼容 | sample_visualizer | 无元数据的 winning_move 样本，argmax 作为标记位，PNG 存在 |
| T13 | `--no-heatmap` 开关 | sample_visualizer | 关闭热力图后仍正常出图 |
| T14 | `--indices` 精确抽取 | sample_visualizer | 传入 [0, 5, 10]，只画这 3 个 |
| T15 | `--seed` 可复现 | sample_visualizer | 相同 seed 两次调用选到同一批样本 |
| T16 | 空样本列表抛异常 | sample_visualizer | `samples=[]` → ValueError |
| T17 | 非法 heatmap_alpha 抛异常 | sample_visualizer | alpha=1.5 / -0.1 → ValueError |
| T18 | CLI smoke | sample_visualizer | subprocess 调用不报错，退出码 0 |

---

## 10. 文件行数预算

| 文件 | 预估 | 上限 |
|------|------|------|
| `visualization/heatmap_color.py` | ~150 | 500 |
| `visualization/__init__.py` | ~15 | 500 |
| `ai/sample_visualizer.py`（重写） | ~350 | 500 |
| `tests/unit/test_heatmap_color.py` | ~200 | 500 |
| `tests/ai/test_sample_visualizer.py`（重写） | ~300 | 500 |

---

## 11. 不纳入范围

以下**明确排除**，本次不处理：

| 项 | 排除原因 |
|----|---------|
| `gui/components/heatmap.py` / `constants.py` 任何修改 | 用户明确要求 GUI 包不动 |
| `gui/components/replay_viewer.py` 中的热力图绘制 | 同上 |
| 旧 `gui/test_heatmap.py` 调整 | 同上 |
| diverging 色板（对称 0 居中） | 目前只为 softmax 全正信号服务；将来 GUI 复用再加 |
| 热力图交互（hover / tooltip） | 静态 PNG，无交互需求 |
| 对数底数精调 / 自适应 K 的调参 | 先默认 K=1/max 实现，留观察 |
| 热力图 Legend（颜色条） | 当前需求够用；后续需要再加 |

---

## 12. 关键设计决策记录

| # | 决策 | 理由 |
|---|------|------|
| D1 | 核心模块放根级 `visualization/` | 中性位置，跨 AI/GUI 复用 |
| D2 | heatmap_color 全部纯函数、无类 | 易测、易复用、无状态 |
| D3 | 默认色板 白→黄→橙→红（4 色顺序） | 信息密度高于 2 色，简洁于 9 色；符合"热度"直觉 |
| D4 | 默认归一化 `log` 而非 `max` | 针对 softmax 极端分布，次高位可辨 |
| D5 | 热力图仅覆盖空位，不覆盖棋子 | 保证棋形可读 |
| D6 | 兼容旧样本（无元数据走 argmax） | 避免破坏 replay 等历史调用场景 |
| D7 | 重写而非渐进改造旧文件 | 旧逻辑几乎全废，整体重写更简洁 |
