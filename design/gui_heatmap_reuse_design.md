# GUI 热力图复用 visualization 模块 设计文档

## 1. 背景与目标

### 1.1 现状
- `gui/components/heatmap.py` 自带一套独立的颜色映射：
  - **9 色渐变**：黑→紫→蓝→绿→白→黄→橙→粉→红
  - **无归一化**：直接把 reward ∈ [0, 1] 线性映射到 40 档
  - **输出 hex**：`#RRGGBB` 字符串（tkinter Canvas 的 fill 要求）
- `visualization/heatmap_color.py`（上一任务交付）提供：
  - **默认 4 色**：白→黄→橙→红
  - **log/max/linear 归一化**
  - **输出浮点 RGB / RGBA**

两者核心算法（"区段等距 + 相邻色节点 RGB 线性插值"）实质等价，但色板、归一化、输出格式皆不同，存在重复代码且 GUI 侧的可视效果在 softmax 极端分布下表现差（设计 `sample_visualizer_redesign.md` §1.1 已指出：画面大面积黑色）。

### 1.2 目标
1. **GUI 侧热力图颜色映射**复用 `visualization.heatmap_color`，消除重复
2. 色板统一为 4 色（白→黄→橙→红），信息密度更高、与 `sample_visualizer` 保持一致
3. 归一化模式可配置，默认 `log`（针对 softmax 分布）
4. Legend 标尺同步改造，与热力图本体视觉一致

### 1.3 约束
- 不改动 `visualization/` 包（已经稳定交付）
- 遵循 `code_rule.md`（TDD、单文件 < 500 行、完成后 git 提交）
- 遵循 `test_rule.md`（测试放原目录 `tests/gui/`）
- 决策依据见"重要决策-GUI 热力图重构三要素"记忆

---

## 2. 模块划分与依赖

```
visualization/heatmap_color.py              （不变，作为算法源）
   ▲
   │ import
   │
gui/components/heatmap.py                   （本次重写内部实现）
   ├─ HeatmapCanvas    (公开类，外部契约兼容 + 新增可选参数)
   ├─ HeatmapLegend    (公开类，公开接口不变，内部算法替换)
   └─ _rgb_float_to_hex  (新增私有函数：浮点 RGB → #RRGGBB)

gui/components/replay_viewer.py             （可选微调：如需传 mode）
tests/gui/test_heatmap.py                   （同步调整；见 §7）
```

### 2.1 为什么适配层放在 `gui/components/heatmap.py` 内
- `_rgb_float_to_hex` 是**tkinter 专属**适配（matplotlib 直接用浮点 RGBA）
- 不应污染 `visualization/` 的"无 UI 依赖"契约
- 函数体 3 行，放在使用方最直接

### 2.2 tkinter 不支持 alpha 的处理
- tk Canvas 的 `fill` 属性不接受 alpha 通道（设置了也会被忽略）
- 方案：**直接调用 `value_to_rgb`（返回 3 元组），不走 `rewards_to_rgba`**
- 不做 bg 手工混合——会引入与棋盘底色耦合的复杂度，本次不收益
- Legend 同理

---

## 3. 接口设计

### 3.1 HeatmapCanvas（改动要点）

```python
class HeatmapCanvas:
    """热力图画布组件（基于 visualization.heatmap_color）。"""

    def __init__(self,
                 parent,
                 title: str = "热力图",
                 width: int = 280,
                 height: int = 280,
                 heatmap_mode: str = "log",
                 color_stops=None,
                 ):
        """
        Args:
            parent:       tk 父容器
            title:        标题（对外显示用，当前未在 canvas 内渲染，保留兼容）
            width/height: 画布尺寸（px）
            heatmap_mode: 归一化模式 "log" / "max" / "linear"；默认 "log"
            color_stops:  色节点列表；None = 使用 DEFAULT_COLOR_STOPS
                          （白→黄→橙→红 4 色）
        """

    def pack(self, **kwargs) -> None: ...
    def pack_side(self, side=tk.LEFT, **kwargs) -> None: ...
    def clear(self) -> None: ...

    def draw_heatmap(self,
                     rewards: list,
                     on_cell_hover=None) -> None:
        """绘制热力图。
        - 长度不为 225 时直接返回（不抛异常，向后兼容）。
        - 内部先调用 normalize_rewards(rewards, self.heatmap_mode) 得到 [0,1]
        - 逐格调用 value_to_rgb(v, self.color_stops) 再转 hex
        """

    def create_tooltip(self, x: int, y: int, text: str) -> None: ...
```

**废弃符号**（public API 移除）：
- 类常量 `PRIMARY_COLORS`（9 色）
- 类常量 `REWARD_MIN` / `REWARD_MAX`（不再使用）
- 私有方法 `_get_reward_color(reward)` —— 归一化必须基于整列 rewards，单值入口已不成立

**保留不变**：
- `frame` / `canvas` 属性
- tooltip 相关 `_show_tooltip` / `_hide_tooltip` / `create_tooltip`
- `draw_heatmap` 签名

### 3.2 HeatmapLegend（改动要点）

```python
class HeatmapLegend:
    """热力图颜色标尺组件。"""

    def __init__(self,
                 parent,
                 color_stops=None,
                 n_bands: int = 24,
                 ):
        """
        Args:
            parent:      tk 父容器
            color_stops: 色节点列表；None = DEFAULT_COLOR_STOPS
            n_bands:     渐变条总档数（默认 24 = 3 段 × 8 档；原为 40 档）
                         注：4 色 3 段下，24 档视觉上已平滑过渡
        """

    def pack(self, **kwargs) -> None: ...
```

**废弃符号**（public API 移除）：
- 类常量 `COLORS_FULL`（9 色名+hex 列表）
- 类常量 `BASE_COLORS`（9 色 RGB 列表）

**保留**：
- `frame` 属性
- `_generate_gradient_colors()`（内部实现替换，但**保留方法签名**以便现有测试可迁移）
  - 新实现：调用 `value_to_rgb(i/(n_bands-1))` 逐档生成

**标签文字**：
- 渐变条下方改为 4 色名并排显示："白 / 黄 / 橙 / 红"
- 共 4 个色块，均匀分布

---

## 4. 视觉对比

| 维度 | 旧 | 新 |
|------|-----|------|
| 色节点数 | 9 | 4 |
| 色板 | 黑-紫-蓝-绿-白-黄-橙-粉-红 | 白-黄-橙-红 |
| 0.0 的颜色 | 黑（#000000） | 白（#FFFFFF） |
| 0.5 的颜色 | 白（#FFFFFF） | 橙（#FF8C00 附近） |
| 1.0 的颜色 | 红（#FF0000） | 红（#FF0000） |
| 归一化 | 无 | log（默认）/ max / linear |
| Legend 档数 | 40 | 24 |

**语义变化**：
- 旧：0 看起来"无热度"=黑；新：0 看起来"无热度"=白（更贴近"什么都没有"的视觉认知）
- 旧：softmax 极端分布下几乎全黑；新：log 归一化让次高位也能看出层次

---

## 5. tkinter 适配细节

### 5.1 浮点 RGB → hex 字符串

```python
def _rgb_float_to_hex(r: float, g: float, b: float) -> str:
    """(0.0~1.0, 0.0~1.0, 0.0~1.0) → "#RRGGBB"
    value_to_rgb 已 clamp 到 [0,1]，此处再乘 255 四舍五入。
    """
    ri = int(round(r * 255))
    gi = int(round(g * 255))
    bi = int(round(b * 255))
    return f"#{ri:02x}{gi:02x}{bi:02x}"
```

放在 `gui/components/heatmap.py` 文件顶层（模块私有，不进 `__all__`）。

### 5.2 alpha 处理
- **不使用** `rewards_to_rgba`（含 alpha 通道）
- **使用** `normalize_rewards` + `value_to_rgb` 两步组合
- 格子的 `outline="#D2691E"` 保留（辅助区分格子边界）

---

## 6. 主流程伪代码

### 6.1 HeatmapCanvas.draw_heatmap

```
function draw_heatmap(rewards, on_cell_hover):
    self.clear()
    if len(rewards) != 225:
        return

    # 关键变化：批量归一化 → 逐格颜色映射
    normalized = normalize_rewards(rewards, mode=self.heatmap_mode)

    for i in 0..224:
        row, col = i // 15, i % 15
        x = margin + col * cell_size
        y = margin + row * cell_size

        r, g, b = value_to_rgb(normalized[i], self.color_stops)
        color = _rgb_float_to_hex(r, g, b)

        rect = canvas.create_rectangle(
            x-7, y-7, x+8, y+8,
            fill=color, outline="#D2691E", width=1
        )

        if on_cell_hover:
            bind <Enter> → show_tooltip(rewards[i], row, col)   # 用原始值，不用归一化后的
            bind <Leave> → hide_tooltip()
```

**关键点**：
- tooltip 显示**原始 reward 值**（便于用户看真实概率），不是归一化后的值
- 归一化只影响颜色，不影响文字

### 6.2 HeatmapLegend._generate_gradient_colors

```
function _generate_gradient_colors():
    colors = []
    for i in 0..n_bands-1:
        v = i / (n_bands - 1)                 # [0, 1] 等距
        r, g, b = value_to_rgb(v, color_stops)
        colors.append(_rgb_float_to_hex(r, g, b))
    return colors
```

替换原"相邻 BASE_COLORS 之间手工 5 档插值"的双层循环。

---

## 7. 测试用例清单

### 7.1 tests/gui/test_heatmap.py 调整

**必删（语义已变）**：
- `test_get_reward_color_zero` / `_high` / `_low` / `_max` / `_min` / `_middle` / `_out_of_range`
  - 原因：`_get_reward_color` 方法已废除
- `test_primary_colors_defined`：`PRIMARY_COLORS` 已移除
- `test_reward_range_constants`：`REWARD_MIN` / `REWARD_MAX` 已移除
- `test_colors_full_defined` / `test_base_colors_defined`：Legend 常量已移除

**新增**：
| 编号 | 用例 | 断言 |
|------|------|------|
| G1 | `test_default_mode_is_log` | `HeatmapCanvas(...)` 创建后 `heatmap_mode == "log"` |
| G2 | `test_custom_mode_accepted` | `HeatmapCanvas(..., heatmap_mode="max")` 不报错 |
| G3 | `test_invalid_mode_raises` | `draw_heatmap` 时内部 normalize_rewards 对非法 mode 抛 ValueError |
| G4 | `test_default_color_stops_is_4_colors` | 默认 `color_stops` 长度 == 4 |
| G5 | `test_custom_color_stops_accepted` | 自定义 9 色也能工作（向后兼容留口） |
| G6 | `test_rgb_float_to_hex_format` | `_rgb_float_to_hex(1.0, 0.0, 0.0) == "#ff0000"` |
| G7 | `test_rgb_float_to_hex_clamp` | 输入在 [0,1] 内都能输出 7 字符 hex |
| G8 | `test_draw_heatmap_225_softmax_log_mode` | log 模式下 softmax 输入能绘制成功（smoke） |
| G9 | `test_legend_default_bands_is_24` | `_generate_gradient_colors()` 长度 == 24 |
| G10 | `test_legend_custom_bands` | `HeatmapLegend(parent, n_bands=16)` → 16 档 |
| G11 | `test_legend_first_color_is_white` | 第一档 == `#ffffff`（4 色板起点） |
| G12 | `test_legend_last_color_is_red` | 最后一档 == `#ff0000`（4 色板终点） |

**保留不动**：
- `test_init` / `test_init_custom_size` / `test_clear`
- `test_draw_heatmap_valid` / `_invalid_length` / `_with_values` / `_with_softmax_values`
- tooltip 相关全部（`test_create_tooltip` / `test_hide_tooltip` / `test_hide_tooltip_when_none` / `test_show_tooltip_creates_window`）
- `TestHeatmapLegend.test_init`

### 7.2 tests/gui/test_replay_viewer.py
- **无需改动**。现有 `test_draw_heatmaps` 等只验证"绘制不抛异常 + 调用链通"，对色值不做断言

---

## 8. 常量/周边清理

### 8.1 `gui/components/constants.py`
- `COLORS["heatmap_zero"] = "#F5DEB3"`：核查是否还有使用方；无使用则删除
- `COLORS["heatmap_outline"] = "#D2691E"`：仍在 `draw_heatmap` 中使用，保留

**核查方法**：全仓库 grep `heatmap_zero`，若无外部引用则清理。

### 8.2 README.md
- 若 README 描述了"9 色渐变"等具体色板信息，需同步更新（上一轮已把 Sample 可视化一节重写过；GUI 热力图在 README 中若有独立描述也需更新）

---

## 9. 风险与回滚

| 风险 | 等级 | 缓解 |
|------|------|------|
| 视觉突变导致用户不适应 | 中 | 本设计已定为"正式切换"，不做灰度；如反馈不佳通过 `git revert` 回滚本次 commit |
| 现有 test_heatmap.py 颜色断言全挂 | 低 | 设计文档明确列出删除/新增清单，按单一 commit 同步替换 |
| replay_viewer 构造处未传 mode | 低 | 默认 log 已经是最佳值，无需调整；若需要再做一轮最小补丁 |
| `_get_reward_color` 被其他模块使用 | 低 | 全仓 grep 已确认仅 heatmap.py 内部调用，可安全移除 |

**回滚策略**：本次改动集中在 `gui/components/heatmap.py` + 测试，单 commit 可 `git revert`。

---

## 10. 实施步骤（TDD Phase）

| Phase | 内容 | 产出 |
|-------|------|------|
| 1 | 调整 `tests/gui/test_heatmap.py`：删除废弃用例 + 添加 G1-G12 | 测试红灯 |
| 2 | 重写 `gui/components/heatmap.py`：HeatmapCanvas + HeatmapLegend + `_rgb_float_to_hex` | G1-G12 通过 |
| 3 | 核查 `heatmap_zero` 是否仍被引用，按需清理 `constants.py` | constants.py 更新 |
| 4 | 全量测试 `python -m pytest tests/` | 全绿 |
| 5 | 更新 README（如需要） | README 更新 |
| 6 | Search subagent 独立 review | review 报告 |
| 7 | 按 review 结果处置 + git commit + push | 本次任务闭环 |

---

## 11. 非目标（明确不做）

1. ❌ 不改动 `visualization/heatmap_color.py`（已稳定）
2. ❌ 不在 tk Canvas 上实现真正的 alpha blending（收益小、复杂度高）
3. ❌ 不增加"归一化模式下拉框"等 UI 控件（后续需求再议）
4. ❌ 不改动 AI 侧 `sample_visualizer.py`（上一任务已交付）
5. ❌ 不改 `replay_viewer.py` 的布局逻辑（只在必要时传参）

---

## 12. 决策记录

| 编号 | 决策点 | 结论 | 理由 |
|------|--------|------|------|
| D1 | 色板 | 统一为新 4 色（白→黄→橙→红） | 信息密度高；与 sample_visualizer 一致；消除色带重复 |
| D2 | 归一化 | 参数可切换，默认 log | 改善 softmax 极端分布的显示；向后兼容 |
| D3 | Legend | 同步改造为新色板 | 避免标尺与热力图视觉不一致误导用户 |
| D4 | alpha | 不处理（tk 不支持） | 收益小，实现代价高 |
| D5 | Legend 档数 | 24（3 段 × 8 档） | 4 色 3 段下 24 档已足够平滑；原 40 档是为 9 色 8 段设计 |
| D6 | 适配函数位置 | `gui/components/heatmap.py` 内 | 不污染 visualization 的无 UI 依赖契约 |
| D7 | tooltip 文字 | 显示原始 reward，不显示归一化值 | 用户需要真实概率 |

---

## 附录 A：与 sample_visualizer_redesign.md 的关系

本文档是 `sample_visualizer_redesign.md` §1.1 中"GUI 那套热力图重构时可直接 import 复用"这一承诺的落地方案。`visualization/heatmap_color.py` 的设计（无 UI 依赖、色板可配置、归一化可配置）是为支持此次复用而做的——本次是兑现合约。
