# SampleVisualizer 样本可视化工具设计

## 背景

`create_samples.py` 生成的 winning_move 模式样本（input 226 维 + output 225 维 + mask 225 维）缺少直观的验证手段。现有 `SampleInspector` 和 `temp/viz_sample.py` 面向旧格式（概率分布 + 奖励热力图），不适用于 output 仅为单点取胜标记的新样本。

需要一个正式工具，抽样可视化样本：棋盘状态 + 取胜位置标记，供人工直观确认。

## 设计目标

1. 读取 `.json` 格式样本文件，随机抽取 N 个样本可视化
2. 每个样本绘制棋盘（黑白子）+ 取胜位置标记（红色星号）
3. 支持 CLI 参数控制抽样数量、指定样本索引、输出路径
4. 作为 `ai/` 下正式工具，含完整测试

## API 设计

### SampleVisualizer 类

```python
class SampleVisualizer:
    def __init__(self):
        """初始化可视化器（配置中文字体、matplotlib 后端）"""

    def visualize(self, samples: list, indices: list = None, count: int = 6,
                  output_path: str = None) -> str:
        """可视化样本列表

        Args:
            samples: 样本列表，每个元素含 input/output/mask
            indices: 指定样本索引列表（与 count 互斥，优先于 count）
            count: 随机抽样数量，默认 6
            output_path: 输出图片路径，默认 temp/sample_visualizer.png

        Returns:
            输出图片路径
        """
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | 样本文件路径（.json） | 必填 |
| `--count` | 随机抽样数量 | 6 |
| `--indices` | 指定样本索引（逗号分隔，如 `0,5,10`） | 无 |
| `--output` | 输出图片路径 | `temp/sample_visualizer.png` |

> `--indices` 与 `--count` 互斥，`--indices` 优先。

## 主流程伪代码

```
main():
    解析 CLI 参数
    加载样本文件 → samples
    确定抽样方式：
        若指定 --indices → 按索引取
        否则 → 随机抽样 count 个
    visualizer = SampleVisualizer()
    output = visualizer.visualize(samples, indices, count, output_path)
    print(f"已保存: {output}")

SampleVisualizer.visualize(samples, indices, count, output_path):
    确定要展示的样本列表 selected
    计算 grid 布局 (nrows, ncols)
    创建 figure + axes
    for each (sample, ax):
        解析 input → board(15x15), player
        解析 output → winning_pos
        绘制棋盘底色 + 网格线 + 星位点
        遍历 225 位：val=1 → 黑子, val=2 → 白子
        在取胜位置画红色星号标记
        设置标题（样本索引、玩家、已占位数）
    保存图片 → 返回路径
```

### 单个样本可视化细节

```
棋盘绘制：
    背景：#DEB887（棋盘色）
    网格线：15×15
    星位点：9 个标准位置
    黑子：黑色实心圆
    白子：白色实心圆 + 黑色描边

取胜位置标记：
    红色星号 (*) 覆盖在取胜坐标上
    大小醒目 (s=200)

标题格式：
    "Sample #{idx} | P={player} | {occ} stones | win=({r},{c})"
```

## 测试用例

| 用例 | 类型 | 说明 |
|------|------|------|
| test_visualize_random_sample | 单元 | 随机抽样 count 个，输出图片存在且尺寸正确 |
| test_visualize_specified_indices | 单元 | 指定 indices，图片中子图数与指定索引数一致 |
| test_visualize_single_sample | 单元 | 仅 1 个样本，正确生成 1×1 布局 |
| test_empty_samples | 单元 | 空样本列表，抛出 ValueError |
| test_cli_basic | 集成 | CLI 基本调用，输出文件存在 |
| test_cli_with_indices | 集成 | --indices 参数，输出文件存在 |
| test_cli_missing_sample | 集成 | 不存在的文件，退出码非 0 |

## 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `ai/sample_visualizer.py` | 新增 | SampleVisualizer 类 + CLI 入口 |
| `tests/ai/test_sample_visualizer.py` | 新增 | 单元测试 + 集成测试 |
| `README.md` | 修改 | 命令速查表增加 sample_visualizer 用法 |
