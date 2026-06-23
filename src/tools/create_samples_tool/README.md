# create_samples 验收工具

基于 Flask 的本地 Web 工具，用于可视化验收 `create_samples.py` 生成的训练样本。
通过子进程调用业务脚本，零耦合（不 import 任何 `ai/` 包模块）。

## 启动

```bash
python -m src.tools.create_samples_tool.app
```

启动后浏览器访问 **http://127.0.0.1:5000**

> 依赖：Flask、requests（如未安装：`pip install flask requests`）

## 前端操作

页面分为左右两栏：

| 区域 | 说明 |
|------|------|
| 参数配置 | 填写模型、模式、对局数、输出路径等参数 |
| 脚本输出 | 展示 `create_samples.py` 的完整 stdout 输出 |
| 温度采样信息 | 展示当前温度 T、采样步数、采样方 |
| 对局列表 | 执行完成后显示所有对局，点击可查看热力图 |
| 热力图对比 | 三行热力图：原始输出 / 温度分布 / 样本分布 |

### 操作流程

1. **选择模型** — 下拉框自动加载可用模型（random / fc_v1 / cnn_v1 / cnn_v2 / cnn_v3.1 / cnn_v3.2 / cnn_v3.3）
2. **填写参数** — model 和 output 为必填项（output 标有红色 `*`）
3. **点击执行** — 按钮灰化等待，脚本运行完毕后自动恢复
4. **查看结果** — 对局列表自动渲染，默认选中第一局并加载热力图
5. **切换对局** — 点击列表中其他对局可切换热力图
6. **Hover 详情** — 鼠标悬停在棋盘格上可查看具体概率值

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model | 模型名称（必填） | — |
| games | 对局数 | 100 |
| output | 输出 `.json` 文件路径（必填） | — |
| 数据增强 | 勾选"禁用"则传 `--no-augment` | 不禁用 |
| temperature | 温度值 T | 1.0 |
| temp-moves | 前 N 步启用温度采样 | 12 |
| temp-side | 采样方：`random` / `both` / `black` / `white` | `random` |
| temp-seed | 随机种子（可选，便于复现） | — |

### 热力图说明

三行热力图分别对应：

| 行 | 数据来源 | 含义 |
|----|----------|------|
| **原始输出** | `step.output` | 模型原始 softmax 输出 |
| **温度分布** | `step.sampling_probs` | 温度采样后的概率分布（argmax 步显示为"argmax"占位） |
| **样本分布** | `sample.output` | 样本标签（仅获胜方变异步有样本，其他步显示为"无样本"占位） |

色板：白→黄→橙→红（与 `src/visualization/heatmap_color.py` 的 `DEFAULT_COLOR_STOPS` 一致）。
落子位以粗绿框标记（不画棋子），其余棋子正常渲染。

## API 端点

### GET /api/create-samples/models

获取可用模型列表。

**响应示例：**
```json
{
  "models": ["random", "fc_v1", "cnn_v1", "cnn_v2", "cnn_v3.1", "cnn_v3.2", "cnn_v3.3"]
}
```

> 内部通过子进程调用 `ai.registry.list_models_with_full_name()`，不 import ai 包。

### POST /api/create-samples/execute

同步执行 `create_samples.py`，阻塞等待完成后返回结果。

**请求体：**
```json
{
  "model": "cnn_v3.1",
  "games": 1,
  "output": "src/ai/train_data/test.json",
  "no_augment": false,
  "temperature": 1.0,
  "temp_moves": 12,
  "temp_side": "random",
  "temp_seed": null
}
```

**响应示例（成功）：**
```json
{
  "status": "completed",
  "command": "python src/ai/create_samples.py --model cnn_v3.1 ...",
  "output": "模型: cnn_v3.1\n对局数 (games): 1\n...",
  "returncode": 0,
  "round_dir": "src/ai/ai_data/cnn_v3.1_battle_3",
  "output_path": "src/ai/train_data/test.json",
  "games": [
    {"game_id": 1, "winner": 1, "winner_display": "黑胜", "temp_side": "both", "temp_side_display": "双方采样", "total_steps": 89, "is_draw": false}
  ]
}
```

**响应示例（失败）：**
```json
{
  "status": "error",
  "command": "python src/ai/create_samples.py ...",
  "output": "create_samples.py: error: ...",
  "returncode": 2,
  "round_dir": null,
  "output_path": null,
  "games": null
}
```

### GET /api/create-samples/games/{game_id}/steps

获取指定对局的步数据（含样本匹配）。

**Query 参数：**

| 参数 | 说明 |
|------|------|
| round_dir | 轮次数据目录路径（execute 返回的 round_dir） |
| sample_path | 样本输出文件路径（execute 返回的 output_path） |

**响应示例：**
```json
{
  "game_id": 1,
  "winner": 1,
  "winner_display": "黑胜",
  "is_draw": false,
  "steps": [
    {
      "step_index": 0,
      "player": 1,
      "move_position": [7, 7],
      "board": [0, 0, ..., 1],
      "raw_output": [0.01, 0.02, ..., 0.05],
      "sampling_probs": [0.01, 0.02, ..., 0.05],
      "strategy": "sampling",
      "sample_output": null
    }
  ]
}
```

> `board`、`raw_output`、`sampling_probs`、`sample_output` 均为 225 维数组。
> `sample_output` 仅在匹配到样本时非 null（winner_path 模式下仅获胜方有样本）。

## 架构

```
src/tools/create_samples_tool/
├── app.py           # Flask 应用（3 个 API 端点 + 主页路由）
├── executor.py      # 子进程同步执行 + CLI 命令构造 + round_dir 发现
├── data_parser.py   # record.json 解析 + step_id 样本匹配 + 采样方推断
├── static/
│   ├── index.html   # 主页面（五区域布局）
│   ├── style.css    # 样式
│   └── app.js       # 前端逻辑（fetch + Canvas 渲染 + hover tooltip）
└── README.md        # 本手册
```

### 零耦合原则

- 不 import 任何 `src/ai/` 包模块（避免触发 torch 等重依赖）
- 模型列表通过子进程 `python -c "..."` 调用 registry 获取
- 样本生成通过子进程调用 `src/ai/create_samples.py`
- 数据读取仅使用标准库 `json` + `os` 操作文件系统

### 样本匹配机制

使用 `step_id`（UUID4）进行样本到对局步骤的匹配，而非棋盘向量比对：
- 每步 `record.json` 的 `history[i].step_id` 与样本的 `step_id` 对应
- D4 数据增强产生的 8 个同 `step_id` 变体，仅取首次（恒等变换 tid=0）
- 旧格式数据（无 `step_id`）自动跳过样本匹配

### 子进程编码

Windows 下子进程默认以 GBK 编码输出中文，工具通过设置 `PYTHONIOENCODING=utf-8` 环境变量强制 UTF-8 输出，避免乱码。

## 测试

```bash
# 运行验收工具的全部单元测试（41 个）
python -m pytest tests/tools/ -v

# 端到端集成测试（需先启动 Flask 服务）
python temp/test_e2e.py
```
