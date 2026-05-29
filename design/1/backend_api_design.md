# create_samples 验收工具 — 后端 API 设计

## 1. API 总览

| # | Method | Endpoint | 作用 | 核心动作 |
|---|--------|----------|------|----------|
| 1 | GET | `/api/create-samples/models` | 获取可用模型列表 | 子进程调用 registry，解析输出 |
| 2 | POST | `/api/create-samples/execute` | 执行 create_samples.py（同步阻塞） | 构造 CLI → Popen → communicate() → 返回全部结果 |
| 3 | GET | `/api/create-samples/temp-info` | 获取温度采样信息 | 从提交参数中提取 |
| 4 | GET | `/api/create-samples/games` | 获取对局列表 | 扫描 round_dir，解析各局 record.json |
| 5 | GET | `/api/create-samples/games/<game_id>/steps` | 获取某局步数据 | 解析 record.json + 匹配样本 |

> 仅 5 个端点。POST /api/create-samples/execute 同步阻塞等待子进程结束，一次性返回完整结果（含输出文本），无需轮询或额外状态查询。

---

## 2. 各端点详细定义

### 2.1 GET /api/create-samples/models

**作用**：获取项目当前可用的 AI 模型列表，供前端下拉框使用。

**核心动作**：
1. 子进程执行 `python -c "from ai.registry import list_models_with_full_name; print(list_models_with_full_name())"`
2. 解析输出为 JSON 列表
3. 返回模型名列表

**参数**：无

**Response**：

```json
{
  "models": ["random", "fc_v1", "cnn_v1", "cnn_v2", "cnn_v3.1", "cnn_v3.2", "cnn_v3.3"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| models | string[] | 模型指定字符串列表，与 registry.list_models_with_full_name() 返回值一致 |

**错误处理**：

| 场景 | HTTP 状态码 | Response |
|------|-------------|----------|
| 子进程执行失败 | 500 | `{"error": "无法获取模型列表: {detail}"}` |

---

### 2.2 POST /api/create-samples/execute

**作用**：接收用户填写的参数，构造 CLI 命令，**同步阻塞**执行子进程，等子进程结束后一次性返回完整结果。

**核心动作**：
1. 记录 `ai/ai_data/` 下已有目录集合 `S_before`
2. 构造 CLI 命令字符串
3. 启动 `subprocess.Popen`（stdout=PIPE, stderr=STDOUT）
4. 调用 `process.communicate()` **阻塞等待**子进程结束，收集全部输出
5. 子进程结束后发现 round_dir
6. 一次性返回 status + command + output + returncode

**参数**（JSON Body）：

| 字段 | 类型 | 必填 | 说明 | CLI 映射 |
|------|------|------|------|----------|
| model | string | 是 | 模型指定 | `--model {model}` |
| mode | string | 否 | 样本模式，默认 "winning_move" | `--mode {mode}` |
| games | integer | 否 | 对局数，默认 100 | `--games {games}` |
| output | string | 是 | 输出文件路径 | `--output {output}` |
| no_augment | boolean | 否 | 禁用数据增强，默认 false | `--no-augment` (仅 true 时追加) |
| no_temp_sampling | boolean | 否 | 禁用温度采样，默认 false | `--no-temp-sampling` (仅 true 时追加) |
| temperature | number | 否 | 温度值 T，默认 1.0 | `--temperature {temperature}` |
| temp_moves | integer | 否 | 前 N 步采样，默认 12 | `--temp-moves {temp_moves}` |
| temp_side | string | 否 | 采样方，默认 "both" | `--temp-side {temp_side}` |
| temp_seed | integer\|null | 否 | 随机种子 | `--temp-seed {temp_seed}` (仅非 null 时追加) |

**参数校验**：零校验策略——前端后端均不校验参数合法性与必填，直接透传给 argparse。若 argparse 报错，错误信息包含在返回的 output 字段中。

**Response（成功）**：

```json
{
  "status": "completed",
  "command": "python ai/create_samples.py --model cnn_v1 --mode winning_move --games 10 --output ai/train_data/samples.json --temperature 1.0 --temp-moves 12 --temp-side both",
  "output": "[INFO] Loading model: cnn_v1 (gen=1)\n[INFO] Model loaded successfully\n[INFO] Running 10 games with winning_move mode...\nGame 1: Black wins (15 steps, 12 sampling steps)\nGame 2: White wins (23 steps, 12 sampling steps)\n...\n[INFO] All 10 games completed\n[INFO] Samples written to ai/train_data/samples.json",
  "returncode": 0
}
```

**Response（失败）**：

```json
{
  "status": "error",
  "command": "python ai/create_samples.py --model not_exist --output test.json",
  "output": "usage: create_samples.py [-h] ...\nerror: argument --model: invalid value 'not_exist'",
  "returncode": 2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | "completed"（returncode==0）/ "error"（returncode!=0） |
| command | string | 实际执行的完整 CLI 命令 |
| output | string | 子进程完整输出文本（stdout + stderr 合并） |
| returncode | integer | 子进程退出码 |

**前端对接要点**：
- 使用 `await fetch(...)` 发起请求，浏览器会自动等待响应
- 执行期间（可能数分钟）前端处于等待状态，按钮保持灰化
- 脚本输出区可显示"执行中..."提示
- 响应返回后：根据 status 展示 output 内容，然后请求温度信息和对局列表

**CLI 命令构造规则**：

```
python ai/create_samples.py \
  --model {model} \
  --mode {mode} \
  --games {games} \
  --output {output} \
  [--no-augment] \
  [--no-temp-sampling] \
  --temperature {temperature} \
  --temp-moves {temp_moves} \
  --temp-side {temp_side} \
  [--temp-seed {temp_seed}]
```

- 布尔开关型参数：仅当值为 true 时追加对应 flag
- 可选数值型参数（temp_seed）：仅当值非 null 时追加
- 必填参数：始终追加
- 工作目录：项目根目录 `d:/projects/gomoku`

**Flask 配置**：`app.run(threaded=True)`，确保阻塞请求不影响其他并发请求。

---

### 2.3 GET /api/create-samples/temp-info

**作用**：获取本次运行的温度采样全局配置信息。

**核心动作**：
1. 从上次 POST /api/create-samples/execute 提交的参数中提取温度配置
2. 返回格式化的温度信息

**参数**：无

**Response**：

```json
{
  "temperature": 1.0,
  "temp_moves": 12,
  "temp_side": "both",
  "no_temp_sampling": false,
  "temp_side_display": "双方"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| temperature | number | 温度值 T |
| temp_moves | integer | 采样步数阈值 |
| temp_side | string | 原始 temp_side 值 |
| no_temp_sampling | boolean | 是否禁用温度采样 |
| temp_side_display | string | 显示用中文："黑方" / "白方" / "双方" / "random" / "无(argmax)" |

**temp_side_display 映射规则**：

| no_temp_sampling | temp_side | display |
|------------------|-----------|---------|
| true | * | "无(argmax)" |
| false | "both" | "双方" |
| false | "black" | "黑方" |
| false | "white" | "白方" |
| false | "random" | "random" |

**错误处理**：

| 场景 | HTTP 状态码 | Response |
|------|-------------|----------|
| 无执行记录 | 404 | `{"error": "尚未执行脚本"}` |

---

### 2.4 GET /api/create-samples/games

**作用**：获取本次执行的所有对局基本信息列表。

**核心动作**：
1. 定位 round_dir
2. 遍历子目录 `1/`, `2/`, ..., 读取 `record.json`
3. 从每局 record.json 提取 winner、temp_side 信息
4. 返回对局列表

**参数**：无

**Response**：

```json
{
  "games": [
    {
      "game_id": 1,
      "winner": 1,
      "winner_display": "黑胜",
      "temp_side": "both",
      "temp_side_display": "双方采样",
      "total_steps": 15,
      "is_draw": false
    },
    {
      "game_id": 5,
      "winner": 3,
      "winner_display": "平局",
      "temp_side": "both",
      "temp_side_display": "双方采样",
      "total_steps": 50,
      "is_draw": true
    }
  ],
  "total": 10
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| games | array | 对局列表，按 game_id 升序 |
| games[].game_id | integer | 对局编号（从 1 开始） |
| games[].winner | integer | 胜方：1=黑胜, 2=白胜, 3=平局 |
| games[].winner_display | string | 显示用："黑胜" / "白胜" / "平局" |
| games[].temp_side | string | 该局实际 temp_side |
| games[].temp_side_display | string | 显示用："双方采样" / "黑方采样" / "白方采样" / "无(argmax)" |
| games[].total_steps | integer | 该局总步数 |
| games[].is_draw | boolean | 是否为平局 |
| total | integer | 对局总数 |

**temp_side_display 映射**（per-game，基于 record.json 中的 strategy 字段推断）：

| 推断条件 | display |
|----------|---------|
| 所有步均为 argmax | "无(argmax)" |
| 仅黑方步有 sampling | "黑方采样" |
| 仅白方步有 sampling | "白方采样" |
| 黑白方均有 sampling | "双方采样" |

**错误处理**：

| 场景 | HTTP 状态码 | Response |
|------|-------------|----------|
| 无执行记录 | 404 | `{"error": "尚未执行脚本"}` |
| round_dir 不存在 | 500 | `{"error": "数据目录不存在"}` |
| record.json 损坏 | — | 跳过该局，不影响其他局数据返回 |

---

### 2.5 GET /api/create-samples/games/\<game_id\>/steps

**作用**：获取指定对局的每步详细数据，供前端渲染三行热力图。

**核心动作**：
1. 读取 `{round_dir}/{game_id}/record.json`
2. 读取样本输出文件，匹配该局的样本
3. 组装每步的完整数据

**参数**：

| 参数 | 位置 | 类型 | 说明 |
|------|------|------|------|
| game_id | URL path | integer | 对局编号（从 1 开始） |

**Response**：

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
      "board": [0, 0, ..., 0],
      "raw_output": [0.001, 0.6, ..., 0.001],
      "sampling_probs": [0.002, 0.55, ..., 0.002],
      "strategy": "sampling",
      "sample_output": null
    },
    {
      "step_index": 1,
      "player": 2,
      "move_position": [7, 8],
      "board": [0, 0, ..., 1, ..., 0],
      "raw_output": [0.01, 0.3, ..., 0.01],
      "sampling_probs": [0.015, 0.28, ..., 0.015],
      "strategy": "sampling",
      "sample_output": null
    }
  ]
}
```

**steps[] 各字段说明**：

| 字段 | 类型 | 说明 | Row 映射 |
|------|------|------|----------|
| step_index | integer | 步序号（0-based） | — |
| player | integer | 当前落子方：1=黑, 2=白 | — |
| move_position | [int, int] | 落子坐标 [row, col] | — |
| board | integer[225] | 当前步棋盘状态（0=空, 1=黑, 2=白） | 棋子绘制 |
| raw_output | float[225] | 模型原始输出概率 | Row 1：原始输出 |
| sampling_probs | float[225]\|null | 温度处理后采样权重；null = 该步为 argmax | Row 2：温度分布 |
| strategy | string\|null | "sampling" / "argmax" / null(旧格式) | 判断 Row 2 是否占位 |
| sample_output | float[225]\|null | 匹配到的样本 output；null = 该步无样本 | Row 3：样本分布 |

**前端渲染逻辑**：

| Row | 条件 | 渲染 |
|-----|------|------|
| Row 1 (原始输出) | raw_output 始终存在 | 正常热力图 |
| Row 2 (温度分布) | sampling_probs != null | 正常热力图 |
| Row 2 (温度分布) | sampling_probs == null | 灰色占位图 + "argmax" |
| Row 3 (样本分布) | sample_output != null | 正常热力图 |
| Row 3 (样本分布) | sample_output == null | 灰色占位图 + "无样本" |

**样本匹配算法**（data_parser 模块实现）：

1. 读取样本输出文件 `[{step_id, input, output, mask}, ...]`
2. 构建 step_id → sample_output 映射
3. 遍历 record.json 的 history，用每步的 step_id 查找映射
4. 匹配成功的步：`sample_output = sample.output`
5. 未匹配的步：`sample_output = null`

> 详见 [step_id_matching_design.md](step_id_matching_design.md)

**平局对局处理**：
- winner=3 的对局，SampleCreator 会跳过（不生成样本）
- 因此所有步的 sample_output 均为 null
- 前端在标题区标注"平局无样本"，Row 3 全为占位图

**错误处理**：

| 场景 | HTTP 状态码 | Response |
|------|-------------|----------|
| 无执行记录 | 404 | `{"error": "尚未执行脚本"}` |
| game_id 不存在 | 404 | `{"error": "对局 {game_id} 不存在"}` |
| record.json 损坏 | 500 | `{"error": "对局 {game_id} 数据损坏"}` |

---

## 3. 公共数据结构

### 3.1 执行参数 (ExecuteParams)

```python
class ExecuteParams:
    model: str
    mode: str = "winning_move"
    games: int = 100
    output: str
    no_augment: bool = False
    no_temp_sampling: bool = False
    temperature: float = 1.0
    temp_moves: int = 12
    temp_side: str = "both"
    temp_seed: int | None = None
```

### 3.2 执行结果 (ExecuteResult)

```python
class ExecuteResult:
    status: str              # "completed" / "error"
    command: str             # 实际执行的 CLI 命令
    output: str              # 子进程完整输出文本 (stdout+stderr)
    returncode: int          # 子进程退出码
    round_dir: str | None    # 发现的 round_dir (仅 completed 时有值)
```

### 3.3 对局摘要 (GameSummary)

```python
class GameSummary:
    game_id: int
    winner: int              # 1=黑胜, 2=白胜, 3=平局
    winner_display: str      # "黑胜" / "白胜" / "平局"
    temp_side: str           # 实际采样方
    temp_side_display: str   # 显示用
    total_steps: int
    is_draw: bool
```

### 3.4 步数据 (StepData)

```python
class StepData:
    step_index: int
    player: int              # 1=黑, 2=白
    move_position: list[int] # [row, col]
    board: list[int]         # 225-dim
    raw_output: list[float]  # 225-dim
    sampling_probs: list[float] | None  # 225-dim or null
    strategy: str | None     # "sampling" / "argmax" / null
    sample_output: list[float] | None   # 225-dim or null
```

---

## 4. 同步阻塞执行机制

### 4.1 核心原理

POST /api/create-samples/execute 的处理函数**同步调用** `process.communicate()`，该函数会阻塞直到子进程结束。HTTP 响应在子进程结束后才返回给前端。

```python
@app.route("/api/create-samples/execute", methods=["POST"])
def execute():
    params = request.get_json()
    result = executor.execute(params)  # 阻塞在这里
    return jsonify(result)             # 子进程结束后才执行到这里
```

### 4.2 executor.execute 伪代码

```
function execute(params):
    S_before = scan ai_data_dir
    command = build_cli_command(params)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, cwd=project_root)
    output_bytes, _ = process.communicate()  # 阻塞等待
    output_text = output_bytes.decode("utf-8", errors="replace")

    if process.returncode == 0:
        round_dir = discover_round_dir(S_before)
        return {
            status: "completed",
            command: command,
            output: output_text,
            returncode: 0,
            round_dir: round_dir
        }
    else:
        return {
            status: "error",
            command: command,
            output: output_text,
            returncode: process.returncode
        }
```

### 4.3 Flask 配置

```python
app.run(host="127.0.0.1", port=5000, threaded=True)
```

- `threaded=True`：让 Flask 为每个请求分配独立线程，阻塞请求不影响其他并发请求
- 对于单用户本地工具，这完全足够

### 4.4 前端对接

```javascript
// 点击执行 → 按钮灰化 → 等 POST 返回 → 展示结果
async function onExecute() {
    disableButton();
    showOutputHint("执行中...");

    const resp = await fetch("/api/create-samples/execute", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(params)
    });
    const result = await resp.json();  // 子进程结束后才到这里

    enableButton();
    showOutput(result.output);

    if (result.status === "completed") {
        loadTempInfo();
        loadGames();
    }
}
```

---

## 5. 错误处理策略

### 5.1 总体原则

- 零参数校验：前端后端均不校验参数，由 argparse 统一处理
- 错误不弹窗：所有错误信息通过脚本输出区或 API response 正常展示
- 降级容错：单局 record.json 损坏不影响其他局数据返回

### 5.2 错误分类

| 错误源 | 表现 | 处理方式 |
|--------|------|----------|
| argparse 参数校验失败 | 子进程 stderr 输出错误信息 | POST /api/execute 返回 status="error"，output 包含错误文本 |
| 模型加载失败 | 子进程 stderr 输出异常 | 同上 |
| 子进程崩溃 | returncode != 0 | POST /api/execute 返回 status="error" |
| record.json 损坏 | JSON 解析失败 | 跳过该局，不影响其他局 |
| 样本文件不存在 | 文件读取失败 | 所有步 sample_output 为 null，Row 3 全占位 |
| round_dir 未发现 | 差集为空 | GET /api/games 返回 500 |

---

## 6. 测试用例表

| # | 测试点 | 所属端点 | 用例说明 |
|---|--------|---------|---------|
| T1 | 模型列表获取 | GET /api/create-samples/models | 返回非空列表，含 "random" |
| T2 | 模型列表子进程失败 | GET /api/create-samples/models | 返回 500 + error 字段 |
| T3 | 正常执行 | POST /api/create-samples/execute | 返回 status="completed" + output 非空 + returncode=0 |
| T4 | 执行失败 | POST /api/create-samples/execute | 返回 status="error" + output 含错误文本 + returncode!=0 |
| T5 | CLI 命令构造正确 | POST /api/create-samples/execute | 布尔开关仅 true 时追加 flag |
| T6 | 温度信息 | GET /api/create-samples/temp-info | 正确映射 temp_side_display |
| T7 | 无执行时温度信息 | GET /api/create-samples/temp-info | 返回 404 |
| T8 | 对局列表 | GET /api/create-samples/games | 返回正确数量和 winner_display |
| T9 | 平局对局标记 | GET /api/create-samples/games | is_draw=true, winner_display="平局" |
| T10 | 步数据完整 | GET /api/create-samples/games/1/steps | raw_output/sampling_probs/sample_output 齐全 |
| T11 | argmax 步无 sampling_probs | GET /api/create-samples/games/1/steps | sampling_probs=null → 前端渲染占位图 |
| T12 | 样本匹配正确 | GET /api/create-samples/games/1/steps | winning_move 仅最后步有 sample_output |
| T13 | 平局全占位 | GET /api/create-samples/games/5/steps | 所有步 sample_output=null |
| T14 | 不存在的 game_id | GET /api/create-samples/games/999/steps | 返回 404 |
| T15 | record.json 损坏 | GET /api/create-samples/games | 跳过该局，其他局正常返回 |
