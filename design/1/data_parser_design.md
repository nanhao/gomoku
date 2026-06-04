# create_samples 验收工具 — 数据解析设计

## 1. 概述

data_parser 模块负责读取 create_samples.py 产出的中间产物并解析为前端可用的结构化数据。主要职责：

1. 扫描 round_dir，解析每局 record.json → 对局摘要列表
2. 解析单局 record.json + 匹配样本 → 步数据列表
3. 处理各类边界情况（字段缺失、格式兼容、数据损坏）

## 2. 数据源格式

### 2.1 record.json 结构

路径：`{round_dir}/{game_id}/record.json`

```json
{
  "history": [
    {
      "step_id": "a3f2c1e0-7b8d-4f9a-b5e6-2c1d0f3a4b5e",
      "input": [226-dim],
      "output": [225-dim],
      "position": [row, col],
      "strategy": "sampling",
      "temperature": 1.0,
      "top_k": null,
      "sampling_probs": [225-dim]
    },
    ...
  ],
  "winner": 1,
  "end_board": [226-dim]
}
```

**字段说明**：

| 字段 | 类型 | 必存在 | 说明 |
|------|------|--------|------|
| history | array | 是 | 每步记录列表 |
| history[].step_id | string | 否 | 全局唯一标识，UUID4 格式；旧格式可能缺失 |
| history[].input | integer[226] | 是 | 225维棋盘 + 1维当前落子方 |
| history[].output | float[225] | 是 | 模型 softmax 输出概率 |
| history[].position | integer[2] | 是 | 落子坐标 [row, col] |
| history[].strategy | string | 否 | "sampling" / "argmax"；旧格式可能缺失 |
| history[].sampling_probs | float[225] | 否 | 仅 strategy="sampling" 时存在 |
| history[].temperature | float | 否 | 仅 strategy="sampling" 时存在 |
| winner | integer | 是 | 1=黑胜, 2=白胜, 3=平局, 0=未结束 |
| end_board | integer[226] | 是 | 最终棋盘状态 |

### 2.2 样本输出文件结构

路径：用户通过 `--output` 指定的路径

```json
[
  {
    "step_id": "a3f2c1e0-7b8d-4f9a-b5e6-2c1d0f3a4b5e",
    "input": [226-dim],
    "output": [225-dim],
    "mask": [225-dim]
  },
  ...
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| step_id | string | 全局唯一标识，与 record.json 对应步一致；旧格式可能缺失 |
| input | integer[226] | 225维棋盘 + 1维当前落子方 |
| output | float[225] | 样本输出分布 |
| mask | integer[225] | 样本 mask |

## 3. 接口定义

### 3.1 scan_round_dir

```python
def scan_round_dir(round_dir: str) -> list[dict]:
    """扫描 round_dir，返回对局摘要列表

    Args:
        round_dir: 轮次目录路径

    Returns:
        对局摘要列表，每项包含 game_id, winner, temp_side 等。
        损坏的 record.json 跳过不报错。

    Raises:
        FileNotFoundError: round_dir 不存在
    """
```

### 3.2 parse_game_steps

```python
def parse_game_steps(round_dir: str, game_id: int,
                     sample_path: str | None = None) -> dict:
    """解析单局步数据

    Args:
        round_dir: 轮次目录路径
        game_id: 对局编号
        sample_path: 样本输出文件路径，None 时不匹配样本

    Returns:
        {game_id, winner, is_draw, steps: [StepData, ...]}

    Raises:
        FileNotFoundError: record.json 不存在
        json.JSONDecodeError: record.json 格式损坏
    """
```

### 3.3 match_samples

```python
def match_samples(history: list[dict],
                  samples: list[dict]) -> dict[int, list[float]]:
    """将样本匹配到 history 各步

    匹配算法：用 step_id 查找，替代 input 向量比对。
    旧格式数据（无 step_id 字段）自动跳过，sample_output 为 null。

    Args:
        history: record.json 的 history 列表
        samples: 样本输出列表

    Returns:
        {step_index: sample_output} 映射。
        未匹配的步不在映射中，由调用方视为 null。
    """
```

### 3.4 infer_temp_side

```python
def infer_temp_side(history: list[dict]) -> str:
    """从 history 的 strategy 字段推断该局的采样方

    Args:
        history: record.json 的 history 列表

    Returns:
        "both" / "black" / "white" / "none"
    """
```

## 4. 核心算法伪代码

### 4.1 scan_round_dir

```
function scan_round_dir(round_dir):
    games = []
    for entry in sorted(os.listdir(round_dir), key=int):
        record_path = round_dir / entry / "record.json"
        if not record_path.exists(): continue

        try:
            record = json.load(record_path)
        except JSONDecodeError:
            continue  # 跳过损坏文件

        winner = record["winner"]
        history = record.get("history", [])

        temp_side = infer_temp_side(history)

        games.append({
            game_id: int(entry),
            winner: winner,
            winner_display: {1: "黑胜", 2: "白胜", 3: "平局"}.get(winner, "未知"),
            temp_side: temp_side,
            temp_side_display: temp_side_display_map[temp_side],
            total_steps: len(history),
            is_draw: winner == 3
        })

    return games
```

### 4.2 match_samples

```
function match_samples(history, samples):
    # 构建 step_id → sample_output 映射
    # D4 增强产生 8 个同 step_id 变体，取第一个（恒等变换 tid=0）
    sample_map = {}
    for sample in samples:
        sid = sample.get("step_id")
        if sid is not None and sid not in sample_map:
            sample_map[sid] = sample["output"]  # 仅首次写入，跳过后续 D4 变体

    # 用 step_id 匹配
    result = {}
    for i, step in enumerate(history):
        sid = step.get("step_id")
        if sid is not None and sid in sample_map:
            result[i] = sample_map[sid]

    return result
```

### 4.3 infer_temp_side

```
function infer_temp_side(history):
    black_sampling = False
    white_sampling = False

    for i, step in enumerate(history):
        if step.get("strategy") == "sampling":
            player = step["input"][225]  # 第226维为当前落子方
            if player == 1: black_sampling = True
            if player == 2: white_sampling = True

    if black_sampling and white_sampling: return "both"
    if black_sampling: return "black"
    if white_sampling: return "white"
    return "none"
```

### 4.4 parse_game_steps

```
function parse_game_steps(round_dir, game_id, sample_path):
    record = json.load(f"{round_dir}/{game_id}/record.json")

    # 读取样本文件（如果存在）
    samples = []
    if sample_path and os.path.exists(sample_path):
        samples = json.load(sample_path)

    # 匹配样本
    sample_map = match_samples(record["history"], samples)

    # 构建步数据
    steps = []
    for i, step in enumerate(record["history"]):
        board = step["input"][:225]
        player = step["input"][225]
        move_pos = step["position"]

        steps.append({
            step_index: i,
            player: player,
            move_position: move_pos,
            board: board,
            raw_output: step["output"],
            sampling_probs: step.get("sampling_probs"),  # None if absent
            strategy: step.get("strategy"),               # None if absent
            sample_output: sample_map.get(i),             # None if unmatched
        })

    return {
        game_id: game_id,
        winner: record["winner"],
        winner_display: ...,
        is_draw: record["winner"] == 3,
        steps: steps
    }
```

## 5. 边界情况处理

| # | 场景 | 处理策略 |
|---|------|----------|
| E1 | record.json 不存在 | scan_round_dir 跳过该目录；parse_game_steps 抛 FileNotFoundError |
| E2 | record.json JSON 解析失败 | scan_round_dir 跳过该局；parse_game_steps 抛 JSONDecodeError |
| E3 | history 为空列表 | 该局 total_steps=0，返回空 steps 列表 |
| E4 | strategy 字段缺失（旧格式） | 视为 null，前端 Row 2 按 argmax 占位图处理 |
| E5 | sampling_probs 字段缺失 | 视为 null，同 E4 |
| E6 | 样本输出文件不存在 | 所有步 sample_output=null，Row 3 全占位图 |
| E7 | 样本输出文件为空列表 | 同 E6 |
| E8 | 平局对局无样本 | SampleCreator 跳过平局，所有步 sample_output=null |
| E9 | winner=0（未结束） | winner_display="未知"，is_draw=false |
| E10 | step_id 字段缺失（旧格式） | step.get("step_id")=None → 跳过匹配，该步 sample_output=null |
| E11 | winner_path 多个样本同 step_id | D4 增强产生 8 个同 step_id 变体，sample_map 仅首次写入（恒等变换 tid=0） |
| E12 | round_dir 差集为空 | 后端返回错误，前端显示"无数据" |

## 6. 格式兼容性

### 6.1 旧格式 record.json（无温度采样字段）

```json
{
  "history": [
    {"input": [...], "output": [...], "position": [r, c]}
  ],
  "winner": 1,
  "end_board": [...]
}
```

- `strategy` 字段缺失 → 视为 null → 前端 Row 2 全占位图
- `sampling_probs` 字段缺失 → 视为 null → 同上
- `step_id` 字段缺失 → 旧格式数据，样本匹配跳过，Row 3 全占位图

### 6.2 winning_move 模式样本

- 每局最多 1 个样本（取最后一步）
- 匹配后仅最后一步 sample_output 非 null
- Row 3：仅最后一步有热力图，其余为"无样本"占位图

### 6.3 winner_path 模式样本

- 每局多个样本（获胜方每步各一个）
- 黑胜：匹配偶数索引步（0, 2, 4, ...）
- 白胜：匹配奇数索引步（1, 3, 5, ...）
- Row 3：仅获胜方步有热力图，其余为"无样本"占位图

## 7. 测试用例表

| # | 测试点 | 用例说明 |
|---|--------|---------|
| T1 | 正常扫描 round_dir | 返回正确 game_id、winner、temp_side |
| T2 | record.json 损坏 | scan_round_dir 跳过，不抛异常 |
| T3 | 空 history | 返回 total_steps=0, steps=[] |
| T4 | 正常样本匹配 | winning_move 仅最后步匹配成功 |
| T5 | winner_path 样本匹配 | 仅获胜方步匹配成功 |
| T6 | 样本文件不存在 | 所有步 sample_output=null |
| T7 | 平局对局 | 无样本匹配，sample_output 全 null |
| T8 | 旧格式无 strategy | strategy=null → 前端 Row 2 占位图 |
| T9 | infer_temp_side 双方 | black+white 均有 sampling → "both" |
| T10 | infer_temp_side 全 argmax | 无 sampling → "none" |
| T11 | step_id 匹配 | step_id 查找，O(1) 无误匹配；旧格式无 step_id → 全 null |
