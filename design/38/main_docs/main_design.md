# MCTS Self-Play: Interrupt-Safe + Periodic Save

## 1. Overview

优化 `src/ai/scripts/mcts_self_play.py` 的样本保存策略，解决当前"所有对局结束后才一次性写入 `sample.json`"导致的中断丢数据问题。

**核心改动：**
- Ctrl+C 时保存已累积的样本数据再退出
- 每 50 局自动保存一次，降低数据丢失风险
- 启动时加载已有 `sample.json`，新样本与之合并后写入

## 2. Current vs New Behavior

| 场景 | 当前行为 | 新行为 |
|------|---------|--------|
| 正常运行结束 | 一次性写入 `sample.json` | 不变（最终写入） |
| 运行中 Ctrl+C | 数据全部丢失 | 已完成的局样本保存到 `sample.json` |
| 运行中崩溃 | 数据全部丢失 | 最多丢失最近 49 局（每 50 局自动保存） |
| 重复使用同一输出目录 | 覆盖旧数据 | 加载旧数据，新样本合并后写入 |

## 3. Design

### 3.1 Startup: Load Existing Data

`run_self_play()` 启动时检查输出目录下是否存在 `sample.json`：
- 存在 → 读取其 JSON 数组，作为 `all_samples` 初始值
- 不存在 → `all_samples` 初始为空列表 `[]`

### 3.2 Signal Handler (Ctrl+C)

注册 `SIGINT` 信号处理器：
- 触发时调用保存函数将当前 `all_samples` 写入 `sample.json`
- 同时更新 `summary.json`
- 输出提示信息后正常退出（`sys.exit(0)`）

信号处理器为轻量函数，仅设置标志位并触发保存，避免在信号上下文中执行复杂逻辑。

### 3.3 Periodic Save (Every 50 Games)

每完成一局对局后检查累计局数：
- 若 `completed_games % 50 == 0` → 触发保存
- 保存内容：当前 `all_samples`（含已加载的旧数据 + 新完成的对局）
- 同时更新 `summary.json`

### 3.4 Atomic Write

所有文件写入采用"先写临时文件，再原子重命名"策略：
- 写入 `sample.json.tmp`
- `os.replace()` 重命名为 `sample.json`
- 防止写入过程中崩溃导致文件损坏

### 3.5 Summary File

`summary.json` 在每次保存时同步更新，记录：
- `model`: 模型路径
- `games`: 总对局数（配置值）
- `total_samples`: 当前已保存样本数
- `completed_games`: 当前已完成局数
- `hyperparams`: 超参数

### 3.6 Save Function (Internal)

抽取保存逻辑为独立函数，供周期保存和信号处理复用：

```
save_progress(all_samples, output_root, args, completed_games)
```

职责：
1. 将 `all_samples` 原子写入 `{output_root}/sample.json`
2. 将汇总信息原子写入 `{output_root}/summary.json`
3. 输出日志：已保存局数、样本数

## 4. Interaction Flow

参见 [self_play_flow.puml](self_play_flow.puml) — 包含正常流程与中断流程。

## 5. Interface Changes

无 CLI 接口变更。详见 [interface_changes.md](interface_changes.md)。

## 6. Affected Code

- `src/ai/scripts/mcts_self_play.py` — `run_self_play()` 函数
