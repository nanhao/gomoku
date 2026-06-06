# 主设计文档：cnn_v3 迭代进化训练脚本

| 字段 | 值 |
|------|-----|
| intention_id | 6 |
| PRD | requirements/6/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

本次变更为**纯新增**：新建一个编排脚本 `ai/iterative_train.py`，将已有的 create_samples.py、train_main.py、battle.py 串联为自动化迭代闭环。不修改任何已有代码。

脚本作为纯编排层，不引入任何对弈、训练、评估的业务逻辑。

交互图见 [iterative_flow.puml](file:///d:/projects/gomoku/design/6/main_docs/iterative_flow.puml)。

---

## 2. 模块设计

### 2.1 新增文件

| 文件 | 类别 | 说明 |
|------|------|------|
| `ai/iterative_train.py` | 源码 | 迭代进化编排脚本 |

### 2.2 核心类：IterativeTrainer

对外暴露的公共接口：

```
IterativeTrainer()
    ├── __init__() → 校验前置条件（gen4 权重存在、锁文件不存在）
    └── run() -> dict → 执行迭代循环，返回统计字典
```

类职责：
- 管理模型文件操作（copy gen4→gen5、copy+remove gen5→gen4 晋升）
- 通过子进程调用三个已有脚本
- 解析 battle.py 输出提取胜率
- 维护迭代状态（历史记录、当前轮次）
- 输出格式化日志和最终统计报告

### 2.3 模块入口

```
main() → CLI 入口，创建 IterativeTrainer 并调用 run()
```

无命令行参数（v1.0 全部硬编码）。

---

## 3. 硬编码常量

| 常量 | 值 | 说明 |
|------|-----|------|
| BASE_MODEL_NAME | `"cnn_v3"` | 基础模型名 |
| PARENT_GEN | `4` | 父代编号 |
| CHILD_GEN | `5` | 子代编号 |
| MODELS_DIR | `ai/models/cnn_v3` | 权重目录（相对于项目根） |
| SAMPLE_OUTPUT_DIR | `ai/merge_samples/v4` | 样本输出目录 |
| SAMPLE_GAMES | `500` | 目标样本局数 |
| BATTLE_GAMES | `100` | 对战局数 |
| TRAIN_EPOCHS | `1` | 训练 epoch 数 |
| TRAIN_BATCH_SIZE | `4096` | 训练 batch size |
| WIN_RATE_THRESHOLD | `50.0` | 晋升胜率阈值（%，浮点数，与 battle.py 输出精度 `.1f` 对齐） |
| PYTHON_EXECUTABLE | `sys.executable` | 子进程 Python 解释器路径，确保与主脚本使用同一解释器 |

---

## 4. 关键流程

### 4.1 主循环流程

> 步骤编号与 [iterative_flow.puml](file:///d:/projects/gomoku/design/6/main_docs/iterative_flow.puml) 对应：
> 步骤 a→轮次标识（打印轮次头）、b→步骤2（模型初始化）、c→步骤3（样本生成）、d→步骤4（模型训练）、e→步骤5（对战评估）、f-i→步骤6（胜率判定）。

```
1. 校验 gen4/cnn_v3.pth 存在
2. round_num = 1
3. 循环：
   a. 打印 "========== 第 {round_num} 轨迭代开始 =========="
   b. 复制 gen4/cnn_v3.pth → gen5/cnn_v3.pth（若 gen5/ 不存在则创建）
   c. 调用 create_samples.py（生成 path{round_num}.json）
      → 完成后打印样本文件路径及实际收集的样本局数
   d. 校验 path{round_num}.json：读取文件，解析 JSON 数组，检查样本数量
      → 若为 0 条：打印 "空样本"，抛 RuntimeError 终止
   e. 调用 train_main.py（训练 gen5）
      → 完成后打印训练状态
   f. 调用 battle.py（gen4 vs gen5，100局）
      → 完成后打印 cnn_v3.4 胜场、cnn_v3.5 胜场、平局数、cnn_v3.5 胜率
   g. 从 stdout 解析 cnn_v3.5 胜率
   h. 记录本轮统计到累加器
   i. 若胜率 > 50.0%：
        - 晋升：copy2 gen5/cnn_v3.pth → gen4/cnn_v3.pth
        - 确认 copy2 成功后删除 gen5/cnn_v3.pth
        - round_num++
        - 打印 "第 {round_num-1} 轨结果: cnn_v3.5 胜率 = {rate}% → 晋升"
        - 继续循环
   j. 否则：
        - 打印 "第 {round_num} 轨结果: cnn_v3.5 胜率 = {rate}% → 停止"
        - 打印最终统计报告
        - 退出循环
```

### 4.1.1 逐轮明细行精确格式

最终统计报告中逐轮明细行的格式与 PRD §3.4 对齐：

**晋升轮**：
```
第{round_num}轮: cnn_v3.4 胜 {gen4_wins} | cnn_v3.5 胜 {gen5_wins} | 平局 {draws} | cnn_v3.5 胜率 {rate}% → 晋升
```

**停止轮**：
```
第{round_num}轮: cnn_v3.4 胜 {gen4_wins} | cnn_v3.5 胜 {gen5_wins} | 平局 {draws} | cnn_v3.5 胜率 {rate}% → 停止
```

**轮次结束时单行日志**（与 PRD §3.3 对齐）：
```
第 {round_num} 轮结果: cnn_v3.5 胜率 = {rate}% → {晋升|停止}
```

### 4.2 子进程调用

三个子进程调用均通过 `subprocess.run()` 执行，关键参数统一如下：

- **Python 解释器**：使用 `sys.executable`（常量 `PYTHON_EXECUTABLE`），确保子进程与主脚本使用同一解释器
- **工作目录**：`cwd=self.project_root`，确保所有相对路径（如 `ai/create_samples.py`、`ai/merge_samples/v4/...`）正确解析
- **输出捕获**：`capture_output=True`（同时捕获 stdout 和 stderr）
- **编码**：设置环境变量 `PYTHONIOENCODING=utf-8`，确保 battle.py 含中文的输出正确解码
- **退出码检查**：任一子进程 `returncode ≠ 0` 即抛 `RuntimeError` 终止

**create_samples.py 调用参数**：
```
{sys.executable} ai/create_samples.py
--model cnn_v3.4 --mode winner_path --sample_games 500
--output ai/merge_samples/v4/path{round_num}.json --temp-side random
```

**train_main.py 调用参数**：
```
{sys.executable} ai/train_main.py
--model cnn_v3.5 --train-from-dir ai/merge_samples/v4/path{round_num}.json
--epochs 1 --batch-size 4096 --shuffle
```

**battle.py 调用参数**：
```
{sys.executable} ai/battle.py
--ai1 cnn_v3.4 --ai2 cnn_v3.5 --games 100
```

### 4.3 胜率解析

从 battle.py 的 stdout 中匹配模式：
```
cnn_v3\.5 胜: \d+ 局 \(([\d.]+)%\)
```
提取捕获组 `{rate}` 作为胜率（浮点数，精度一位小数，与 battle.py 输出 `.1f` 格式对齐）。解析失败时抛异常终止。

battle.py 的 stdout 同时包含每局逐行结果和最终统计行，解析时以最终统计行（`cnn_v3.5 胜: ... (...)`）为准。需注意：stdout 中同时存在 `cnn_v3.4 胜: ...` 和 `cnn_v3.5 胜: ...` 两行，正则必须精确匹配 `cnn_v3.5` 行，避免误匹配 `cnn_v3.4`。

胜率与阈值比较：`rate > 50.0`（浮点比较），battle.py 输出精度为 `.1f`，不存在浮点精度截断问题。

### 4.4 文件操作

| 操作 | 时机 | 说明 |
|------|------|------|
| `shutil.copy2` | 每轮开始时 | gen4/cnn_v3.pth → gen5/cnn_v3.pth，保留元数据 |
| `os.makedirs` | copy 前 | 若 gen5/ 目录不存在则创建 |
| `shutil.copy2` + `os.remove` | 晋升时 | 先 copy2 gen5→gen4（覆盖），确认 copy 成功后 remove gen5。比 move 更安全：copy 失败时两文件均保留 |
| `os.makedirs` | create_samples 前 | 若 v4/ 目录不存在则创建 |
| 读取 JSON 文件 | train_main 前 | 读取 path{round_num}.json，解析为 JSON 数组，校验样本数量 |

---

## 5. 数据流

```
gen4/cnn_v3.pth ──[copy2]──→ gen5/cnn_v3.pth
                                  │
gen4 model ──[create_samples]──→ path{round_num}.json
                                  │
path{round_num}.json ──[校验非空]──┬─ 空 → RuntimeError 终止
                                  │
path{round_num}.json ──[train_main]──→ gen5/cnn_v3.pth (更新)
                                  │
gen4 vs gen5 ──[battle]────────→ stdout (胜率统计)
                                  │
                          ┌─ >50.0% → gen5 ──[copy2→gen4]──→ gen4 ──[remove gen5] (晋升，round_num++)
                          │
                          └─ ≤50.0% → 停止 + 报告
```

---

## 6. 统计数据结构

`run()` 返回的 dict 结构：

| 键 | 类型 | 说明 |
|----|------|------|
| total_iterations | int | 成功晋升的轮次数 |
| final_round | int | 停止时的轮次编号 |
| rounds | list[dict] | 每轮明细（见下表） |
| stop_reason | str | 停止原因文本，格式："第 {round_num} 轮 cnn_v3.5 胜率 {rate}% ≤ 50.0%，停止迭代" |
| final_model_path | str | 最终 gen4 权重文件的**绝对路径** |

每轮明细 dict：

| 键 | 类型 | 说明 |
|----|------|------|
| round_num | int | 轮次编号 |
| gen4_wins | int | 父代胜场 |
| gen5_wins | int | 子代胜场 |
| draws | int | 平局数 |
| gen5_win_rate | float | 子代胜率（%） |
| promoted | bool | 是否晋升 |

---

## 7. 错误处理策略

| 场景 | 处理 |
|------|------|
| gen4/cnn_v3.pth 不存在 | `__init__()` 中检测，立即抛 `FileNotFoundError` 退出 |
| gen5/ 目录不存在 | copy 前自动创建 |
| v4/ 目录不存在 | create_samples 前自动创建 |
| 子进程 returncode ≠ 0 | 打印 stderr，抛 `RuntimeError` 终止，不做重试，不设超时 |
| 胜率解析失败（stdout 格式异常） | 打印原始输出，抛 `ValueError` 终止 |
| 样本为空（0 条） | train_main 调用前读取 path{round_num}.json，解析 JSON 数组并校验长度；为 0 条时打印 "空样本" 并抛 `RuntimeError`；文件不存在或 JSON 解析失败时同样抛 `RuntimeError` |
| shutil.copy2 失败（如磁盘满） | 抛 `IOError` 终止，保留已有日志和统计 |
| 晋升 copy2 失败（如磁盘满） | 抛 `IOError` 终止，gen4 和 gen5 均保留（copy 失败时目标未更新，源不被删除） |
| 晋升 copy2 成功后 remove 失败 | 抛 `IOError`，gen4 已更新为 gen5 内容（晋升语义正确），gen5 副本残留但不影响后续逻辑（下一轮开始时 copy 会覆盖 gen5） |
| KeyboardInterrupt (Ctrl+C) | `run()` 方法内部捕获，打印"用户中断"，输出当前已完成的统计，正常返回已收集的 dict（不抛异常）。若 Ctrl+C 在子进程执行期间触发，子进程会随主进程退出（Windows 下 subprocess.run 不需手动 terminate） |
| 多实例并发运行 | v1.0 不支持并发保护，同时运行多个实例会导致 gen4/gen5 权重竞争修改。建议在 `__init__()` 中检测锁文件 `ai/models/cnn_v3/.iterative_lock`，存在时抛异常拒绝启动 |
| 多次运行数据覆盖 | v1.0 不处理多次运行的数据隔离，path{round_num}.json 文件名相同会覆盖前次运行数据。如需保留历史数据，用户应手动备份 v4 目录 |
| create_samples 中间数据 | v1.0 不清理 `ai/ai_data/` 下的 round 目录等中间数据，长期运行可能占用磁盘空间。由用户手动管理 |
| `main()` 异常处理 | `main()` 捕获 `FileNotFoundError`、`RuntimeError`、`ValueError`、`IOError` 等异常，打印友好错误消息后以非零退出码退出。`KeyboardInterrupt` 由 `run()` 内部处理，`main()` 不重复捕获 |
