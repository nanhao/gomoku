# Test Design

## Test Case Set

### T1: Normal Exit — All Games Complete

**Given**: 输出目录为空，运行 100 局对弈
**When**: 正常完成所有对局
**Then**:
- `sample.json` 包含 100 局全部样本
- `summary.json` 中 `completed_games` = 100，`total_samples` = 实际样本数
- 文件内容格式正确（每个样本含 state/pi/z 字段）

### T2: Periodic Save — Verify Every 50 Games

**Given**: 输出目录为空，运行 100 局对弈
**When**: 正常完成所有对局
**Then**:
- 在第 50 局完成后，`sample.json` 已存在于磁盘，包含前 50 局样本
- 在第 100 局（终局保存）完成后，`sample.json` 包含全部 100 局样本
- `summary.json` 中 `completed_games` 在各保存点正确递增

### T3: Ctrl+C Interrupt — Data Preserved

**Given**: 输出目录为空，运行 100 局对弈
**When**: 在第 35 局完成后发送 SIGINT
**Then**:
- `sample.json` 存在于磁盘，包含前 35 局样本
- `summary.json` 中 `completed_games` = 35
- 脚本以 exit code 0 退出并输出中断提示

### T4: Ctrl+C During Game — Completed Games Only

**Given**: 输出目录为空，运行 100 局对弈
**When**: 在第 36 局进行中发送 SIGINT
**Then**:
- `sample.json` 包含前 35 局样本（第 36 局未完成，不包含）
- `summary.json` 中 `completed_games` = 35
- 不包含 z=0 的脏样本

### T5: Existing sample.json — Merge on Startup

**Given**: 输出目录已有 `sample.json`（含前 50 局样本）和 `summary.json`
**When**: 使用相同 `--output` 再次运行 50 局
**Then**:
- 启动时 `all_samples` 初始值为已有 50 局样本
- 最终 `sample.json` 包含 100 局样本（50 旧 + 50 新）
- `summary.json` 中 `total_samples` 为合并后总数

### T6: No Existing sample.json — Clean Start

**Given**: 输出目录存在但无 `sample.json`
**When**: 运行 100 局
**Then**:
- 正常完成，`sample.json` 包含 100 局样本
- `summary.json` 正确生成

### T7: Atomic Write — No Corruption

**Given**: 输出目录已存在 `sample.json`
**When**: 在保存过程中模拟进程崩溃（写入 .tmp 后、rename 前终止）
**Then**:
- 原 `sample.json` 内容不变（未被部分覆盖）
- 残留的 `.tmp` 文件不影响下次启动（下次启动仅读取 `sample.json`）

### T8: Multiple Periodic Saves — Cumulative Correctness

**Given**: 输出目录为空，运行 120 局对弈
**When**: 正常完成
**Then**:
- 第 50 局保存：50 局样本
- 第 100 局保存：100 局样本
- 最终保存：120 局样本
- 所有保存点样本数单调递增

### T9: Ctrl+C Before Any Game Completes

**Given**: 输出目录为空，运行 100 局对弈
**When**: 在第 1 局开始后、完成前发送 SIGINT
**Then**:
- `sample.json` 不存在（无已完成对局）
- `summary.json` 不存在
- 脚本输出提示"无已完成对局"后退出

### T10: Empty Output Directory Creation

**Given**: `--output` 指定不存在的目录路径
**When**: 运行对弈
**Then**:
- 自动创建输出目录（与旧行为一致）
- 正常保存 `sample.json` 和 `summary.json`

### T11: Load Existing — Corrupted JSON

**Given**: 输出目录存在格式损坏的 `sample.json`（如被截断的 JSON）
**When**: 启动脚本
**Then**:
- 输出警告信息
- 以空列表 `[]` 作为初始值继续运行（不阻塞）
