# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/test_sample_creator.py` | 修改 | 移除 `winning_move` 测试；新增变异步过滤和增强恢复测试；更新 `create()` 返回值断言 |
| `tests/ai/test_create_samples.py` | 修改 | 移除 `--mode` 和 `--no-temp-sampling` 测试；新增 `is_variant_move` 单元测试和 `--temp-side` 默认值测试；新增步级统计输出测试 |
| `tests/ai/test_sample_creator_winner_path.py` | 修改 | 更新样本数预期（因 `create()` 返回值变更） |
| `tests/ai/test_sample_creator_winner_path_soft_label.py` | 修改 | 更新样本数预期（因 `create()` 返回值变更） |

> **原则**：`winning_move` 相关测试直接删除，不再需要。`winner_path` 相关测试的核心逻辑（软标签、步索引、跳过规则）不变，仅适配返回值类型变更。所有调用 `create()` 的测试需改为 `samples, stats = creator.create(...)` 二元组解包形式。

---

## 2. 单元测试用例：SampleCreator

### 2.1 构造函数变更

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| SC-01 | 不传任何参数，默认 augment=True | `SampleCreator()` | `creator.augment == True` |
| SC-02 | 显式传 augment=False | `SampleCreator(augment=False)` | `creator.augment == False` |
| SC-03 | 不再接受 mode 参数 | `SampleCreator(mode="winner_path")` | `TypeError`（不接受 mode 关键字参数） |

### 2.2 变异步过滤

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| MV-01 | 单方采样 + 单步变异步 | 黑胜对局（temp_side_this_game="black"），1 步 history，模型 output 在落子位概率为 0.1，其他位置有 0.9 | 该步是变异步 → 保留；samples 包含该步（若 augment=True 则 8 个变体）；stats.variant_moves=1, stats.skipped_moves=0 |
| MV-02 | 单方采样 + 单步非变异步 | 同上，但落子位 probability=0.9（唯一最大值） | 该步非变异步 → 跳过；samples 为空；stats.variant_moves=0, stats.skipped_moves=1 |
| MV-03 | 单方采样 + 落子位与另一个位置并列 argmax | 落子位 probability=0.5，另一个位置也是 0.5 | 落子位 ∈ argmax 集合 → 非变异步，跳过；samples 为空 |
| MV-04 | 单方采样 + 所有位置概率相等 | output 全部为 1/225 | 落子位 ∈ argmax 集合（所有都是）→ 非变异步，跳过；samples 为空 |
| MV-05 | 单方采样 + 多步混合 | 3 步：步1 变异、步2 非变异、步3 变异 | 保留步1和步3，跳步步2；stats.variant_moves=2, stats.skipped_moves=1, stats.total_moves=3 |
| MV-06 | 单方采样 + 全部变异步 | 3 步全为变异步 | 全部保留；stats.variant_moves=3, stats.skipped_moves=0 |
| MV-07 | 单方采样 + 全部非变异步 | 3 步全为非变异步 | 全部跳过；samples 为空；stats.variant_moves=0, stats.skipped_moves=3 |

### 2.3 变异步过滤：双方采样场景

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| MV-08 | 双方采样（temp_side="both"）+ 混合步 | 传入 `move_filter=is_variant_move`，temp_side="both"，3 步中 2 步变异 | move_filter **生效**，仅保留 2 步变异步；stats.move_filter_active=True, stats.variant_moves=2, stats.skipped_moves=1 |
| MV-09 | 双方采样 + 全部非变异步 | temp_side="both"，3 步全为非变异步 | move_filter 生效，全部跳过；samples 为空；stats.move_filter_active=True |

### 2.4 变异步过滤：move_filter 为 None

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| MV-10 | 不传 move_filter | `create(record_files, record_filter=should_keep_game)`，temp_side="black" | 全部获胜方步生成样本（不做步级过滤）；stats.move_filter_active=False |

### 2.5 变异步过滤：边界条件

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| MV-11 | position 越界 | temp_side="black"，落子坐标 [20, 20]（越界） | `is_variant_move` 返回 False，该步被跳过；不崩溃 |
| MV-12 | position 负坐标 | 落子坐标 [-1, 5] | 同上，返回 False，跳过 |
| MV-13 | output 全零 | output 为 225 个 0.0，落子在 [7,7] | max=0.0，落子位=0.0 → idx ∈ argmax（全零时所有位置并列）→ 非变异步，跳过 |
| MV-14 | 多局混合（单方+双方） | 3 局：局1 temp_side="black" + 局2 temp_side="both" + 局3 temp_side="white" | 三局均启用步级过滤；stats.move_filter_active=True |

### 2.6 异常与边界输入

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| EDGE-01 | step 缺少 output 字段 | `move_filter(step)`，step = `{"position": [7,7]}` | `KeyError`（fail-fast） |
| EDGE-02 | step 缺少 position 字段 | `move_filter(step)`，step = `{"output": [...]}` | `KeyError`（fail-fast） |
| EDGE-03 | output 长度 ≠ 225 | `is_variant_move`，output 长度 = 100 | `False`（视为越界，兜底跳过） |
| EDGE-04 | output 含 NaN | `is_variant_move`，output 含 `float("nan")` | `max()` 行为取决于 Python 版本；建议在函数中显式处理 `math.isnan` 返回 `False` |
| EDGE-05 | record_files 为空列表 | `create([])` | 返回 `([], stats)`，stats 各字段为 0，不崩溃 |
| EDGE-06 | record 缺少 temp_side_this_game 字段 | 旧格式 record 无此字段 | 不影响步级过滤（始终生效），但对局级过滤行为取决于 `should_keep_game` 实现 |

### 2.7 数据增强恢复

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| AUG-01 | augment=True（默认），单步变异步 | `SampleCreator()` + 1 步变异步 | 产出 8 个样本（D4 群完整变换） |
| AUG-02 | augment=False，单步变异步 | `SampleCreator(augment=False)` + 1 步变异步 | 产出 1 个样本 |
| AUG-03 | augment=True + 无变异步 | `SampleCreator()` + 全部步非变异步 | 产出 0 个样本 |
| AUG-04 | augment=True + 多步变异步 | 3 步变异步 | 产出 24 个样本（3 × 8） |
| AUG-05 | augment=True + 双方采样 + 变异步过滤 | temp_side="both"，3 步中 2 步变异 | 产出 16 个样本（2 × 8） |

### 2.8 winner_path 原有逻辑不变

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| WP-01 | 黑胜，获胜方步索引正确 | 黑胜（winner=1），4 步 history | 保留 history[0] 和 history[2]（索引 0,2） |
| WP-02 | 白胜，获胜方步索引正确 | 白胜（winner=2），5 步 history | 保留 history[1] 和 history[3]（索引 1,3） |
| WP-03 | 平局跳过 | winner=3 | samples 为空 |
| WP-04 | 未结束跳过 | winner=0 | samples 为空 |
| WP-05 | 空 history 跳过 | history=[] | samples 为空 |
| WP-06 | 软标签不变 | 落子位非 top1 | output 经软标签算法抬升后落子位为严格最大 |
| WP-07 | 软标签 sum=1 | 任意正常步 | output 所有元素之和 ≈ 1.0 |

### 2.9 create() 返回值变更

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| RET-01 | 返回二元组 | `create(record_files)` | 返回 `(samples, stats)`，类型为 `Tuple[List, dict]` |
| RET-02 | stats 字段完整 | 正常对局 | stats 含 `total_moves`、`variant_moves`、`skipped_moves`、`move_filter_active` 四个键 |
| RET-03 | stats.move_filter_active 正确 | temp_side="black" + 传 move_filter | `stats["move_filter_active"] == True` |
| RET-04 | stats.move_filter_active 正确 | temp_side="both" + 传 move_filter | `stats["move_filter_active"] == True` |
| RET-05 | stats.move_filter_active 正确 | 不传 move_filter | `stats["move_filter_active"] == False` |

---

## 3. 单元测试用例：create_samples.py

### 3.1 is_variant_move() 函数

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| IV-01 | 落子位是唯一最大值 | output[idx] = 0.9，其余位置均 ≤ 0.01 | `False`（非变异步） |
| IV-02 | 落子位是并列最大值之一 | output[idx] = 0.5，output[0] = 0.5，其余为 0 | `False`（非变异步，落子位 ∈ argmax） |
| IV-03 | 落子位不是最大值 | output[idx] = 0.1，output[0] = 0.8，其余为 0 | `True`（变异步） |
| IV-04 | 全零分布 | output 全为 0.0 | `False`（落子位 ∈ argmax，全零时所有位置并列） |
| IV-05 | 越界坐标 row=20 | position=[20, 20] | `False`（越界兜底，非变异步） |
| IV-06 | 越界坐标 row=-1 | position=[-1, 5] | `False`（越界兜底） |

### 3.2 CLI --mode 参数

**已删除**。`--mode` 参数已从 argparse 中完全移除。需验证：

| 用例编号 | 用例描述 | 命令行参数 | 预期结果 |
|----------|---------|----------|---------|
| CLI-01 | 不传 --mode | `--model x --output out.json --games 1` | 正常执行，无 mode 相关输出 |
| CLI-02 | 传 --mode（已删除） | `--mode winner_path` | `SystemExit`，退出码 2，stderr 含 "unrecognized arguments" |

### 3.3 CLI --no-temp-sampling 参数

**已删除**。需验证：

| 用例编号 | 用例描述 | 命令行参数 | 预期结果 |
|----------|---------|----------|---------|
| CLI-03 | 传 --no-temp-sampling（已删除） | `--no-temp-sampling` | `SystemExit`，退出码 2，stderr 含 "unrecognized arguments" |

### 3.4 CLI --temp-side 参数

| 用例编号 | 用例描述 | 命令行参数 | 预期结果 |
|----------|---------|----------|---------|
| CLI-04 | 不传 --temp-side | 无此参数 | 默认值为 `random` |
| CLI-05 | 传 --temp-side black | `--temp-side black` | 使用 `black` |

### 3.5 CLI --no-augment 参数

| 用例编号 | 用例描述 | 命令行参数 | 预期结果 |
|----------|---------|----------|---------|
| CLI-06 | 不传 --no-augment | 无此参数 | `SampleCreator(augment=True)` |
| CLI-07 | 传 --no-augment | `--no-augment` | `SampleCreator(augment=False)` |

### 3.6 统计输出

| 用例编号 | 用例描述 | 前置条件 | 预期输出片段 |
|----------|---------|---------|------------|
| STAT-01 | 单方采样时有步级统计 | temp_side="black"，有变异步和非变异步 | 输出含 "步级过滤: 启用 (变异步过滤)" 及 "总步数"、"变异步"、"跳过" 三行 |
| STAT-02 | 双方采样时也有步级统计 | temp_side="both"，有变异步和非变异步 | 输出含 "步级过滤: 启用 (变异步过滤)" 及统计数字 |
| STAT-03 | 数据增强为启用 | augment=True | 输出含 "数据增强: 启用 (D4 群 8 变换)" |
| STAT-04 | 数据增强为禁用 | augment=False | 输出含 "数据增强: 禁用" |

---

## 4. 集成测试用例

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| INT-01 | 单方采样端到端 + 变异步过滤 | `--temp-side black` | 仅采样方获胜的变异步生成样本；增强默认开启；统计输出完整 |
| INT-02 | 双方采样端到端 + 变异步过滤 | `--temp-side both` | 仅获胜方变异步生成样本；统计输出含步级过滤 |
| INT-03 | 端到端 + --no-augment | `--no-augment` | 每步仅 1 样本，无增强 |
| INT-04 | 默认参数端到端 | 无额外参数 | `--temp-side` 默认 `random`；变异步过滤生效；增强默认开启 |

---

## 5. 验收用例对照（PRD）

| PRD 需求点 | 覆盖用例 |
|-----------|---------|
| 变异步过滤仅保留引擎落子偏离 argmax 的步 | MV-01 ~ MV-07, IV-01 ~ IV-06 |
| 温度采样始终启用，变异步过滤始终生效 | MV-01 ~ MV-09, MV-14 |
| 双方采样也触发变异步过滤 | MV-08, MV-09 |
| 获胜方全程无变异步 → 0 样本 | MV-07, MV-09 |
| 移除 winning_move 模式 | SC-03 |
| 移除 --mode CLI 参数 | CLI-01, CLI-02 |
| 移除 --no-temp-sampling CLI 参数 | CLI-03 |
| --temp-side 默认值 random | CLI-04 |
| 默认开启 D4 增强 | AUG-01 |
| --no-augment 可关闭增强 | AUG-02, CLI-07 |
| 步级过滤统计输出 | STAT-01 ~ STAT-04 |
| 对局级过滤不变 | WP-01 ~ WP-05（复用已有测试） |
| 软标签算法不变 | WP-06, WP-07（复用已有测试） |

---

## 6. Mock 策略

### 6.1 SampleCreator 单元测试

不 Mock。使用真实 `SampleCreator` 实例，通过 `tempfile` 构造临时 `record.json` 文件，直接调用 `create()` 验证。

### 6.2 is_variant_move 单元测试

纯函数，直接调用验证，不 Mock。

### 6.3 CLI 参数测试

使用 `argparse.ArgumentParser.parse_args` 直接测试参数解析。对已删除的 `--mode` 和 `--no-temp-sampling` 参数使用 `pytest.raises(SystemExit)` 捕获。

### 6.4 集成测试

使用 `subprocess.run` 执行 `create_samples.py` 脚本，验证输出文件和 stdout。
