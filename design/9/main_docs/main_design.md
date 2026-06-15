# 主设计文档：变异步过滤 + 移除 winning_move 模式 + winner_path 恢复增强

| 字段 | 值 |
|------|-----|
| intention_id | 9 |
| PRD | requirements/9/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

本次变更涉及三个核心改动：

1. **新增变异步过滤**：在 `SampleCreator` 中新增步级过滤能力，仅保留"变异步"（引擎落子不在模型 argmax 集合中）生成样本。温度采样始终启用，步级过滤始终生效。
2. **移除 `winning_move` 模式**：从 `SampleCreator`、CLI 参数、所有代码路径中彻底移除，仅保留 `winner_path`
3. **恢复 `winner_path` 数据增强**：`SampleCreator` 默认启用 D4 增强（8 变换），`--no-augment` 恢复为有效开关

交互图见 [variant_move_filter_flow.puml](variant_move_filter_flow.puml)。

---

## 2. 变异步定义与判定

### 2.1 定义

**变异步**：对局中采样方的一步，该步引擎实际落子位置**不在**模型输出概率分布的 argmax 集合中。

形式化：设模型输出概率分布 \( \mathbf{p} = (p_1, \dots, p_{225}) \)，argmax 集合为 \( \{j \mid p_j = \max(\mathbf{p})\} \)，实际落子索引为 \( k \)。若 \( k \notin \text{argmax}(\mathbf{p}) \)，则该步为变异步。

### 2.2 判定函数

在 `ai/create_samples.py` 中新增 `is_variant_move(step: dict) -> bool`：

```
1. output = step["output"]           # 225 维概率分布
2. pos = step["position"]            # [row, col]
3. idx = pos[0] * 15 + pos[1]
4. 若 idx 越界 → 视为非变异步，返回 False（兜底：越界步不保留）
5. max_val = max(output)
6. argmax_indices = {i for i, v in enumerate(output) if v == max_val}
7. return idx not in argmax_indices
```

### 2.3 生效条件

变异步过滤始终生效（温度采样始终启用）：

- 温度采样没有开关（`--no-temp-sampling` 已移除），所有 record 均有 `temp_side_this_game` 值
- `move_filter` 参数注入后即对所有获胜方步生效，不再有任何跳过场景
- 注：API 层面 `move_filter=None` 时不执行步级过滤，这是为保持 `create()` 接口灵活性与 `record_filter` 设计一致（均通过注入可选回调控制行为）。CLI 调用路径始终传入 `is_variant_move`

---

## 3. 模块变更

### 3.1 变更清单

| 文件 | 变更类别 | 说明 |
|------|---------|------|
| `ai/sample_creator.py` | 重构 | 移除 `mode` 参数；`create()` 新增 `move_filter`；恢复 `augment` 默认行为 |
| `ai/create_samples.py` | 修改 | 新增 `is_variant_move()`；CLI 移除 `winning_move`、`--mode`、`--no-temp-sampling`；`--temp-side` 默认值改为 `random`；新增步级统计输出 |

### 3.2 不变模块

| 模块 | 原因 |
|------|------|
| `ai/game_filter.py` | 对局级过滤逻辑不变 |
| `ai/move_sampler.py` | 温度采样逻辑不变 |
| `ai/data_recorder.py` | 数据记录已完备（含 output 和 position） |
| `ai/game_manager.py` | 对战管理不变 |

---

## 4. SampleCreator 重构

### 4.1 构造函数简化

移除 `mode` 参数，仅保留 `augment`：

**变更前**：
```python
SampleCreator(mode: str = "winning_move", augment: bool = True)
```

**变更后**：
```python
SampleCreator(augment: bool = True)
```

- 不再有 `SUPPORTED_MODES` 类属性（无多模式需求）
- `self.augment = augment`（不再有 `mode != "winner_path"` 条件）
- 无模式校验逻辑

### 4.2 create() 新增 move_filter

**变更前**：
```python
def create(self, record_files: List[str],
           record_filter: Optional[Callable[[dict], bool]] = None) -> List[Dict]
```

**变更后**：
```python
def create(self, record_files: List[str],
           record_filter: Optional[Callable[[dict], bool]] = None,
           move_filter: Optional[Callable[[dict], bool]] = None) -> Tuple[List[Dict], dict]
```

| 新参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `move_filter` | `Optional[Callable[[dict], bool]]` | `None` | 步级过滤函数。接收单个 history step dict，返回 `True` 保留、`False` 跳过。`None` 表示不过滤 |

**返回值变更**：从 `List[Dict]` 变为 `Tuple[List[Dict], dict]`：
```python
stats = {
    "total_moves": int,        # 获胜方总步数
    "variant_moves": int,      # 变异步数（move_filter 保留的步）
    "skipped_moves": int,      # 跳过的步数
    "move_filter_active": bool # 是否有任何对局启用了步级过滤
}
```

### 4.3 核心逻辑变更

`create()` 方法内部流程变更：

```
遍历每局 record：
    winner 校验 → 平局/未结束跳过
    history 为空 → 跳过
    record_filter 校验 → 不满足则跳过

    winner_indices = 获胜方步索引 (黑胜 → [0,2,4,...]; 白胜 → [1,3,5,...])

    对每个 i ∈ winner_indices:
        step = history[i]

        # 变异步过滤（始终生效）
        if move_filter is not None:
            if not move_filter(step):
                skipped += 1
                continue

        kept += 1
        构造软标签 output
        构造 sample
        if self.augment:
            samples.extend(8 个变换变体)
        else:
            samples.append(sample)
```

### 4.4 移除的代码

| 移除项 | 说明 |
|--------|------|
| `SUPPORTED_MODES` 类属性 | 无多模式需求 |
| `self.mode` 属性 | 不再需要 |
| `_build_winning_move_samples()` | winning_move 模式移除 |
| `_build_sample()` | 仅被 `_build_winning_move_samples` 调用，无其他调用方 |
| `_is_temp_enabled()` | 温度采样始终启用，无需判断 |
| `mode != "winner_path"` 增强禁用逻辑 | 改为始终按 `augment` 参数决定 |
| 模式分派逻辑（`create()` 中的 if/elif） | 代码块 | 固定为 winner_path 逻辑 |

#### 保留的方法（不变）

| 方法 | 说明 |
|------|------|
| `_build_winner_path_samples()` | 获胜方每步构造软标签样本 |
| `_build_winner_path_output()` | 软标签 output 构造（不变） |
| `_transform_index()` | D4 变换索引（不变） |
| `_apply_transform()` | D4 变换向量（不变） |
| `_augment_sample()` | D4 增强（不变） |

---

## 5. create_samples.py CLI 变更

### 5.1 --mode 参数

**删除**。由于仅剩 `winner_path` 一种模式，`--mode` 参数不再需要，从 argparse 中移除。

### 5.2 --no-temp-sampling 参数

**删除**。温度采样始终启用，不开启无意义，该参数从 argparse 中移除。

### 5.3 --temp-side 参数

默认值从 `both` 改为 `random`，每局随机决定采样方。

### 5.4 --no-augment 参数

- 行为恢复：传入时禁用增强，不传时默认启用增强
- 不再因 `winner_path` 模式而被强制禁用

### 5.5 新增函数

`is_variant_move(step: dict) -> bool`：判定单个历史步是否为变异步（定义见 §2.2）。

### 5.6 调用链路

```
# 变更前
creator = SampleCreator(mode=args.mode, augment=not args.no_augment)
samples = creator.create_from_round_dir(round_dir, record_filter=should_keep_game)

# 变更后
creator = SampleCreator(augment=not args.no_augment)
samples, stats = creator.create_from_round_dir(
    round_dir,
    record_filter=should_keep_game,
    move_filter=is_variant_move,
)
```

### 5.7 统计输出

变更后的统计输出格式（见 PRD §4.4）：

```
样本过滤: 启用 (采样方获胜过滤)
  保留对局: 45
  过滤对局: 55
步级过滤: 启用 (变异步过滤)
  总步数:     320
  变异步:     142
  跳过:       178
数据增强: 启用 (D4 群 8 变换)
  最终样本数: 1136
```

---

## 6. 数据流

```
create_samples.py CLI
    │  argparse 解析 --no-augment
    │
    ▼
SampleCreator(augment=not args.no_augment)
    │
    │  调用 create_from_round_dir(round_dir,
    │      record_filter=should_keep_game,
    │      move_filter=is_variant_move)
    │
    ▼
SampleCreator.create()
    │
    │  对局级过滤: should_keep_game(record)
    │      ├─ 单方采样 + 采样方胜 → 保留
    │      ├─ 双方采样 + 有胜者 → 保留
    │      └─ 其余 → 丢弃
    │
    │  步级过滤: is_variant_move(step)  [始终生效]
    │      ├─ 实际位置 ∉ argmax → 变异步，保留
    │      └─ 实际位置 ∈ argmax → 跳过
    │
    │  样本生成 + D4 增强
    │
    ▼
返回 (samples, stats)
    │
    ▼
统计输出 + 写入 JSON
```

---

## 7. 向后兼容

| 场景 | 兼容性 |
|------|--------|
| `--mode` 参数 | **已删除**。传此参数时 argparse 报错 |
| `--no-temp-sampling` 参数 | **已删除**。传此参数时 argparse 报错 |
| `--temp-side` 参数 | 默认值从 `both` 改为 `random` |
| `--no-augment` | 恢复为有效开关 |
| 不传任何参数 | 默认行为：winner_path + temp-side=random + augment 启用 |
| Python API 直接调用 `SampleCreator()` | 不再接受 `mode` 参数，仅 `augment` |
| `create()` 返回值 | 从 `List[Dict]` 变为 `Tuple[List[Dict], dict]`，调用方需适配 |

> **实施前检查**：用 `grep` 搜索全仓库对 `SampleCreator.create(` 和 `create_from_round_dir(` 的调用，确保所有调用方都适配返回值变更。

---

## 8. 错误处理

| 场景 | 处理方式 |
|------|---------|
| 获胜方全程无变异步 | 该局产出 0 样本（正常行为，不告警） |
| step 中 `output` 字段缺失 | `KeyError` 直接抛出（调用方责任，fail-fast） |
| step 中 `position` 字段缺失 | `KeyError` 直接抛出（与 `output` 缺失策略一致） |
| position 越界 | `is_variant_move` 返回 `False`，跳过该步 |
| output 长度 ≠ 225 | `is_variant_move` 返回 `False`，跳过该步 |
| record_files 为空 | 返回空 samples + 零值 stats |

---

## 9. 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `design/9/main_docs/main_design.md` | 新增 | 本设计文档 |
| `design/9/main_docs/interface_changes.md` | 新增 | 接口变更详情 |
| `design/9/main_docs/test_design.md` | 新增 | 测试用例集 |
| `design/9/main_docs/variant_move_filter_flow.puml` | 新增 | 流程交互图 |
| `ai/sample_creator.py` | 修改 | 移除 mode；新增 move_filter；恢复增强 |
| `ai/create_samples.py` | 修改 | 新增 is_variant_move；CLI 调整；统计输出 |
| `tests/ai/test_sample_creator.py` | 修改 | 移除 winning_move 测试；新增变异步过滤测试 |
| `tests/ai/test_create_samples.py` | 修改 | 更新 CLI 测试 |
