# 接口变更详细设计

---

## 1. 变更模块：ai/sample_creator.py

### 1.1 类：SampleCreator

#### 构造函数签名变更

**变更前**：
```
SampleCreator(mode: str = "winning_move", augment: bool = True)
```

**变更后**：
```
SampleCreator(augment: bool = True)
```

| 移除参数 | 说明 |
|----------|------|
| `mode` | 不再有多模式需求，仅 `winner_path` 一种行为 |

| 保留参数 | 变更 |
|----------|------|
| `augment` | 默认 `True`。不再有 `mode != "winner_path"` 强制禁用逻辑，直接使用传入值 |

#### 移除的方法和属性

| 移除项 | 类型 | 说明 |
|--------|------|------|
| `SUPPORTED_MODES` | 类属性 | 无多模式需求 |
| `_build_winning_move_samples()` | 方法 | winning_move 模式已移除 |
| `_build_sample()` | 方法 | 仅被 `_build_winning_move_samples` 调用，无其他调用方 |
| `self.mode` | 实例属性 | 不再需要 |
| 模式分派逻辑（`create()` 中的 if/elif） | 代码块 | 固定为 winner_path 逻辑 |
| `_is_temp_enabled()` | 方法 | 温度采样始终启用，无需判断 |

#### 保留的方法（不变）

| 方法 | 说明 |
|------|------|
| `_build_winner_path_samples()` | 获胜方每步构造软标签样本 |
| `_build_winner_path_output()` | 软标签 output 构造（不变） |
| `_transform_index()` | D4 变换索引（不变） |
| `_apply_transform()` | D4 变换向量（不变） |
| `_augment_sample()` | D4 增强（不变） |

---

### 1.2 方法：create()

#### 签名变更

**变更前**：
```
def create(self, record_files: List[str],
           record_filter: Optional[Callable[[dict], bool]] = None) -> List[Dict]
```

**变更后**：
```
def create(self, record_files: List[str],
           record_filter: Optional[Callable[[dict], bool]] = None,
           move_filter: Optional[Callable[[dict], bool]] = None) -> Tuple[List[Dict], dict]
```

#### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `move_filter` | `Optional[Callable[[dict], bool]]` | `None` | 步级过滤函数。接收单个 history step dict（含 `output` 和 `position` 字段），返回 `True` 保留、`False` 跳过。始终生效（温度采样始终启用）；`None` 不过滤 |

#### 返回值变更

**变更前**：`List[Dict]` — 样本列表

**变更后**：`Tuple[List[Dict], dict]` — 样本列表 + 统计信息

统计信息 dict 结构：
```
{
    "total_moves": int,          # 获胜方总步数（含被过滤的）
    "variant_moves": int,        # 变异步数（通过 move_filter 的步）
    "skipped_moves": int,        # 被 move_filter 跳过的步数
    "move_filter_active": bool,  # 是否有任何对局实际启用了步级过滤
}
```

#### 行为变更

- 移除 `mode` 分派：不再检查 `self.mode`，固定按 winner_path 逻辑处理
- 新增步级过滤：在遍历获胜方步时，若 `move_filter` 不为 `None`，则对每步调用 `move_filter(step)` 判定是否保留
- `augment` 直接生效：不再因 `winner_path` 模式而强制禁用增强

---

### 1.3 方法：create_from_round_dir()

#### 签名变更

**变更前**：
```
def create_from_round_dir(self, round_dir: str,
                          record_filter: Optional[Callable[[dict], bool]] = None) -> List[Dict]
```

**变更后**：
```
def create_from_round_dir(self, round_dir: str,
                          record_filter: Optional[Callable[[dict], bool]] = None,
                          move_filter: Optional[Callable[[dict], bool]] = None) -> Tuple[List[Dict], dict]
```

#### 变更说明

- 新增 `move_filter` 参数，透传给 `create()`
- 返回值类型与 `create()` 同步变更

---

## 2. 变更模块：ai/create_samples.py

### 2.1 新增函数：is_variant_move()

```
def is_variant_move(step: dict) -> bool
```

- **说明**：判定单个历史步是否为变异步（实际落子不在模型 argmax 集合中）
- **参数**：`step` — history 中的单步字典，需含 `output`（225 维概率）和 `position`（`[row, col]`）
- **返回**：`True` 表示变异步；`False` 表示非变异步或越界步
- **算法**：
  1. 从 `step["output"]` 获取概率分布
  2. 从 `step["position"]` 计算一维索引 `idx = row * 15 + col`
  3. 若 `idx` 越界 → 返回 `False`
  4. 计算 `argmax` 集合（所有概率等于最大值的索引）
  5. 返回 `idx not in argmax_set`

### 2.2 main() 变更

#### --mode 参数

**删除**。从 argparse 中完全移除 `--mode` 参数。

#### --no-temp-sampling 参数

**删除**。从 argparse 中完全移除 `--no-temp-sampling` 参数。温度采样始终启用。

#### --temp-side 参数

默认值从 `both` 改为 `random`。

#### --no-augment 语义恢复

- 变更前：`winner_path` 模式下 `--no-augment` 实际无效（因 `SampleCreator.__init__` 强制禁用增强）
- 变更后：`--no-augment` 为有效开关，传入时 `augment=False`，不传时 `augment=True`

#### SampleCreator 构造变更

**变更前**：
```python
creator = SampleCreator(mode=args.mode, augment=not args.no_augment)
```

**变更后**：
```python
creator = SampleCreator(augment=not args.no_augment)
```

#### create_from_round_dir 调用变更

**变更前**：
```python
samples = creator.create_from_round_dir(round_dir, record_filter=should_keep_game)
```

**变更后**：
```python
samples, stats = creator.create_from_round_dir(
    round_dir,
    record_filter=should_keep_game,
    move_filter=is_variant_move,
)
```

#### _print_filter_stats() 变更

新增步级过滤统计输出（参见 [main_design.md §5.7](main_design.md#57-统计输出)）。函数签名新增 `move_stats: dict` 参数。

#### 移除的代码

- `--mode` 参数定义及所有引用
- `--no-temp-sampling` 参数定义及所有引用
- `--mode` help 和 epilog 中关于 `winning_move` 的描述和示例
- `_build_winning_move_samples` 调用路径（已不存在）
