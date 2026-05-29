# step_id 匹配机制设计

## 1. 概述

### 1.1 背景

当前验收工具通过比较 input 向量（226-dim tuple）在 record.json 与样本文件之间建立 step 级映射。该方案存在以下问题：

| 问题 | 说明 |
|------|------|
| 不可靠 | D4 增强后的样本 input 已被变换，无法与原始 record.json 的 input 匹配 |
| 性能差 | 226 维 tuple 逐一比较，O(n·m) 复杂度（n=步数, m=样本数） |
| 有歧义 | 理论上不同局面可能产生相同 input（如对称局面），导致误匹配 |

### 1.2 方案

在 record.json 的每个 history entry 和样本文件的每个 sample 中加入全局唯一的 `step_id` 字段，以 step_id 作为匹配键替代 input 向量比对。

**step_id 格式**：UUID4 字符串

- 生成方式：`str(uuid.uuid4())`
- 示例：`"a3f2c1e0-7b8d-4f9a-b5e6-2c1d0f3a4b5e"`
- 特性：跨轮次、跨文件、跨机器全局唯一，零碰撞

### 1.3 影响范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `ai/data_recorder.py` | 修改 | record_move() 写入 step_id |
| `ai/sample_creator.py` | 修改 | 样本输出中加入 step_id |
| `tools/create_samples_tool/data_parser.py` | 修改 | 匹配逻辑从 input 比对改为 step_id 查找 |

---

## 2. 数据格式变更

### 2.1 record.json 新格式

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
    {
      "step_id": "b7e4d2c1-9a3f-4e5b-8c6d-1f0a3b2e4c5d",
      "input": [226-dim],
      "output": [225-dim],
      "position": [row, col]
    }
  ],
  "winner": 1,
  "end_board": [226-dim]
}
```

**新增字段**：

| 字段 | 类型 | 必存在 | 说明 |
|------|------|--------|------|
| history[].step_id | string | 是 | 全局唯一标识，UUID4 格式 |

### 2.2 样本输出文件新格式

```json
[
  {
    "step_id": "a3f2c1e0-7b8d-4f9a-b5e6-2c1d0f3a4b5e",
    "input": [226-dim],
    "output": [225-dim],
    "mask": [225-dim]
  },
  {
    "step_id": "c5d7e9f1-3a2b-4c8d-9e0f-1a2b3c4d5e6f",
    "input": [226-dim],
    "output": [225-dim],
    "mask": [225-dim]
  }
]
```

**新增字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| step_id | string | 与 record.json 中对应步的 step_id 一致 |

---

## 3. 实现方案

### 3.1 data_recorder.py 变更

**修改点**：`record_move()` 方法

```python
# 变更前
entry = {
    "input": input_vector,
    "output": output_vector,
    "position": list(position)
}

# 变更后
import uuid

entry = {
    "step_id": str(uuid.uuid4()),
    "input": input_vector,
    "output": output_vector,
    "position": list(position)
}
```

**关键约束**：
- step_id 在 `record_move()` 中调用 `uuid.uuid4()` 生成，无需任何上下文信息
- step_id 必须在 entry 的**第一个字段**，便于人工阅读和调试时快速定位
- uuid 为标准库模块，无需额外依赖

### 3.2 sample_creator.py 变更

**修改点 1**：`_build_sample()` — winning_move 模式

需要接收 step_id 参数并写入样本：

```python
# 变更前
@staticmethod
def _build_sample(input_vector: list, position: list) -> Dict:
    output = [0.0] * 225
    r, c = position[0], position[1]
    if 0 <= r < 15 and 0 <= c < 15:
        output[r * 15 + c] = 1.0
    mask = [1] * 225
    return {
        "input": input_vector,
        "output": output,
        "mask": mask,
    }

# 变更后
@staticmethod
def _build_sample(input_vector: list, position: list, step_id: str = None) -> Dict:
    output = [0.0] * 225
    r, c = position[0], position[1]
    if 0 <= r < 15 and 0 <= c < 15:
        output[r * 15 + c] = 1.0
    mask = [1] * 225
    sample = {
        "input": input_vector,
        "output": output,
        "mask": mask,
    }
    if step_id is not None:
        sample["step_id"] = step_id
    return sample
```

**修改点 2**：`_build_winning_move_samples()` — 传入 step_id

```python
# 变更前
def _build_winning_move_samples(self, history: list) -> List[Dict]:
    last_step = history[-1]
    return [self._build_sample(last_step["input"], last_step["position"])]

# 变更后
def _build_winning_move_samples(self, history: list) -> List[Dict]:
    last_step = history[-1]
    step_id = last_step.get("step_id")
    return [self._build_sample(last_step["input"], last_step["position"], step_id)]
```

**修改点 3**：`_build_winner_path_samples()` — 传入 step_id

```python
# 变更前
samples.append({
    "input": step["input"],
    "output": output,
    "mask": [1] * 225,
})

# 变更后
sample = {
    "input": step["input"],
    "output": output,
    "mask": [1] * 225,
}
if "step_id" in step:
    sample["step_id"] = step["step_id"]
samples.append(sample)
```

**修改点 4**：`_augment_sample()` — D4 增强样本保留 step_id

```python
# 变更后
augmented.append({
    "input": new_board + [player],
    "output": new_output,
    "mask": new_mask,
    **({"step_id": sample["step_id"]} if "step_id" in sample else {})
})
```

> **D4 增强说明**：增强后的样本与原始样本共享同一个 step_id。这是合理的——它们来自同一步，只是经过了旋转变换。验收工具展示时无需区分增强变体（因为 Row 3 只需知道"该步有没有样本"以及"样本 output 是什么"）。对于 D4 产生的 8 个变体，匹配时取第一个（恒等变换 tid=0），即原始样本。

### 3.3 data_parser.py 变更（验收工具侧）

**修改点**：`match_samples()` 函数

```python
# 变更前（input 向量比对）
def match_samples(history, samples):
    step_inputs = {}
    for i, step in enumerate(history):
        step_inputs[i] = tuple(step["input"])
    sample_map = {}
    for sample in samples:
        key = tuple(sample["input"])
        sample_map[key] = sample["output"]
    result = {}
    for i, input_vec in step_inputs.items():
        if input_vec in sample_map:
            result[i] = sample_map[input_vec]
    return result

# 变更后（step_id 查找）
def match_samples(history, samples):
    # 构建 step_id → sample_output 映射
    sample_map = {}
    for sample in samples:
        sid = sample.get("step_id")
        if sid is not None:
            sample_map[sid] = sample["output"]

    # 用 step_id 匹配
    result = {}
    for i, step in enumerate(history):
        sid = step.get("step_id")
        if sid is not None and sid in sample_map:
            result[i] = sample_map[sid]

    return result
```

**性能对比**：

| 指标 | input 向量比对 | step_id 查找 |
|------|---------------|-------------|
| 构建映射 | O(m) × 226-dim tuple | O(m) × string hash |
| 查找 | O(n) × 226-dim tuple hash | O(n) × string hash |
| 误匹配风险 | 有（理论存在） | 无 |
| D4 增强样本 | 无法匹配（input 已变换） | 正常匹配（step_id 不变） |

---

## 4. 向后兼容性

### 4.1 旧格式 record.json（无 step_id 字段）

- `step.get("step_id")` 返回 None → 跳过该步匹配
- 与当前"input 向量比对失败时不匹配"行为一致
- **不提供** fallback 到 input 向量比对的兼容逻辑（旧数据本身就不需要验收工具验证）

### 4.2 旧格式样本文件（无 step_id 字段）

- `sample.get("step_id")` 返回 None → 跳过该样本
- 所有步 sample_output=null → 前端 Row 3 全占位图

### 4.3 过渡策略

- 改造后**新执行**的 create_samples.py 会同时产出含 step_id 的 record.json 和样本文件
- 旧的已存在数据不迁移，验收工具对旧数据显示 Row 3 全占位图
- 无需数据迁移脚本

---

## 5. 决策记录

| # | 决策 | 理由 |
|---|------|------|
| S1 | step_id 使用 UUID4 | 跨轮次/跨文件全局唯一，不依赖 game_num 等局部上下文，零碰撞 |
| S2 | step_id 放在 entry 的第一个字段 | 便于调试时快速定位 |
| S3 | D4 增强样本共享原始 step_id | 同一步的不同变换本质上是同一步，验收展示无需区分 |
| S4 | D4 匹配取恒等变换(tid=0)的 output | 验收工具关注的是"原始 output 是什么"，不是变换后的 |
| S5 | 不提供 input 向量 fallback 兼容 | 旧数据无需验收，兼容逻辑增加复杂度不值得 |
| S6 | step_id 在 record_move() 中生成 | 每步落子时生成 UUID，是最自然的生成时机 |

---

## 6. 测试用例表

| # | 测试点 | 所属模块 | 用例说明 |
|---|--------|---------|---------|
| ST1 | record.json 含 step_id | data_recorder | 每个 history entry 的 step_id 为 UUID4 格式 |
| ST2 | step_id 全局唯一 | data_recorder | 不同步的 step_id 互不相同（UUID4 天然保证） |
| ST3 | 样本文件含 step_id | sample_creator | winning_move 模式样本的 step_id 与 record.json 最后一步一致 |
| ST4 | winner_path 样本 step_id | sample_creator | 每个样本的 step_id 与 record.json 对应步一致 |
| ST5 | D4 增强保留 step_id | sample_creator | 增强后的 8 个变体都有相同的 step_id |
| ST6 | step_id 匹配正确 | data_parser | match_samples 用 step_id 查找，结果与预期一致 |
| ST7 | 旧 record.json 无 step_id | data_parser | step.get("step_id")=None → 所有步 sample_output=null |
| ST8 | 旧样本文件无 step_id | data_parser | sample.get("step_id")=None → 所有步 sample_output=null |
| ST9 | 平局对局无样本 | data_parser | SampleCreator 跳过平局 → 无 step_id 可匹配 → 全 null |
| ST10 | step_id 首字段位置 | data_recorder | JSON 输出中 step_id 在 input 之前 |
