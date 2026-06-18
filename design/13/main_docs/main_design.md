# 主设计文档：iterative_train.py 参数化

| 字段 | 值 |
|------|-----|
| intention_id | 13 |
| PRD | intention/intention.md (L77-L91) |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

将 `ai/iterative_train.py` 中硬编码的模型标识和样本目录提取为 CLI 参数，使迭代训练脚本不再绑定特定模型和代数，支持任意多代模型（cnn_v3、cnn_v4 等）的进化训练。

**核心变量计算**（从 `--ai1`、`--ai2` 解析得出）：

| 变量 | 来源 | 计算方式 | 示例（`--ai1 cnn_v4.99 --ai2 cnn_v4.100`） |
|------|------|---------|------------------------------------------|
| `model_name` | 解析 `--ai1` | `spec.rsplit(".", 1)[0]` | `"cnn_v4"` |
| `gen_of_ai1` | 解析 `--ai1` | `int(spec.rsplit(".", 1)[1])` | `99` |
| `gen_of_ai2` | 解析 `--ai2` | `int(spec.rsplit(".", 1)[1])` | `100` |
| `pth_of_ai1` | 计算 | `ai/models/{model_name}/gen{gen_of_ai1}/{model_name}.pth` | `ai/models/cnn_v4/gen99/cnn_v4.pth` |
| `pth_of_ai2` | 计算 | `ai/models/{model_name}/gen{gen_of_ai2}/{model_name}.pth` | `ai/models/cnn_v4/gen100/cnn_v4.pth` |

交互图见 [iterative_train_flow.puml](iterative_train_flow.puml)。

---

## 2. 模块变更

### 2.1 变更清单

| 文件 | 变更类别 | 说明 |
|------|---------|------|
| `ai/iterative_train.py` | 重构 | 硬编码常量 → CLI 参数 + 动态计算；新增 argparse |

### 2.2 删除的模块级常量

| 常量 | 原因 |
|------|------|
| `BASE_MODEL_NAME = "cnn_v4"` | 从 `--ai1` 解析得出 `model_name` |
| `PARENT_GEN = 99` | 从 `--ai1` 解析得出 `gen_of_ai1` |
| `CHILD_GEN = 100` | 从 `--ai2` 解析得出 `gen_of_ai2` |
| `MODELS_DIR = "ai/models/cnn_v4"` | 从 `model_name` 动态拼接 |
| `SAMPLE_OUTPUT_DIR = "ai/merge_samples2/v100"` | 从 `--sample_dir` + `gen_of_ai2` 动态拼接 |

### 2.3 保留的模块级常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `SAMPLE_GAMES` | `500` | 目标样本局数 |
| `BATTLE_GAMES` | `100` | 对战局数 |
| `TRAIN_EPOCHS` | `300` | 训练 epoch 数 |
| `TRAIN_BATCH_SIZE` | `4096` | 训练 batch size |
| `TEMP_MOVES` | `2` | 温度采样步数 |
| `TEMPERATURE` | `2` | 温度值 |
| `PYTHON_EXECUTABLE` | `sys.executable` | 子进程 Python 解释器 |

### 2.4 新增 CLI 参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--ai1` | `str` | 是 | 无 | 父代模型 spec，如 `cnn_v4.99` |
| `--ai2` | `str` | 是 | 无 | 子代模型 spec，如 `cnn_v4.100` |
| `--sample_dir` | `str` | 是 | 无 | 样本输出目录基路径，如 `ai/merge_samples2` |

### 2.5 参数校验

| 校验项 | 处理方式 |
|--------|---------|
| `--ai1` / `--ai2` 格式（含 `.`，gen 为正整数） | `main()` 中 `rsplit(".", 1)` + `int()` 转换，失败时 `parser.error()` |
| `--ai1` 与 `--ai2` 的模型名必须一致 | 自定义校验，不一致时 `parser.error()` |
| `pth_of_ai1` 文件存在 | `IterativeTrainer.__init__` 中 `FileNotFoundError` |
| `--sample_dir` 非空 | argparse `required=True` |

### 2.6 动态正则（对战输出解析）

`_parse_battle_output` 中原本硬编码的 `cnn_v4\.99` / `cnn_v4\.100` 改为 `re.escape(ai1)` / `re.escape(ai2)` 动态构造正则：

```python
gen5_match = re.search(
    rf'{re.escape(ai2)} 胜:\s*(\d+)\s*局\s*\(([\d.]+)%\)', stdout
)
gen4_match = re.search(
    rf'{re.escape(ai1)} 胜:\s*(\d+)\s*局', stdout
)
```

### 2.7 日志输出中的模型名

所有硬编码 `cnn_v4.99` / `cnn_v4.100` 的打印语句改为使用变量 `ai1` / `ai2`：

- `_print_round_result`：`f"{ai2} 胜率 = {win_rate:.1f}%"` 
- `_print_final_report`：使用 `ai1`、`ai2` 替代硬编码字符串
- `_generate_samples` 中的 `--model "cnn_v4.99"` → `--model ai1`
- `_train_model` 中的 `--model "cnn_v4.100"` → `--model ai2`
- `_run_battle` 中的 `--ai1 "cnn_v4.99"` / `--ai2 "cnn_v4.100"` → 使用变量

---

## 3. 数据流

```
CLI (argparse)
    │
    │  --ai1 "cnn_v4.99"  ──→  解析: model_name="cnn_v4", gen_of_ai1=99
    │  --ai2 "cnn_v4.100" ──→  解析: gen_of_ai2=100
    │  --sample_dir "ai/merge_samples2"
    │
    ├─→ 校验: ai1 与 ai2 的 model_name 一致
    │
    ▼
IterativeTrainer.__init__(ai1, ai2, sample_dir)
    │
    ├─→ pth_of_ai1 = models/{model_name}/gen{gen_of_ai1}/{model_name}.pth
    ├─→ pth_of_ai2 = models/{model_name}/gen{gen_of_ai2}/{model_name}.pth
    ├─→ sample_output_dir = {sample_dir}/v{gen_of_ai2}
    ├─→ lock_path = models/{model_name}/.iterative_lock
    │
    ▼
迭代循环（gen_of_ai1, gen_of_ai2 全程不变）:
    cp pth_of_ai1 → pth_of_ai2
    → create_samples --model ai1 --output sample_output_dir/path.json
    → train_main --model ai2 --train-from-dir sample_output_dir
    → battle --ai1 ai1 --ai2 ai2
    → mv pth_of_ai2 → pth_of_ai1
    → i++
```

---

## 4. 循环中的变量不变性

`model_name`、`gen_of_ai1`、`gen_of_ai2` 在整个循环期间为**常量**。每一轮：

1. `cp pth_of_ai1 → pth_of_ai2`：将 gen99 权重复制到 gen100
2. 训练 gen100（更新 gen100 的权重文件）
3. 对战 gen99 vs gen100
4. `mv pth_of_ai2 → pth_of_ai1`：将训练后的 gen100 覆盖回 gen99

这与现有实现逻辑完全一致——只是代数编号不再硬编码。

---

## 5. 向后兼容

| 场景 | 影响 |
|------|------|
| 现有直接调用 `python ai/iterative_train.py`（无参数） | **破坏性变更**：必须显式指定 `--ai1`、`--ai2`、`--sample_dir` |
| `IterativeTrainer` 类被外部 import 调用 | `__init__` 签名变更（新增必选参数），现有调用方需更新 |
| 意图 12 的用法假设 | 不再有效，需改用新 CLI |

由于 `iterative_train.py` 本身是面向开发者的内部工具脚本，此次破坏性变更可接受。

---

## 6. 错误处理

| 场景 | 处理方式 |
|------|---------|
| `--ai1` / `--ai2` 不含 `.` 或 gen 非正整数 | `main()` 中 `rsplit` + `int()` 转换失败 → `parser.error()` |
| `--ai1` 和 `--ai2` 模型名不一致 | `parser.error()` |
| `pth_of_ai1` 文件不存在 | `FileNotFoundError`（保留现有行为） |
| 锁文件已存在 | `RuntimeError`（保留现有行为） |
| 子进程非零退出 | `RuntimeError`（保留现有行为） |
