# 接口变更详细设计

---

## 1. 变更模块：ai/iterative_train.py

### 1.1 类：IterativeTrainer

#### 构造器签名变更

**变更前**：

```
IterativeTrainer(project_root=None)
```

**变更后**：

```
IterativeTrainer(ai1, ai2, sample_dir, project_root=None)
```

#### 新增参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `ai1` | `str` | 父代模型 spec，如 `"cnn_v4.99"` |
| `ai2` | `str` | 子代模型 spec，如 `"cnn_v4.100"` |
| `sample_dir` | `str` | 样本输出目录基路径，如 `"ai/merge_samples2"` |

#### 内部属性变更

| 旧属性 | 新属性 | 说明 |
|--------|--------|------|
| `_gen4_path` | `_ai1_path` | 父代 .pth 文件路径 |
| `_gen5_dir` | `_ai2_dir` | 子代权重目录 |
| `_gen5_path` | `_ai2_path` | 子代 .pth 文件路径 |
| (无) | `_ai1` | 父代模型 spec 字符串 |
| (无) | `_ai2` | 子代模型 spec 字符串 |
| (无) | `_model_name` | 解析出的模型名称 |
| (无) | `_gen_of_ai1` | 父代代数 |
| (无) | `_gen_of_ai2` | 子代代数 |

#### 新增校验

| 校验项 | 抛出异常 |
|--------|---------|
| `ai1` 与 `ai2` 的模型名不一致 | `ValueError` |

保留的现有校验：`pth_of_ai1` 文件存在（`FileNotFoundError`）、锁文件存在（`RuntimeError`）。

---

### 1.2 函数：main()

#### 签名

不变：`main()`

#### 新增逻辑

在函数体内新增 argparse 定义和参数校验：

```python
parser = argparse.ArgumentParser(...)
parser.add_argument('--ai1', type=str, required=True, ...)
parser.add_argument('--ai2', type=str, required=True, ...)
parser.add_argument('--sample_dir', type=str, required=True, ...)
args = parser.parse_args()

# 校验: ai1/ai2 格式（含 . 且 gen 为正整数）
try:
    name1, gen_str1 = args.ai1.rsplit(".", 1)
    gen1 = int(gen_str1)
    name2, gen_str2 = args.ai2.rsplit(".", 1)
    gen2 = int(gen_str2)
except (ValueError, IndexError):
    parser.error(
        "--ai1 和 --ai2 的格式必须为 模型名.代数，如 cnn_v4.99"
    )

# 校验: ai1 与 ai2 模型名一致
if name1 != name2:
    parser.error("--ai1 和 --ai2 的模型名必须一致")

trainer = IterativeTrainer(
    ai1=args.ai1,
    ai2=args.ai2,
    sample_dir=args.sample_dir,
)
trainer.run()
```

#### 新增异常捕获

`main()` 中的格式校验和模型名一致性校验均通过 `parser.error()` 处理，不依赖 try/except 捕获 `ValueError`。

`IterativeTrainer.__init__` 中新增 `ValueError`（模型名不一致，作为代码直接调用时的防御性校验）。

---

## 2. 内部方法签名变更

以下方法无签名变更，仅内部实现将硬编码字符串替换为 `self._ai1` / `self._ai2`：

| 方法 | 变更内容 |
|------|---------|
| `_generate_samples` | `--model "cnn_v4.99"` → `--model self._ai1`；output 路径使用 `self._sample_output_dir` |
| `_train_model` | `--model "cnn_v4.100"` → `--model self._ai2` |
| `_run_battle` | `--ai1 "cnn_v4.99"` / `--ai2 "cnn_v4.100"` → `--ai1 self._ai1` / `--ai2 self._ai2` |
| `_parse_battle_output` | 正则中 `cnn_v4\.99` / `cnn_v4\.100` → `re.escape(self._ai1)` / `re.escape(self._ai2)` |
| `_print_round_result` | 硬编码 `cnn_v4.100` → `self._ai2` |
| `_print_final_report` | 硬编码 `cnn_v4.99` / `cnn_v4.100` → 动态 |
| `_build_stop_reason` | 同上 |
| `_print_round_header` | 无变更 |
| `_init_model` | 路径使用 `self._ai1_path` / `self._ai2_path` |
| `_promote` | 同上 |
| `_create_lock` / `_remove_lock` | lock 路径基于 `self._model_name` 的 models 目录 |
| `_validate_samples` | 无变更 |
| `_build_result_dict` | `final_model_path` 使用 `self._ai1_path` |

---

## 3. 未变更模块

| 模块 | 原因 |
|------|------|
| `ai/registry.py` | 无需修改 |
| `ai/base_model.py` | gen 路径计算逻辑已存在，无需修改 |
| `ai/create_samples.py` | 仅作为子进程调用，CLI 接口不变 |
| `ai/train_main.py` | 同上 |
| `ai/battle.py` | 同上 |
