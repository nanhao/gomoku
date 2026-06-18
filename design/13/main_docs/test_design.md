# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/unit/test_iterative_train_params.py` | 单元测试 | CLI 参数解析、路径计算、模型名一致性校验 |

---

## 2. 单元测试用例

### 2.1 CLI 参数解析

| 用例编号 | 用例描述 | 命令行参数 | 预期结果 |
|----------|---------|-----------|---------|
| UT-1.1 | 三个参数正常指定 | `--ai1 cnn_v4.99 --ai2 cnn_v4.100 --sample_dir ai/merge_samples2` | `args.ai1="cnn_v4.99"`, `args.ai2="cnn_v4.100"`, `args.sample_dir="ai/merge_samples2"` |
| UT-1.2 | 缺少 --ai1 | `--ai2 cnn_v4.100 --sample_dir ai/merge_samples2` | `SystemExit`, 退出码 2 |
| UT-1.3 | 缺少 --ai2 | `--ai1 cnn_v4.99 --sample_dir ai/merge_samples2` | `SystemExit`, 退出码 2 |
| UT-1.4 | 缺少 --sample_dir | `--ai1 cnn_v4.99 --ai2 cnn_v4.100` | `SystemExit`, 退出码 2 |
| UT-1.5 | --ai1 不含代数（无 `.`） | `--ai1 cnn_v4 --ai2 cnn_v4.100 --sample_dir ai/merge_samples2` | `SystemExit`, 退出码 2（`rsplit` 解包失败 → `parser.error()`） |

### 2.2 模型名一致性校验

| 用例编号 | 用例描述 | 输入 | 预期结果 |
|----------|---------|------|---------|
| UT-2.1 | 相同模型名通过 | `ai1="cnn_v4.99"`, `ai2="cnn_v4.100"` | 校验通过 |
| UT-2.2 | 不同模型名报错 | `ai1="cnn_v3.99"`, `ai2="cnn_v4.100"` | `parser.error("模型名必须一致")` |
| UT-2.3 | 相同模型名但不同 gen 通过 | `ai1="cnn_v4.1"`, `ai2="cnn_v4.200"` | 校验通过 |

### 2.3 spec 解析与路径计算

| 用例编号 | 用例描述 | 输入 | 预期结果 |
|----------|---------|------|---------|
| UT-3.1 | 解析 cnn_v4.99 | `spec="cnn_v4.99"` | `("cnn_v4", 99)` |
| UT-3.2 | 解析 cnn_v3.1 | `spec="cnn_v3.1"` | `("cnn_v3", 1)` |
| UT-3.3 | 解析含多点模型名 | `spec="cnn_v4.2.99"` 用 `rsplit(".", 1)` 解析 | 得到 `("cnn_v4.2", 99)`（`rsplit` 仅分割最右 `.`） |
| UT-3.4 | 计算 pth_of_ai1 | `model_name="cnn_v4"`, `gen=99` | `ai/models/cnn_v4/gen99/cnn_v4.pth` |
| UT-3.5 | 计算 pth_of_ai2 | `model_name="cnn_v4"`, `gen=100` | `ai/models/cnn_v4/gen100/cnn_v4.pth` |
| UT-3.6 | 计算 sample_output_dir | `sample_dir="ai/merge_samples2"`, `gen_of_ai2=100` | `ai/merge_samples2/v100` |

### 2.4 对战输出解析（动态正则）

| 用例编号 | 用例描述 | 输入 stdout | 预期结果 |
|----------|---------|------------|---------|
| UT-4.1 | 解析 cnn_v4.99 vs cnn_v4.100 对战输出 | `"cnn_v4.100 胜: 55 局 (55.0%)"` | `gen5_wins=55`, `gen5_win_rate=55.0` |
| UT-4.2 | 解析含 model_name 中含点号的场景（如 `cnn_v4`） | `"cnn_v4.100 胜: 55 局 (55.0%)"` | 正则不误匹配 cnn_v4.100 中多余的点号 |
| UT-4.3 | 解析含平局输出 | stdout 含 `"平局: 10 局"` | `draws=10` |
| UT-4.4 | 解析含父代胜场输出 | stdout 含 `"cnn_v4.99 胜: 35 局"` | `gen4_wins=35` |

---

## 3. 集成测试用例

### 3.1 端到端流程

| 用例编号 | 用例描述 | 前置条件 | 验证点 |
|----------|---------|---------|--------|
| IT-1.1 | 完整一轮迭代（mock 子进程） | 存在 `ai/models/cnn_v4/gen1/cnn_v4.pth`；mock `create_samples`、`train_main`、`battle` 子进程返回成功 | ① cp 执行 ② 子进程参数正确 ③ mv 执行 ④ `_rounds` 包含一条记录 |
| IT-1.2 | KeyboardInterrupt 中断 | mock 子进程，运行中发送 Ctrl+C | ① 锁文件被清理 ② `_print_final_report` 被调用 |
| IT-1.3 | pth_of_ai1 不存在 | `--ai1 cnn_v4.999`（不存在的代数） | `FileNotFoundError` |

### 3.2 锁文件并发保护

| 用例编号 | 用例描述 | 前置条件 | 验证点 |
|----------|---------|---------|--------|
| IT-2.1 | 锁文件已存在 | 手动创建锁文件 | `RuntimeError`，消息含"锁文件" |

---

## 4. 验收用例集

### AC-1 参数解析验证

| PRD 意图 | 覆盖用例 |
|----------|---------|
| "--ai1 eg: cnn_v4.99" | UT-1.1, UT-3.1 |
| "--ai2 eg: cnn_v4.100" | UT-1.1, UT-3.5 |
| "--sample_dir eg: ai\\merge_samples2" | UT-1.1, UT-3.6 |
| 三个参数缺一不可 | UT-1.2, UT-1.3, UT-1.4 |

### AC-2 路径计算验证

| PRD 意图 | 覆盖用例 |
|----------|---------|
| "compute {pth_of_ai1} {pth_of_ai2} {gen_of_ai2} correctly" | UT-3.4, UT-3.5, UT-3.6 |

### AC-3 循环逻辑验证

| PRD 意图 | 覆盖用例 |
|----------|---------|
| 步骤 2~6 流程正确执行 | IT-1.1 |
| 无条件晋升 | IT-1.1 |
| 中断处理 | IT-1.2 |

---

## 5. Mock 策略

### 5.1 CLI 参数测试

使用 `argparse.ArgumentParser.parse_args` 直接测试参数解析。对校验逻辑的测试通过 `pytest.raises(SystemExit)` 捕获退出。

### 5.2 路径计算测试

直接测试 `IterativeTrainer.__init__` 的路径计算逻辑，不启动循环。需要 mock 文件系统存在性检查（`os.path.isfile` 返回 True 用于 pth 文件，False 用于 lock 文件）。

### 5.3 集成测试

Mock `subprocess.run` 返回成功的 `CompletedProcess`（含预定义的 stdout 用于对战输出解析测试）。mock `_create_lock` / `_remove_lock` 不执行实际文件操作。

### 5.4 不 Mock 的模块

- `IterativeTrainer.__init__`：完整执行路径计算和校验逻辑
- `re.escape` / `re.search`：真实调用，验证正则构造正确性
