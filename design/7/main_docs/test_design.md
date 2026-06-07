# 测试设计

---

## 1. 测试文件清单

| 文件 | 类别 | 说明 |
|------|------|------|
| `tests/ai/unit/test_base_trainer_lr.py` | 单元测试 | BaseTrainer 构造器 `lr` 参数相关测试 |
| `tests/ai/unit/test_train_main_lr.py` | 单元测试 | CLI `--lr` 参数解析和校验测试 |
| `tests/ai/integration/test_train_main_lr.py` | 集成测试 | 端到端 `--lr` 参数传递链路测试 |

---

## 2. 单元测试用例

### 2.1 BaseTrainer 构造器 — lr 参数

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| UT-1.1 | 不传 lr 时使用默认值 | `BaseTrainer(fresh=True)` | `optimizer.param_groups[0]['lr']` 等于 0.001 |
| UT-1.2 | 显式传入 lr=0.005 | `BaseTrainer(fresh=True, lr=0.005)` | `optimizer.param_groups[0]['lr']` 等于 0.005 |
| UT-1.3 | 传入科学计数法 lr=1e-4 | `BaseTrainer(fresh=True, lr=1e-4)` | `optimizer.param_groups[0]['lr']` 等于 0.0001 |
| UT-1.4 | 传入 lr=0 应抛异常 | `BaseTrainer(fresh=True, lr=0)` | 抛 `ValueError`，消息含 "lr" |
| UT-1.5 | 传入 lr=-0.01 应抛异常 | `BaseTrainer(fresh=True, lr=-0.01)` | 抛 `ValueError`，消息含 "lr" |
| UT-1.6 | lr=None 语义等于不传 | `BaseTrainer(fresh=True, lr=None)` | `optimizer.param_groups[0]['lr']` 等于 0.001 |
| UT-1.7 | lr 与 weight_decay 同时传入不冲突 | `BaseTrainer(fresh=True, lr=0.005, weight_decay=0.02)` | lr=0.005，weight_decay=0.02 |

### 2.2 BaseTrainer 子类兼容性

| 用例编号 | 用例描述 | 输入 | 预期输出 |
|----------|---------|------|---------|
| UT-2.1 | cnn_v3 训练器透传 lr | `get_trainer("cnn_v3.2", lr=0.005)` | 返回的训练器 optimizer lr 等于 0.005 |
| UT-2.2 | cnn_v3 训练器不传 lr 使用默认值 | `get_trainer("cnn_v3.2")` | 返回的训练器 optimizer lr 等于 0.001 |
| UT-2.3 | fc_v1 训练器透传 lr | `get_trainer("fc_v1", lr=0.01)` | 返回的训练器 optimizer lr 等于 0.01 |

### 2.3 train_main.py CLI 参数解析

| 用例编号 | 用例描述 | 命令行参数模拟 | 预期结果 |
|----------|---------|--------------|---------|
| UT-3.1 | 正常解析 --lr 0.0005 | `--model fc_v1 --train-from-dir /tmp --lr 0.0005` | `args.lr` 等于 0.0005 |
| UT-3.2 | 正常解析 --lr 1e-4 | `--model fc_v1 --train-from-dir /tmp --lr 1e-4` | `args.lr` 等于 0.0001 |
| UT-3.3 | 不指定 --lr 时取默认值 | `--model fc_v1 --train-from-dir /tmp` | `args.lr` 等于 0.001 |
| UT-3.4 | --lr 0 解析通过但标记为非法 | `--model fc_v1 --train-from-dir /tmp --lr 0` | `args.lr` 等于 0.0，`args.lr <= 0` 为 True（后置校验由 main() 中的 `parser.error` 在 IT-3.1 验证） |
| UT-3.5 | --lr -0.01 解析通过但标记为非法 | `--model fc_v1 --train-from-dir /tmp --lr -0.01` | `args.lr` 等于 -0.01，`args.lr < 0` 为 True（后置校验由 main() 中的 `parser.error` 在 IT-3.1 验证） |
| UT-3.6 | --lr abc 报错退出 | `--model fc_v1 --train-from-dir /tmp --lr abc` | `SystemExit`，退出码 2，stderr 含类型错误 |
| UT-3.7 | --help 输出含 --lr 说明 | `--help` | stdout 含 "--lr" 及其帮助文本 |

### 2.4 runners.py 训练参数打印

| 用例编号 | 用例描述 | 前置条件 | 预期输出 |
|----------|---------|---------|---------|
| UT-4.1 | 指定 --lr 后打印对应值 | 传入 lr=0.005 的训练器 | stdout 含 "初始学习率 (lr): 0.005" |
| UT-4.2 | 不指定 --lr 时打印默认值 | 传入 lr=None（默认）的训练器 | stdout 含 "初始学习率 (lr): 0.001" |
| UT-4.3 | 打印位于训练参数区块 | 执行 `run_train_from_dir` | "初始学习率 (lr)" 行位于 "训练方法" 行之后 |

---

## 3. 集成测试用例

### 3.1 端到端参数传递

| 用例编号 | 用例描述 | 前置条件 | 验证点 |
|----------|---------|---------|--------|
| IT-1.1 | --lr 值正确传递到优化器 | mock 完整流程（含样本数据），指定 `--lr 0.005` | ① optimizer lr = 0.005 ② 打印信息含 0.005 ③ 训练正常完成 |
| IT-1.2 | 不指定 --lr 使用默认值 | mock 完整流程，不指定 `--lr` | ① optimizer lr = 0.001 ② 打印信息含 0.001 ③ 训练正常完成 |
| IT-1.3 | --lr 配合 --lr-scheduler step | mock 完整流程，`--lr 0.01 --lr-scheduler step --lr-step-size 5 --lr-gamma 0.5`，epochs=12 | ① epoch 1 lr=0.01 ② epoch 6 lr=0.005 ③ epoch 11 lr=0.0025 |
| IT-1.4 | --lr 配合 --lr-scheduler plateau | mock 完整流程，`--lr 0.01 --lr-scheduler plateau --lr-factor 0.5 --lr-patience 1000` | ① 初始 lr=0.01 ② scheduler 从此起点衰减 |

### 3.2 向后兼容

| 用例编号 | 用例描述 | 前置条件 | 验证点 |
|----------|---------|---------|--------|
| IT-2.1 | 意图 6 中 train_main 调用不受影响 | 模拟 `python ai/train_main.py --model cnn_v3.5 --train-from-dir data.json --epochs 1 --batch-size 4096 --shuffle` | ① 不指定 --lr ② 训练正常完成 ③ optimizer lr=0.001 |
| IT-2.2 | iterative_train.py 间接调用不受影响 | 通过 get_trainer 创建训练器，不传 lr | optimizer lr 等于 0.001 |

### 3.3 错误处理

| 用例编号 | 用例描述 | 前置条件 | 预期结果 |
|----------|---------|---------|---------|
| IT-3.1 | --lr 0 时训练不启动 | `--lr 0` | argparse 报错退出，不创建训练器，不加载数据 |
| IT-3.2 | --lr 非法字符串时训练不启动 | `--lr abc` | argparse 报错退出 |

---

## 4. 验收用例集（与 PRD §5 对齐）

### AC-5.1 参数解析验证（对应 PRD §5.1）

| PRD ID | 覆盖用例 |
|--------|---------|
| A1 "指定有效学习率 0.0005" | UT-1.2、UT-3.1 |
| A2 "指定科学计数法 1e-4" | UT-1.3、UT-3.2 |
| A3 "指定默认值 0.001" | UT-1.1、UT-3.3 |
| A4 "不指定参数使用默认值" | UT-1.1、UT-1.6、UT-3.3 |
| A5 "--lr 0 报错" | UT-1.4、UT-3.4 |
| A6 "--lr -0.01 报错" | UT-1.5、UT-3.5 |

### AC-5.2 训练行为验证（对应 PRD §5.2）

| PRD ID | 覆盖用例 |
|--------|---------|
| B1 "初始学习率传递给优化器" | IT-1.1、IT-1.2 |
| B2 "verbose 输出包含学习率" | UT-4.1、UT-4.2 |
| B3 "学习率随 scheduler 衰减" | IT-1.3、IT-1.4 |
| B4 "无 scheduler 时学习率不变" | IT-1.1（不指定 scheduler） |

### AC-5.3 向后兼容验证（对应 PRD §5.3）

| PRD ID | 覆盖用例 |
|--------|---------|
| C1 "现有命令不受影响" | IT-2.1 |
| C2 "意图 6 命令正常执行" | IT-2.1 |
| C3 "--help 含新参数" | UT-3.7 |

### AC-5.4 训练信息打印验证（对应 PRD §5.4）

| PRD ID | 覆盖用例 |
|--------|---------|
| D1 "指定 --lr 时打印" | UT-4.1 |
| D2 "未指定 --lr 时打印" | UT-4.2 |

---

## 5. Mock 策略

### 5.1 训练器单元测试

Mock 模型实例（`BaseModel`），使 `network` 为 `nn.Linear(10, 225)`，仅验证 optimizer 的 lr 设置，不执行实际训练。使用 `unittest.mock.MagicMock` 构造 model 和 network。

### 5.2 CLI 参数测试

使用 `argparse.ArgumentParser.parse_args` 直接测试参数解析，不启动完整训练流程。对校验逻辑的测试通过 `pytest.raises(SystemExit)` 捕获退出。

### 5.3 集成测试

Mock 样本数据加载（`SampleLoader.load` 和 `prepare_tensors` 返回小规模随机张量），epochs 设为小值（如 3~12），避免长时间训练。scheduler 测试使用小 step_size（如 5）加速验证。

### 5.4 不 Mock 的模块

- `torch.optim.AdamW`：真实构造，验证 lr 参数正确传入
- `BaseTrainer.__init__`：完整执行构造流程，不 mock
- `argparse`：真实解析，验证参数组合行为
