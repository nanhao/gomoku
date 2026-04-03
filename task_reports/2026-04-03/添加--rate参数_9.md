# 添加 --rate 参数支持

## 1. 需求内容

在 `requirements/key_points.md` 第 409 行新增一行需求：
```
- `--rate`: r的值，默认为 0.9
```

即在训练周期命令行中支持 `--rate` 参数，用于配置评价器（Evaluator）的奖励衰减率 r（原来硬编码为 0.9）。

## 2. 解决思路

`r` 值在 `Evaluator._compute_rewards` 中用于反向传播计算每步的 reward（`current_reward *= r`），之前硬编码为类常量 `DECAY_RATE = 0.9`。

改造链路：
```
train.py (--rate 参数)
    → FCv1Trainer.train_cycle(rate=...)
    → Evaluator(decay_rate=rate)
    → _compute_rewards 使用 self.decay_rate
```

## 3. 执行过程

### 步骤 1：更新设计文档 `design/fc_v1_design.md`

- FCv1Trainer.train_cycle 方法签名增加 `rate: float = 0.9`
- 参数说明表增加 `--rate` 行
- 训练周期伪代码更新为 `Evaluator(decay_rate=rate)`
- 版本历史新增 v1.1

### 步骤 2：编写测试用例（TDD）

**`tests/ai/test_evaluator.py`** 新增 `TestEvaluatorDecayRate` 测试类（4 个用例）：
- `test_default_decay_rate_is_0_9`：默认 decay_rate 为 0.9
- `test_custom_decay_rate_is_used`：自定义 decay_rate 被存储
- `test_custom_decay_rate_affects_reward_calculation`：decay_rate 影响实际 reward 计算
- `test_class_constant_decay_rate_unchanged`：类常量不被实例修改

**`tests/ai/test_fc_v1_trainer.py`** 新增 `TestTrainCycleRateParameter` 测试类（2 个用例）：
- `test_train_cycle_has_rate_parameter`：train_cycle 有 rate 参数且默认值 0.9
- `test_train_script_rate_default`：train.py --rate 默认值为 0.9

测试初始均为红色（失败），符合 TDD 预期。

### 步骤 3：修改 `ai/evaluator.py`

- `__init__` 增加 `decay_rate: float = None` 参数
- 初始化 `self.decay_rate = decay_rate if decay_rate is not None else self.DECAY_RATE`
- `_compute_rewards` 中将 `self.DECAY_RATE` 改为 `self.decay_rate`（2 处）

### 步骤 4：修改 `ai/fc_v1/trainer.py`

- `train_cycle` 签名新增 `rate: float = 0.9`
- 打印信息增加 rate 显示
- 创建 Evaluator 时传入 `decay_rate=rate`

### 步骤 5：修改 `ai/fc_v1/train.py`

- `argparse` 新增 `--rate` 参数（type=float, default=0.9）
- 打印信息增加 rate 显示
- `train_cycle` 调用增加 `rate=args.rate`

### 步骤 6：运行测试

- `test_evaluator.py`：9 个测试全部通过
- `test_fc_v1_trainer.py`：37 个测试全部通过（含新增测试）

## 4. 执行结果

全部测试通过。用法示例：

```bash
# 使用自定义衰减率 0.8
python ai/fc_v1/train.py --cycle --rounds 50 --games 100 --rate 0.8

# 使用默认衰减率 0.9
python ai/fc_v1/train.py --cycle
```

## 5. 涉及文件

| 文件 | 变更类型 |
|------|----------|
| `requirements/key_points.md` | 需求文档（只读，用户新增） |
| `design/fc_v1_design.md` | 更新设计文档（v1.0 → v1.1） |
| `ai/evaluator.py` | 修改：`__init__` 支持 `decay_rate` 参数 |
| `ai/fc_v1/trainer.py` | 修改：`train_cycle` 新增 `rate` 参数 |
| `ai/fc_v1/train.py` | 修改：命令行新增 `--rate` 参数 |
| `tests/ai/test_evaluator.py` | 新增 `TestEvaluatorDecayRate` 测试类 |
| `tests/ai/test_fc_v1_trainer.py` | 新增 `TestTrainCycleRateParameter` 测试类 |
