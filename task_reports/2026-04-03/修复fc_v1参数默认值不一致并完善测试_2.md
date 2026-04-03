# 修复 fc_v1 参数默认值不一致并完善测试

## 1. 需求内容
修复代码实现与设计文档参数默认值不一致的问题，并完善测试用例覆盖修复的情形。

## 2. 解决思路
1. 对比需求文档、设计文档和代码实现的参数默认值
2. 修复代码中的参数默认值使其与需求一致
3. 添加测试用例验证参数默认值
4. 更新设计文档以反映代码实现

## 3. 执行过程

### 3.1 发现的不一致项

| 参数 | 需求文档 | 设计文档 | train.py | trainer.py |
|------|---------|---------|----------|------------|
| `--epochs` | 1000 | 1000 | **10** ❌ | 10 |
| `--rounds` | 10 | 10 | **100** ❌ | 10 |
| `--games` | 100 | 100 | 100 | 100 |

### 3.2 修复内容

**train.py 修复**:
- `--epochs` 默认值从 10 改为 1000
- `--rounds` 默认值从 100 改为 10

**trainer.py 修复**:
- `train_cycle` 的 `epochs` 默认值从 100 改为 1000

### 3.3 添加测试用例

在 `tests/ai/test_fc_v1_trainer.py` 中添加 `TestFCv1TrainScript` 测试类：
- `test_train_script_epochs_default`: 测试命令行参数默认值
- `test_trainer_train_single_epochs_default`: 测试 train_single 默认 epochs
- `test_trainer_train_cycle_defaults`: 测试 train_cycle 默认参数值

### 3.4 更新设计文档

- 更新 FCv1Trainer 方法签名表
- 更新代码框架中的方法签名
- 更新训练周期流程描述（移除 TODO 标记）

## 4. 执行结果

### 4.1 测试结果
```
Ran 17 tests in 8.346s
OK
```

所有测试通过，新增 3 个测试用例：
- test_train_script_epochs_default
- test_trainer_train_cycle_defaults
- test_trainer_train_single_epochs_default

### 4.2 修复后参数一致性

| 参数 | 需求文档 | 设计文档 | train.py | trainer.py |
|------|---------|---------|----------|------------|
| `--epochs` | 1000 | 1000 | 1000 ✅ | 1000 ✅ |
| `--rounds` | 10 | 10 | 10 ✅ | 10 ✅ |
| `--games` | 100 | 100 | 100 ✅ | 100 ✅ |

## 5. 涉及文件
- `d:/projects/gomoku/ai/fc_v1/train.py` (已修改)
- `d:/projects/gomoku/ai/fc_v1/trainer.py` (已修改)
- `d:/projects/gomoku/tests/ai/test_fc_v1_trainer.py` (已修改)
- `d:/projects/gomoku/design/fc_v1_design.md` (已修改)
