# FC v1 模型默认加载行为修复

## 1. 需求内容

用户需求（来自 `requirements/key_points.md` 第 387 行）：
> 默认从已保存模型继续训练，支持通过参数指定从新模型开始训练

**问题分析**：
- 设计文档和代码都没有完全体现此需求
- `FCv1Model()` 默认不加载已保存模型，只有显式传入 `model_path` 才加载
- `FCv1Trainer()` 默认创建新的 `FCv1Model()`，不会自动加载 `fc_v1.pth`
- `train.py` 缺少 `--fresh` 参数支持从新模型开始训练

## 2. 解决思路

采用 TDD 开发流程：
1. 先更新设计文档，明确默认加载行为和 `--fresh` 参数
2. 编写测试用例覆盖新功能
3. 修改代码实现
4. 运行测试确保全部通过

## 3. 执行过程

### 3.1 更新设计文档

修改 `design/fc_v1_design.md`：
- 添加 `--fresh` 参数说明（命令行参数表）
- 添加使用示例
- 更新 8.3 节"默认加载行为"说明
- 更新 `FCv1Model.__init__` 和 `FCv1Trainer.__init__` 方法签名

### 3.2 编写测试用例

在 `tests/ai/test_fc_v1_model.py` 添加 `TestFCv1ModelDefaultLoad` 测试类：
- `test_default_load_when_model_exists`：测试默认加载已保存模型
- `test_no_load_when_model_not_exists`：测试模型不存在时创建新模型
- `test_fresh_parameter_skips_load`：测试 fresh=True 跳过加载
- `test_fresh_parameter_with_existing_default_model`：测试 fresh=True 强制使用新模型
- `test_model_path_takes_priority_over_default`：测试自定义路径优先级

在 `tests/ai/test_fc_v1_trainer.py` 添加 `TestFCv1TrainerFreshParameter` 测试类：
- `test_trainer_fresh_parameter_creates_new_model`：测试 fresh 参数
- `test_trainer_default_loads_existing_model`：测试默认加载行为
- `test_trainer_with_model_parameter`：测试传入 model 参数
- `test_trainer_fresh_with_sample_dir`：测试 fresh 与训练结合

### 3.3 修改代码实现

**model.py**：
- 添加 `fresh: bool = False` 参数
- 实现默认加载逻辑：fresh=True → 跳过加载 → 尝试加载 model_path → 尝试加载 DEFAULT_MODEL_PATH

**trainer.py**：
- 添加 `fresh: bool = False` 参数
- 调整初始化逻辑：fresh=True → 创建新模型 → 使用传入 model → 默认加载

**train.py**：
- 添加 `--fresh` 命令行参数
- 创建训练器时传入 `fresh=args.fresh`

### 3.4 运行测试

执行命令：
```bash
python -m unittest tests.ai.test_fc_v1_model tests.ai.test_fc_v1_trainer -v
```

结果：**40 个测试全部通过**

## 4. 执行结果

- 设计文档已更新，明确了默认加载行为
- 代码实现符合需求：默认加载已保存模型，可通过 `--fresh` 参数从新模型开始
- 所有测试通过，功能验证完成

## 5. 涉及文件

| 文件路径 | 操作类型 |
|----------|----------|
| `design/fc_v1_design.md` | 修改 |
| `ai/fc_v1/model.py` | 修改 |
| `ai/fc_v1/trainer.py` | 修改 |
| `ai/fc_v1/train.py` | 修改 |
| `tests/ai/test_fc_v1_model.py` | 修改 |
| `tests/ai/test_fc_v1_trainer.py` | 修改 |
