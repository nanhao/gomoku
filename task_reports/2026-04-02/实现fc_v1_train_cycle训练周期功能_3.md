# 实现 fc_v1 train_cycle 训练周期功能

## 需求内容

用户运行 `python ai/fc_v1/train.py --cycle` 时，提示功能未实现：
```
功能未实现: train_cycle 需要集成 game_manager 和 evaluator 实现。
建议先使用 train_single 进行单轮次训练。
```

需要实现 `train_cycle` 方法，使模型能够进行自我对弈训练迭代。

## 解决思路

根据设计文档 [fc_v1_design.md](d:\projects\gomoku\design\fc_v1_design.md) 中的训练周期流程设计：

1. 模型自我对弈 n 个轮次
2. 每个轮次 m 场对局
3. 每轮结束后用 sample 数据训练
4. 保存更新后的模型

需要集成的模块：
- `AIEngine`: 使用当前模型进行对弈决策
- `AIGameManager.start_ai_vs_ai()`: 执行 AI 内战
- `DataRecorder`: 记录对局数据
- `Evaluator`: 从 record.json 生成 sample.json

## 执行过程

### 1. 实现 trainer.py 中的 train_cycle 方法

文件: `d:\projects\gomoku\ai\fc_v1\trainer.py`

实现了完整的训练周期逻辑：
- 循环执行指定轮次数
- 每轮创建 DataRecorder 记录数据
- 调用 AIGameManager.start_ai_vs_ai() 执行自我对弈
- 使用 Evaluator 生成训练样本
- 调用 train_single() 训练模型
- 保存更新后的模型

关键代码结构：
```python
def train_cycle(self, rounds=100, games=100, epochs=10, project_root=None):
    for round_num in range(1, rounds + 1):
        # 1. 创建数据记录器
        recorder = DataRecorder(ai_name=f"fc_v1_cycle_{round_num}")
        
        # 2. 执行 AI 内战
        results = AIGameManager.start_ai_vs_ai(engine1, engine2, ...)
        
        # 3. 生成训练样本
        evaluator.evaluate_record(record_file)
        
        # 4. 训练模型
        self.train_single(sample_dir, epochs=epochs)
        
        # 5. 保存模型
        self.model.save()
```

### 2. 更新 train.py 支持训练周期模式

文件: `d:\projects\gomoku\ai\fc_v1\train.py`

修改了命令行调用，将 `--epochs` 参数传递给 `train_cycle`：
```python
trainer.train_cycle(
    rounds=args.rounds,
    games=args.games,
    epochs=args.epochs
)
```

### 3. 编写测试用例

文件: `d:\projects\gomoku\tests\ai\test_fc_v1_trainer.py`

添加了 `TestFCv1TrainerCycle` 测试类，包含 3 个测试用例：
- `test_train_cycle_basic`: 测试训练周期基本功能
- `test_train_cycle_creates_model_file`: 测试模型文件保存
- `test_train_cycle_with_existing_model`: 测试使用已有模型训练

### 4. 运行测试验证

执行完整测试套件：
```bash
python -m unittest tests.ai.test_fc_v1_trainer -v
```

结果：14 个测试全部通过

验证命令行工具：
```bash
python ai/fc_v1/train.py --cycle --rounds 1 --games 2
```

输出：
```
开始 fc_v1 训练周期...
轮次数: 1
每轮对局数: 2
...
对局结果: 黑胜 1, 白胜 1, 平局 0
生成 2 个样本文件
训练模型 (10 epochs)...
本轮平均损失: 91176.9578
模型已保存: ai/fc_v1/fc_v1.pth
训练周期完成！
```

## 涉及文件

### 修改的文件
1. `d:\projects\gomoku\ai\fc_v1\trainer.py` - 实现 train_cycle 方法
2. `d:\projects\gomoku\ai\fc_v1\train.py` - 更新命令行参数传递
3. `d:\projects\gomoku\tests\ai\test_fc_v1_trainer.py` - 添加测试用例

### 生成的文件
- `d:\projects\gomoku\ai\fc_v1\fc_v1.pth` - 训练后的模型文件
- `d:\projects\gomoku\ai_data/fc_v1_cycle_*/` - 训练数据目录
- `d:\projects\gomoku\train_data/fc_v1_cycle_*/` - 训练样本目录

## 使用说明

### 训练周期模式
```bash
# 默认参数（100轮，每轮100局）
python ai/fc_v1/train.py --cycle

# 自定义轮次和局数
python ai/fc_v1/train.py --cycle --rounds 50 --games 50

# 自定义训练轮数
python ai/fc_v1/train.py --cycle --rounds 10 --games 10 --epochs 5
```

### 单轮次训练（原有功能）
```bash
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1
```

## 注意事项

1. 训练周期会覆盖 `ai/fc_v1/fc_v1.pth` 模型文件
2. 每轮训练后会自动保存模型，可用于中断恢复
3. 训练数据保存在 `ai_data/` 和 `train_data/` 目录下
4. 每轮使用独立的目录命名（fc_v1_cycle_{轮次编号}）
