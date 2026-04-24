# 自我对弈与训练分离设计文档

## 问题背景

当前 `--cycle` 模式将自我对弈数据生成和模型训练耦合在一个循环中：每轮先对弈→评价→训练→保存。用户希望将这两个步骤独立执行，实现：

1. **仅生成数据**：自我对弈 → 评价 → 输出 sample.json，不训练
2. **仅训练**：从已有的 sample 数据目录训练模型

## 设计目标

1. 新增 `--self-play` CLI 模式：仅执行自我对弈+评价，产出训练数据
2. 新增 `--train-from-dir` CLI 模式：从已有的训练数据目录训练，自动发现所有 sample.json
3. 三种模式互斥：`--cycle`、`--self-play`、`--sample-dir`、`--train-from-dir` 四选一
4. 向后兼容：`--cycle` 和 `--sample-dir` 行为不变

## 数据目录结构回顾

```
ai/
  ai_data/                              ← DataRecorder 写入
    {model}_cycle_{round}_{N}/          ← 轮次目录（round_dir）
      1/record.json
      2/record.json
      ...
  train_data/                           ← Evaluator 写入
    {model}_cycle_{round}_{N}/          ← 映射目录
      1/sample.json
      2/sample.json
      ...
```

Evaluator 将 `ai_data` 路径映射为 `train_data` 路径，目录结构一一对应。

## API 设计

### 1. 新类：SelfPlayGenerator

**文件**: `ai/self_play_generator.py`

```python
class SelfPlayGenerator:
    """自我对弈数据生成器（仅生成数据，不训练）"""

    def __init__(self, model, max_workers: int = None):
        """
        Args:
            model: BaseModel 实例，用于创建 AIEngine
            max_workers: 评价阶段并行进程数
        """

    def generate(self, games: int = 100, rate: float = 0.9,
                 project_root: str = None) -> dict:
        """执行自我对弈并生成训练样本

        Args:
            games: 对局数
            rate: 奖励衰减率，传给 Evaluator
            project_root: 项目根目录

        Returns:
            {
                "round_dir": str,           # ai_data 中的轮次目录
                "train_data_dir": str,      # train_data 中对应目录
                "sample_dirs": List[str],   # 所有 sample.json 所在目录
                "sample_count": int,        # 样本文件数量
                "results": List[dict]       # 对局结果
            }
        """
```

### 2. 辅助函数：discover_sample_dirs

**文件**: `ai/train_main.py`

```python
def discover_sample_dirs(train_data_dir: str) -> List[str]:
    """发现目录下所有包含 sample.json 的子目录

    扫描 train_data_dir 下的直接子目录，返回包含 sample.json 的子目录列表。

    Args:
        train_data_dir: 训练数据根目录（如 ai/train_data/fc_v1_cycle_1_5）

    Returns:
        包含 sample.json 的子目录路径列表，按子目录名数字排序
    """
```

### 3. CLI 参数变更

**文件**: `ai/train_main.py`

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--self-play` | flag | - | 仅执行自我对弈+评价，生成训练数据 |
| `--train-from-dir` | str | - | 训练数据目录路径（自动发现其下所有 sample.json） |

参数互斥验证：
- `--cycle`、`--self-play`、`--sample-dir`、`--train-from-dir` 四选一
- `--self-play` 必须配合 `--model` 和 `--games`
- `--train-from-dir` 必须配合 `--model`，可选 `--epochs`、`--batch-size`、`--shuffle`、`--lr-scheduler` 等

### 4. 使用示例

```bash
# 仅生成数据（100 局自我对弈 + 评价）
python ai/train_main.py --model cnn_v1 --self-play --games 100

# 仅训练（从指定目录发现所有 sample.json 合并训练）
python ai/train_main.py --model cnn_v1 --train-from-dir ai/train_data/cnn_v1_cycle_1_5 --epochs 1000

# 仅训练（mini-batch 模式）
python ai/train_main.py --model cnn_v1 --train-from-dir ai/train_data/cnn_v1_cycle_1_5 --epochs 1000 --batch-size 32 --shuffle

# 现有模式不受影响
python ai/train_main.py --model cnn_v1 --cycle --rounds 10 --games 100
python ai/train_main.py --model cnn_v1 --sample-dir ai/train_data/cnn_v1_cycle_1_5/1
```

## 主流程伪代码

### --self-play 模式

```
1. get_trainer(model_name) → 获取 trainer（仅使用 trainer.model）
2. generator = SelfPlayGenerator(trainer.model)
3. result = generator.generate(games=games, rate=rate)
4. 打印结果：
   - 对局统计（黑胜/白胜/平局）
   - 生成样本数
   - 训练数据目录路径（供 --train-from-dir 使用）
```

### --train-from-dir 模式

```
1. get_trainer(model_name) → 获取 trainer
2. sample_dirs = discover_sample_dirs(train_from_dir)
3. 若 sample_dirs 为空，报错退出
4. 打印训练信息（样本数、epochs 等）
5. 根据 batch_size 选择训练方法：
   - 有 batch_size → trainer.train_mini_batch(sample_dirs, ...)
   - 无 batch_size → trainer.train_batch(sample_dirs, ...)
6. 保存模型
```

### SelfPlayGenerator.generate 流程

```
1. 确定项目根目录和数据目录
2. 创建 AIEngine(model)
3. 创建 DataRecorder(ai_name=f"{model_name}_self_play")
4. 执行 AIGameManager.start_ai_vs_ai(engine, engine, games, recorder)
5. 统计对局结果
6. 创建 Evaluator，调用 evaluate_round(round_dir)
7. 收集 sample_dirs = [os.path.dirname(f) for f in result["sample_files"]]
8. 计算训练数据目录路径（将 ai_data 路径映射为 train_data 路径）
9. 返回结果字典
```

## 测试用例

### SelfPlayGenerator 测试

| 用例 | 说明 |
|------|------|
| `test_generate_creates_data_recorder_with_correct_name` | DataRecorder 使用 `{model_name}_self_play` 作为 ai_name |
| `test_generate_calls_start_ai_vs_ai_with_correct_games` | start_ai_vs_ai 被调用，games 参数正确 |
| `test_generate_calls_evaluator_evaluate_round` | evaluate_round 被调用，round_dir 正确 |
| `test_generate_returns_correct_result_dict` | 返回字典包含 round_dir、sample_dirs、sample_count 等字段 |
| `test_generate_with_keyboard_interrupt` | Ctrl+C 时优雅退出，打印已有结果 |

### discover_sample_dirs 测试

| 用例 | 说明 |
|------|------|
| `test_discover_finds_all_sample_dirs` | 正确发现所有包含 sample.json 的子目录 |
| `test_discover_skips_dirs_without_sample` | 跳过没有 sample.json 的子目录 |
| `test_discover_returns_sorted_dirs` | 返回按数字排序的目录列表 |
| `test_discover_empty_dir_returns_empty` | 空目录返回空列表 |
| `test_discover_nonexistent_dir_returns_empty` | 不存在的目录返回空列表 |

### CLI 参数互斥测试

| 用例 | 说明 |
|------|------|
| `test_self_play_exclusive_with_cycle` | --self-play 与 --cycle 互斥 |
| `test_self_play_exclusive_with_sample_dir` | --self-play 与 --sample-dir 互斥 |
| `test_self_play_exclusive_with_train_from_dir` | --self-play 与 --train-from-dir 互斥 |
| `test_train_from_dir_exclusive_with_cycle` | --train-from-dir 与 --cycle 互斥 |
| `test_train_from_dir_exclusive_with_sample_dir` | --train-from-dir 与 --sample-dir 互斥 |

## 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ai/self_play_generator.py` | 新增 | 自我对弈数据生成器 |
| `ai/train_main.py` | 修改 | 新增 --self-play / --train-from-dir 参数和执行逻辑 |
| `tests/ai/test_self_play_generator.py` | 新增 | SelfPlayGenerator 单元测试 |
| `tests/cli/test_train_main_cli.py` | 新增 | CLI 参数互斥测试 |
