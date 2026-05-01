# Sample Creator 与自我对弈数据生成独立脚本设计

## 背景

当前 `--self-play` 模式内嵌在 `train_main.py` 中，使用 Evaluator 生成训练样本。需求将其剥离为独立脚本，并引入全新的 `sample_creator` 替代 Evaluator，支持多种样本生成模式。

## 设计目标

1. 从 `train_main.py` 中彻底删除 `--self-play` 参数
2. 新建独立 CLI 脚本 `ai/create_samples.py`，执行对弈 + sample_creator 生成样本
3. 新建 `ai/sample_creator.py` 模块，支持多种模式（目前实现 `winning_move` 模式）
4. 新样本的 mask 全为 1，确保已落子位置参与训练时的 loss 计算

## API 设计

### 1. 新模块：ai/sample_creator.py

```python
class SampleCreator:
    """样本生成器，支持多种模式从 record.json 生成 sample"""

    def __init__(self, mode: str = "winning_move"):
        """
        Args:
            mode: 样本生成模式，目前仅支持 "winning_move"
        """

    def create(self, record_files: List[str]) -> List[Dict]:
        """从 record.json 文件列表生成 sample 列表

        Args:
            record_files: record.json 文件路径列表

        Returns:
            sample 数据列表，每个元素为 {"input": [...], "output": [...], "mask": [...]}
        """

    def create_from_round_dir(self, round_dir: str) -> List[Dict]:
        """从轮次目录自动发现所有 record.json 并生成 sample

        Args:
            round_dir: 轮次目录路径（含子目录如 1/record.json, 2/record.json）

        Returns:
            sample 数据列表
        """
```

### 2. 新脚本：ai/create_samples.py

CLI 参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必填 | 模型名称，如 cnn_v3 |
| `--mode` | str | winning_move | sample_creator 模式 |
| `--games` | int | 100 | 对局数 |
| `--output` | str | 必填 | 输出文件路径（.json） |

使用示例：
```bash
python ai/create_samples.py --model cnn_v3 --mode winning_move --games 100 --output ai/train_data/win_samples.json
```

### 3. 修改：ai/train_main.py

删除以下参数和逻辑：
- `--self-play` argparse 参数
- `--merge-output` argparse 参数（因为 --self-play 被删除，此参数也失去意义）
- `--self-play` 与 `--cycle` / `--train-from-dir` 的互斥校验中移除 `--self-play`

### 4. 修改：ai/runners.py

删除 `run_self_play` 函数及导入。

## 主流程伪代码

### create_samples.py CLI

```
1. 解析参数：--model, --mode, --games, --output
2. 加载模型：model = get_model(args.model)
3. 创建 AIEngine：engine = AIEngine(model)
4. 创建 DataRecorder：recorder = DataRecorder(ai_name=f"{model.name}_battle")
5. 执行对弈：results = AIGameManager.start_ai_vs_ai(engine, engine, args.games, recorder)
6. 打印对局统计
7. 获取轮次目录：round_dir = recorder.get_round_dir()
8. 生成样本：
   creator = SampleCreator(mode=args.mode)
   samples = creator.create_from_round_dir(round_dir)
9. 写入输出文件：
   确保输出目录存在
   json.dump(samples, args.output)
10. 打印统计：对局数、样本数、输出路径
```

### SampleCreator (winning_move 模式)

```
1. 遍历每个 record_file：
   a. json.load(record_file) → record
   b. 如果 record["winner"] == 0 或 3：跳过（平局或未结束）
   c. history = record["history"]
   d. 如果 history 为空：跳过
   e. last_step = history[-1]
   f. input_vector = last_step["input"]
   g. position = last_step["position"]  # [x, y]
   h. output = [0.0] * 225
   i. idx = position[0] * 15 + position[1]
   j. 如果 0 <= idx < 225：output[idx] = 1.0
   k. mask = [1] * 225
   l. samples.append({"input": input_vector, "output": output, "mask": mask})
2. 返回 samples
```

## 测试用例

### SampleCreator 单元测试

| 用例 | 说明 |
|------|------|
| `test_winning_move_basic` | 正常对局，最后一位置 output=1，其余=0，mask 全 1 |
| `test_winning_move_skip_draw` | winner=3 时跳过 |
| `test_winning_move_skip_no_winner` | winner=0 时跳过 |
| `test_winning_move_empty_history` | history 为空时跳过 |
| `test_winning_move_multiple_files` | 多文件合并，样本数正确 |
| `test_winning_move_invalid_position` | position 越界时该位置不设置（output 全 0）|
| `test_create_from_round_dir` | 从轮次目录自动发现 record.json |

### create_samples.py 集成测试

| 用例 | 说明 |
|------|------|
| `test_cli_creates_output_file` | 执行后输出文件存在且格式正确 |
| `test_cli_output_has_correct_sample_count` | 输出样本数与对局数一致（排除平局）|

### train_main.py 删除验证

| 用例 | 说明 |
|------|------|
| `test_self_play_param_removed` | --self-play 参数已不存在，解析时报错 |
| `test_merge_output_param_removed` | --merge-output 参数已不存在 |
| `test_cycle_and_train_from_dir_still_work` | --cycle 与 --train-from-dir 互斥校验仍正常 |

## 数据增强设计（第三十二次追加）

### 设计目标

对 `sample_creator` 生成的每个 sample 进行 D4 群（正方形对称群）变换，产生 **8 个变体（包含原始）**，数据量提升 8 倍。

### 8 种变换定义

设棋盘位置 (r, c)，0 ≤ r, c < 15，一维索引 idx = r × 15 + c。

| 编号 | 变换 | 新位置 (r', c') |
|------|------|-----------------|
| 0 | 恒等 | (r, c) |
| 1 | 顺时针 90° | (c, 14−r) |
| 2 | 180° | (14−r, 14−c) |
| 3 | 顺时针 270° | (14−c, r) |
| 4 | 左右镜像 | (r, 14−c) |
| 5 | 上下镜像 | (14−r, c) |
| 6 | 主对角线翻转 | (c, r) |
| 7 | 反对角线翻转 | (14−c, 14−r) |

### API 变更

`SampleCreator.__init__` 增加 `augment` 参数：

```python
class SampleCreator:
    def __init__(self, mode: str = "winning_move", augment: bool = True):
        """
        Args:
            mode: 样本生成模式
            augment: 是否启用数据增强（D4 群 8 变换），默认 True
        """
```

### CLI 变更

`create_samples.py` 新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-augment` | flag | False | 禁用数据增强 |

### 数据增强伪代码

```python
def _transform_index(idx, tid):
    r, c = divmod(idx, 15)
    # 根据 tid 选择变换公式计算新位置 (r2, c2)
    return r2 * 15 + c2

def _apply_transform(vector_225, tid):
    result = [0.0] * 225
    for old_idx in range(225):
        new_idx = _transform_index(old_idx, tid)
        result[new_idx] = vector_225[old_idx]
    return result

def _augment_sample(sample):
    board = sample["input"][:225]
    player = sample["input"][225]
    output = sample["output"]
    mask = sample["mask"]

    augmented = []
    for tid in range(8):
        new_board = _apply_transform(board, tid)
        new_output = _apply_transform(output, tid)
        new_mask = _apply_transform(mask, tid)
        augmented.append({
            "input": new_board + [player],
            "output": new_output,
            "mask": new_mask
        })
    return augmented
```

### 数据增强测试用例

| 用例 | 说明 |
|------|------|
| `test_augment_produces_eight_samples` | 1 个原始 sample 产生 8 个变体 |
| `test_augment_identity_is_first` | 第 0 个变体（恒等变换）与原始完全一致 |
| `test_augment_rotation_90_position` | 验证 R90 后 output 的 1 位置正确移动 |
| `test_augment_horizontal_flip_position` | 验证左右镜像后 output 的 1 位置正确移动 |
| `test_augment_composition_r90_twice` | 两次 R90 等价于 R180 |
| `test_augment_disabled_returns_one` | augment=False 时只返回 1 个原始 sample |

## 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `design/sample_creator_design.md` | 新增 | 本设计文档 |
| `ai/sample_creator.py` | 修改 | 增加数据增强逻辑 |
| `ai/create_samples.py` | 修改 | 增加 `--no-augment` CLI 参数 |
| `tests/ai/test_sample_creator.py` | 修改 | 增加数据增强测试 |
| `tests/ai/test_create_samples.py` | 修改 | 更新样本数预期（启用增强时为 8 倍）|

## 第三十八次追加：winner_path 模式

### 背景与动机

`winning_move` 模式只保留了每局的"最后一步（制胜那步）"作为 sample。这种策略
让模型只学到"如何把即将成形的五连补上最后一子"，对于整局布局缺乏监督信号。

新模式 `winner_path` 把**整局中获胜方的每一步**（除开局第 1 步）都作为 sample，
目的是让模型拟合"获胜方走出的整条路径"，学习整局的布局决策，而不仅仅是封口一步。

### 核心规则

| 规则项 | 取值/说明 |
|--------|-----------|
| 模式名 | `winner_path` |
| 选取范围 | 仅获胜方的落子步（败方的步丢弃） |
| 剔除开局 | **剔除获胜方的第 1 步**（开局第 1 子几乎无战术信息，且经增强后会把对称位置反复强化，引入位置偏置） |
| input | 每一步 history[i]["input"]（即落子前的棋盘 226 维向量） |
| output | 该步落子位 = 1.0，其余 225 维位置 = 0.0 |
| mask | 全 1（与 winning_move 一致，已落子位也参与 loss 计算） |
| 数据增强（D4） | **强制不启用**。因为 winner_path 本身已极大膨胀 sample 量（每局产生 k-1 条，k 为获胜方步数），叠加 8 倍 D4 会导致训练数据体量失控。传入 `augment=True` 时也忽略 |
| 平局（winner=3） | 整局跳过 |
| 未结束（winner=0） | 整局跳过 |
| 空 history | 整局跳过 |
| 获胜方仅 1 步 | 剔首后无剩余，产生 0 个 sample |
| position 越界 | 该 sample 的 output 保持全 0，不崩溃 |

### 获胜方步序的识别

项目惯例为"黑先行、双方交替落子"，history 中：
- `history[0]` = 黑第 1 步，`history[1]` = 白第 1 步，`history[2]` = 黑第 2 步，……
- 因此：
    - winner=1（黑胜）→ 获胜方步索引集 = `[0, 2, 4, ...]`
    - winner=2（白胜）→ 获胜方步索引集 = `[1, 3, 5, ...]`
- **剔除首步后**：
    - winner=1 → `[2, 4, 6, ...]`
    - winner=2 → `[3, 5, 7, ...]`

### API 变更

`SampleCreator.__init__` mode 白名单扩展为 `{"winning_move", "winner_path"}`。
`create()` 内部按 mode 分派到两条处理路径，D4 增强/平局跳过等公共逻辑复用。

```python
class SampleCreator:
    SUPPORTED_MODES = ("winning_move", "winner_path")

    def __init__(self, mode: str = "winning_move", augment: bool = True):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"不支持的样本生成模式: '{mode}'")
        self.mode = mode
        # winner_path 强制禁用增强
        self.augment = augment and mode != "winner_path"
```

### winner_path 伪代码

```
遍历每个 record_file：
    load record；winner in {0, 3} → 跳过；history 为空 → 跳过
    winner=1 时 start_idx = 0；winner=2 时 start_idx = 1
    获胜方索引 = [start_idx, start_idx+2, ..., <= len(history)-1]
    剔除第一个（开局） → winner_indices = 获胜方索引[1:]
    对每个 i ∈ winner_indices：
        input = history[i]["input"]
        output = [0.0]*225；idx = pos[0]*15 + pos[1]；若 0 <= idx < 225 → output[idx] = 1.0
        mask = [1]*225
        append sample（不做增强）
```

### CLI 变更

`ai/create_samples.py`：
- `--mode` 的 help 文本更新为列出支持的两种模式
- `epilog` 新增 winner_path 用法示例
- `--no-augment` 在 winner_path 下语义为无操作（因为该模式固定不增强）

示例：
```bash
python ai/create_samples.py --model cnn_v3.1 --mode winner_path \
    --games 200 --output ai/train_data/winner_path_samples.json
```

### 新增测试用例（winner_path）

测试文件放在 `tests/ai/test_sample_creator_winner_path.py`，
单独成文避免 `tests/ai/test_sample_creator.py` 超过 500 行。

| 用例 | 说明 |
|------|------|
| `test_black_wins_generates_correct_count` | 黑胜、history=5 步、获胜方 3 步，剔首 → 2 个 sample |
| `test_white_wins_generates_correct_count` | 白胜、history=6 步、获胜方 3 步，剔首 → 2 个 sample |
| `test_each_sample_input_is_per_step` | 每个 sample 的 input 与对应 history[i]["input"] 一致 |
| `test_output_one_hot_at_position` | 每个 sample output 落子位=1.0，其它=0 |
| `test_mask_all_ones` | 所有 sample mask 全 1 |
| `test_skip_draw` | winner=3 整局跳过 |
| `test_skip_no_winner` | winner=0 整局跳过 |
| `test_first_move_is_excluded` | 显式验证 history[0]（黑胜）或 history[1]（白胜）不在样本中 |
| `test_short_game_returns_empty` | 获胜方只有 1 步时剔首后为空 |
| `test_no_augmentation_even_if_enabled` | augment=True 时仍不增强，样本数 = 获胜方步数-1 |
| `test_invalid_position_no_crash` | position 越界时对应 output 全 0 不崩溃 |
| `test_multiple_files` | 多文件合并数量正确 |

### create_samples CLI 集成测试追加

| 用例 | 说明 |
|------|------|
| `test_cli_winner_path_mode` | `--mode winner_path` 端到端贯通，产物 sample 数符合预期且不含增强 |

### 涉及文件（本节追加）

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `design/sample_creator_design.md` | 修改 | 本节（第三十八次追加） |
| `ai/sample_creator.py` | 修改 | mode 白名单扩展 + 新增 `_create_winner_path()` 路径；winner_path 强制禁用增强 |
| `ai/create_samples.py` | 修改 | 更新 `--mode` help 文本、epilog 示例 |
| `tests/ai/test_sample_creator_winner_path.py` | 新增 | winner_path 模式 12 条单元测试 |
| `tests/ai/test_create_samples.py` | 修改 | 补充 1 条 winner_path CLI 集成测试 |
| `README.md` | 修改 | `--mode` 参数说明列出支持的模式；新增 winner_path 示例 |
