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

> **注意**：本节中关于 output 构造的规则（落子位=1.0、其余=0.0 的硬标签；
> position 越界时 output 全 0）已被 **第三十九次追加** 替换，不再生效。
> input / mask / 跳过规则 / D4 禁用等其他规则仍然保留。

### 背景与动机

`winning_move` 模式只保留了每局的"最后一步（制胜那步）"作为 sample。这种策略
让模型只学到"如何把即将成形的五连补上最后一子"，对于整局布局缺乏监督信号。

新模式 `winner_path` 把**整局中获胜方的每一步**都作为 sample，
目的是让模型拟合"获胜方走出的整条路径"，学习整局的布局决策，而不仅仅是封口一步。

### 核心规则

| 规则项 | 取值/说明 |
|--------|-----------|
| 模式名 | `winner_path` |
| 选取范围 | 仅获胜方的落子步（败方的步丢弃） |

| input | 每一步 history[i]["input"]（即落子前的棋盘 226 维向量） |
| output | 该步落子位 = 1.0，其余 225 维位置 = 0.0 |
| mask | 全 1（与 winning_move 一致，已落子位也参与 loss 计算） |
| 数据增强（D4） | **强制不启用**。因为 winner_path 本身已极大膨胀 sample 量（每局产生 k 条，k 为获胜方步数），叠加 8 倍 D4 会导致训练数据体量失控。传入 `augment=True` 时也忽略 |
| 平局（winner=3） | 整局跳过 |
| 未结束（winner=0） | 整局跳过 |
| 空 history | 整局跳过 |
| 获胜方仅 1 步 | 产生 1 个 sample |
| position 越界 | 该 sample 的 output 保持全 0，不崩溃 |

### 获胜方步序的识别

项目惯例为"黑先行、双方交替落子"，history 中：
- `history[0]` = 黑第 1 步，`history[1]` = 白第 1 步，`history[2]` = 黑第 2 步，……
- 因此：
    - winner=1（黑胜）→ 获胜方步索引集 = `[0, 2, 4, ...]`
    - winner=2（白胜）→ 获胜方步索引集 = `[1, 3, 5, ...]`
- 因此获胜方步索引集即为：
    - winner=1 → `[0, 2, 4, 6, ...]`
    - winner=2 → `[1, 3, 5, 7, ...]`

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

## 第三十九次追加：winner_path 软标签化（替换硬标签）

### 背景与动机

第三十八次追加中，`winner_path` 每步 output 采用硬标签：落子位=1.0、其他=0.0。
该策略让模型把获胜方每一步都当作"唯一正确落点"学习，信号过于刚性，容易
破坏模型已习得的合理分布。

本次修订将 output 构造由硬标签改为**基于模型自身输出的软标签**，对落子位
做**最小增强**使其成为严格 top1，其余增量由非落子位**等额汲取 + clip + 归一化**
吸收。本次修订**直接替换**第三十八次追加的 output 行为；硬标签语义不再保留。

### 核心规则

| 规则项 | 取值/说明 |
|--------|-----------|
| 适用模式 | 仅 `winner_path` |
| input / mask | 不变（input = `history[i]["input"]`；mask = 全 1）|
| output 数据源 | `history[i]["output"]`（DataRecorder 已记录的当步模型 softmax 输出，225 维）|
| 落子位已严格 top1 | 保持不变，output = 模型输出的拷贝 |
| 落子位非严格 top1 | 按下方算法抬升至严格 top1，其余位置均摊汲取、clip 到 0、重归一化 |
| EPS（超越阈值）| `1e-3`，作为模块级常量 `WINNER_PATH_TOP1_EPS` 暴露于 `ai/sample_creator.py` |
| D4 增强 | 仍然强制禁用（与第三十八次追加一致）|
| 跳过条件 | 平局/未结束/空 history 仍然跳过 |
| position 越界兜底 | output 直接等于模型原分布（替换第三十八次追加的"全 0"行为）|
| 输出总和 | 归一化后 `sum(output) == 1`，与 softmax 分布同量纲 |

### 主流程伪代码

```
对 winner_path 每个入选步 i：
    p   = list(history[i]["output"])                # 长度 225，sum≈1
    idx = pos[0] * 15 + pos[1]

    if not 0 <= idx < 225:
        output = p                                  # 越界兜底：原分布
    else:
        p_move = p[idx]
        p_max  = max(p)
        strict_top1_at_idx = (p_move == p_max) and (p.count(p_max) == 1)

        if strict_top1_at_idx:
            output = p                              # 已严格 top1，原样使用
        else:
            delta     = p_max - p_move + EPS        # EPS = 1e-3
            per_share = delta / 224
            q = [p[j] - per_share for j in 0..224]  # 其余均摊汲取
            q[idx] = p_move + delta                 # 抬升落子位
            q = [max(v, 0.0) for v in q]            # clip 到 0（防负）
            s = sum(q)                              # s 恒 > 0（q[idx] > 0 已保证）
            output = [v / s for v in q]             # 归一化到 sum=1

    mask = [1]*225
    samples.append({"input": history[i]["input"], "output": output, "mask": mask})
```

### 归一化正确性说明（关键）

1. 均摊扣减后理论 `sum = 1`；但 clip 负数归 0 会让 `sum ≥ 1`。归一化让总和回到 1，
   与模型 softmax 输出同量纲。
2. 归一化后落子位仍严格 top1，可证明：clip 前 `q[idx] = p_max + EPS`；其他位置
   `q[j] ≤ p[j] ≤ p_max < q[idx]`。clip 只会把某些 `q[j]` 进一步降到 0，不会
   抬高任何位置；同除以正 `s` 后不等式方向不变，故落子位在归一化后仍是严格最大。

### API 变更

仅影响 `SampleCreator` 在 `winner_path` 模式下的 output 构造，
**不新增公开参数，不新增模式**。
`SUPPORTED_MODES`、`__init__` 签名、`create()` / `create_from_round_dir()` 签名均不变。

新增模块级常量：

```python
# ai/sample_creator.py
WINNER_PATH_TOP1_EPS = 1e-3
```

### 与 record.json 的兼容性

`history[i]["output"]` 自 `DataRecorder.record_move` 起即写入，现有所有
`record.json` 均包含该字段。若未来出现缺失 `output` 字段的记录，视为非法输入
直接抛出 `KeyError`（不回退到硬标签，避免静默退化）。

### 新增测试用例（winner_path 软标签）

测试追加至 `tests/ai/test_sample_creator_winner_path.py`。同时**删除**第三十八次
追加节中以下两条已失效的用例：
- `test_output_one_hot_at_position`（one-hot 语义已不存在）
- `test_invalid_position_no_crash`（越界兜底从"全 0"改为"原分布"）

| 用例 | 说明 |
|------|------|
| `test_soft_label_when_move_already_strict_top1` | 构造 record 使落子位概率严格最大 → output 与 `history[i]["output"]` 完全一致 |
| `test_soft_label_boosts_move_to_strict_top1` | 落子位非 top1 → 生成 output 中落子位为严格最大值（> 其他所有位置）|
| `test_soft_label_sum_is_one` | 生成 output 的 `sum` 与 1.0 的差值 < 1e-9 |
| `test_soft_label_no_negative` | 生成 output 所有元素 ≥ 0 |
| `test_soft_label_uses_recorded_output` | Mock record 中的 `output` 为特定分布 → 验证 SampleCreator 读取该字段而非模型重新推理 |
| `test_soft_label_tie_top1_is_boosted` | 落子位与其他位置并列 top1 → 仍触发增强，使落子位成为严格 top1 |
| `test_soft_label_clip_extreme_distribution` | 极不均分布（某位置 0.95，其余接近 0）→ 汲取后多数位置本应为负，clip+归一化后 output 非负且 `sum=1`，落子位仍严格 top1 |
| `test_soft_label_eps_respected` | 显式断言归一化前 `q[idx] - 其他最大值 ≥ EPS` |
| `test_winner_path_no_longer_one_hot` | 回归断言：output **不**是 one-hot（存在至少两个 > 0 的元素，且落子位 ≠ 1.0）|
| `test_invalid_position_returns_model_output` | position 越界时 output 直接等于模型原分布（替换旧的"全 0"期望）|

### 与第三十八次追加的差异回溯

| 项 | 第三十八次（原）| 第三十九次（本次）|
|---|---|---|
| output 构造 | 落子位=1，其余=0（硬标签 / one-hot）| 基于模型输出的软标签 + 抬升到严格 top1 + 归一化 |
| position 越界兜底 | output 全 0 | output = 模型原分布 |
| record.json 依赖 | 仅读 `input` / `position` | 还读 `history[i]["output"]`（缺失则 KeyError）|
| mask | 全 1 | 全 1（不变）|
| D4 增强 | 强制禁用 | 强制禁用（不变）|
| 跳过规则 | 不变 | 不变 |

### 涉及文件（本节追加）

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `design/sample_creator_design.md` | 修改 | 追加本节；在第三十八次追加节开头加替换引用 |
| `ai/sample_creator.py` | 修改 | `_build_winner_path_samples` output 构造改为软标签算法；新增常量 `WINNER_PATH_TOP1_EPS = 1e-3` |
| `tests/ai/test_sample_creator_winner_path.py` | 修改 | 删除 2 条失效用例；新增 10 条软标签测试 |
| `requirements/key_points.md` | 已追加 | 第三十九次追加（用户本次提交的需求）|

## 第四十次追加：移除 winner_path 剔首逻辑

### 背景与动机

第三十八次追加中引入了"剔除获胜方第 1 步"的规则，理由是"开局第 1 子几乎无
战术信息，且经增强后会把对称位置反复强化，引入位置偏置"。但实践证明这是
**错误的**：

1. **开局步有战略价值**。例如第一步下在天元的胜率最高。如果排除开局步，
   模型永远学不到开局怎么下，这是致命的训练盲区。
2. 随着软标签化（第三十九次追加），剔首的原始担忧（位置偏置）已经大幅减弱。
3. D4 增强在 `winner_path` 下本就强制禁用，"8 倍增强放大偏置"的风险不存在。

因此，本次修订**移除 winner_path 的剔首逻辑**，获胜方的每一步（包括开局首步）
都纳入样本。

### 核心规则变更

| 规则项 | 第三十八/三十九次（旧）| 第四十次（新）|
|--------|----------------------|-------------|
| 选取范围 | 获胜方的第 2 步起 | 获胜方**全部**步 |
| 每局 sample 数 | 获胜方步数 − 1 | 获胜方步数 |
| 获胜方仅 1 步 | 产生 0 个 sample | 产生 1 个 sample |
| 获胜方步索引（黑胜）| `[2, 4, 6, ...]` | `[0, 2, 4, 6, ...]` |
| 获胜方步索引（白胜）| `[3, 5, 7, ...]` | `[1, 3, 5, 7, ...]` |
| 跳过规则 | 不变 | 不变（平局/未结束/空 history 仍跳过）|
| D4 增强 | 不变 | 不变（仍强制禁用）|
| output 构造 | 不变 | 不变（软标签算法不变）|

### API 变更

无新增参数/模式。`SampleCreator` 签名不变，仅 `_build_winner_path_samples`
内部移除 `winner_indices[1:]` 切片。

### 涉及文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `design/sample_creator_design.md` | 修改 | 第三十八/三十九次追加节去除剔首引用；追加本节 |
| `ai/sample_creator.py` | 修改 | `_build_winner_path_samples` 移除剔首切片；更新 docstring |
| `ai/create_samples.py` | 修改 | 更新 `--mode` help 文本和 epilog 示例（移除"剔除开局"字样）|
| `tests/ai/test_sample_creator_winner_path.py` | 修改 | sample 计数预期 N−1 → N；反转 `TestWinnerPathFirstMoveExcluded` 语义；更新短对局用例 |
| `tests/ai/test_sample_creator_winner_path_soft_label.py` | 修改 | 更新辅助函数适配不再剔首 |
| `tests/ai/test_create_samples.py` | 修改 | 更新 winner_path CLI 测试的样本数预期 |
| `README.md` | 修改 | 移除"剔除开局"描述 |
| `requirements/key_points.md` | 已追加 | 第四十次追加（用户本次提交的需求）|
