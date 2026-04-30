# 棋型样本生成器（sample_generator）设计

## 背景

`temp/gen_chong4_samples.py` 是一次性脚本，仅生成冲四棋型样本。
需求（key_points.md 第 34 次追加）扩展到**活四、活三、活二**共 4 种进攻棋型，
并将脚本正式化为 `ai/sample_generator.py` 作为长期工具，同时重构验证指标。
在调研 [ai/reward_calculator.py](../ai/reward_calculator.py) 后确认其 reward
取 `max(己方棋型, 对手棋型 × 0.95)`，天然支持"防守位高分"语义，
因此本次一并覆盖 **2 种防守棋型**（`defend_chong4` / `defend_huo3`），共 6 种。

## 设计目标

1. 单次运行**只生成一种**棋型、**写一个**文件，棋型 + 样本数 + 输出路径全部参数化
2. 6 种棋型（4 进攻 + 2 防守）共用统一生成核心，不为每种棋型复制一份
3. 生成阶段直接记录"进攻位/防守位"（统一命名 `attack_positions`），验证阶段直接消费，不再事后 detect
4. 验证指标按每个棋型的 attack_positions 个数动态计算 `Hit@N`，并提供辅助指标辅助诊断
5. `temp/gen_chong4_samples.py` 保留不动（历史一次性脚本）

## 核心概念

### 棋型定义（严格对齐 [ai/shape_detector.py](../ai/shape_detector.py)）

**进攻棋型**（attacker 视角，`current_player` = 持有该棋型的一方）：

| 棋型 key | 连珠数 | 两端开放数 | 进攻位数 N | 进攻位语义 |
|---|---|---|---|---|
| `chong4` | 4 | 1 | 1 | 落子即连五 |
| `huo4` | 4 | 2 | 2 | 两端任一都连五 |
| `huo3` | 3 | 2 | 2 | 两端任一都形成活四 |
| `huo2` | 2 | 2 | 2 | 两端任一都形成活三 |

**防守棋型**（defender 视角，`current_player` = 面对对手威胁的一方）：

| 棋型 key | 对手威胁 | 防守位数 N | 防守位语义 | reward 来源 |
|---|---|---|---|---|
| `defend_chong4` | 对手冲四 | 1 | 堵对手唯一空端 | 对手落此形成**连五**：0.95 × 1e6 ≈ 950k |
| `defend_huo3` | 对手活三 | 2 | 堵活三两端之一 | 对手落此形成**活四**：0.9 × 0.95 × 1e6 ≈ 855k |

**不生成跳棋型**（`_X_XX_` 等），与 shape_detector 判定保持同构。
**不生成 `defend_huo4` / `defend_huo2`**：活四已必败（防守无意义），活二威胁过弱（reward ≈ 475k 区分度不足）。

### 样本 JSON 格式

```json
{
  "input":  [float; 226],          // 225 格 + 1 位当前玩家
  "output": [float; 225],          // masked_softmax 后的概率
  "mask":   [int; 225],            // 1=合法空位, 0=已落子
  "_attack_positions": [int, ...], // 生成器标记的进攻位索引（验证/可视化用）
  "_shape": "huo3"                 // 生成时指定的棋型 key
}
```

- 下划线开头字段为**元数据**，训练代码只读取 `input / output / mask`
- 保留元数据便于事后可视化、问题追踪、样本合并

## API 设计

### 模块：`ai/sample_generator.py`

```python
SUPPORTED_SHAPES = (
    "chong4", "huo4", "huo3", "huo2",          # 进攻
    "defend_chong4", "defend_huo3",            # 防守
)

ATTACK_SHAPES = {"chong4", "huo4", "huo3", "huo2"}
DEFEND_TO_ATTACK = {                           # 防守棋型复用进攻生成器
    "defend_chong4": "chong4",
    "defend_huo3":   "huo3",
}

def generate_shape_board(
    shape: str,
    random_gen: random.Random,
    max_extra: int = 4,
) -> Optional[Tuple[List[int], List[int], int]]:
    """生成一个目标棋型的棋盘

    - 进攻棋型：生成持有该棋型的 attacker 棋盘，current_player = attacker
    - 防守棋型：调用对应进攻棋型的生成器，**仅切换视角**：
                current_player = 原 attacker 的对手（即被威胁方）。
                棋盘保持不变，attack_positions 索引不变
                （原"进攻位"现在是"防守位"）。

    Returns:
        (board, attack_positions, current_player) 或 None（本次生成失败，外层重试）

    Raises:
        ValueError: shape 不在 SUPPORTED_SHAPES 内
    """


def create_sample(
    board: List[int],
    attack_positions: List[int],
    current_player: int,
    shape: str,
    reward_calculator: RewardCalculator,
) -> Dict:
    """将棋盘状态封装为训练样本（含 _attack_positions / _shape 元数据）"""


def verify_samples(samples: List[Dict]) -> Dict:
    """对样本做标签质量体检

    Returns:
        {
            "total":              int,
            "hit_at_N":           int,   # 所有进攻位都在 output top-N
            "hit_at_N_plus_1":    int,
            "hit_at_N_plus_2":    int,
            "attack_prob_sum":    {"min": f, "avg": f, "max": f},
            "prob_gap":           {"min": f, "avg": f, "max": f},  # 进攻位均值 - 最高非进攻位
            "failed_indices":     List[int],  # Hit@N 未命中的样本索引（最多前 100 个）
        }
    """
```

### CLI：`python -m ai.sample_generator`

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--shape` | str | 必填 | `chong4` / `huo4` / `huo3` / `huo2` / `defend_chong4` / `defend_huo3` |
| `--num-samples` | int | 必填 | 目标样本数 |
| `--output` | str | 必填 | 输出 json 绝对或相对路径 |
| `--seed` | int | `42` | 随机种子 |
| `--max-extra` | int | `4` | 每样本最多放几颗干扰子 |
| `--print-failures` | int | `5` | 验证未命中样本打印条数（0 = 不打印） |

使用示例：
```
python -m ai.sample_generator --shape huo3 --num-samples 40960 \
    --output ai/merge_samples/huo3.json
```

## 主流程伪代码

```python
def main():
    args = parse_args()
    random_gen = random.Random(args.seed)
    reward_calc = RewardCalculator(RewardConfig())

    samples = []
    while len(samples) < args.num_samples:
        result = generate_shape_board(args.shape, random_gen, args.max_extra)
        if result is None:
            continue                                   # 本次未生成有效棋盘，重试
        board, attack_positions, player = result

        # 生成阶段的合法性守卫：避免干扰子产生更强棋型污染 label
        if _has_stronger_conflict(board, player, args.shape):
            continue

        samples.append(create_sample(
            board, attack_positions, player, args.shape, reward_calc))

    report = verify_samples(samples)
    print_report(report, print_failures=args.print_failures)

    save_json(args.output, samples)
```

### generate_shape_board 内部流程

```
若 shape ∈ DEFEND_TO_ATTACK:
    # 防守棋型：委托给进攻生成器，仅切换视角（不翻转棋盘）
    #
    # 关键语义：
    # - 进攻生成器返回的棋盘上，attacker 色形成棋型
    # - 防守样本 = 同一棋盘换视角：current_player = attacker 的对手（被威胁方）
    # - attack_positions 索引不变（原进攻位 = 现防守位）
    #
    # 注：不做颜色翻转。若既翻转颜色又把 current_player 设成 3-attacker，
    # 会让 current_player 与棋型持有方重合 → 样本语义退化成进攻样本。
    attack_shape = DEFEND_TO_ATTACK[shape]
    result = generate_shape_board(attack_shape, random_gen, max_extra)
    若 result is None: return None
    board, attack_positions, attacker_player = result
    defender_player = 3 - attacker_player
    return (board, attack_positions, defender_player)

# 以下是进攻棋型主路径
1. 参数映射：shape → (stone_count, required_open_ends)
2. 循环尝试最多 300 次：
   a. 随机方向 (dx, dy) 和起点 (sx, sy)
   b. 沿方向放置 stone_count 颗同色棋子，越界 → 重试
   c. 计算两端坐标，检查各端「边界 or 对手棋子 or 空位」
   d. 若「开放端数」达不到 required_open_ends → 重试
   e. 根据 required_open_ends 放置阻挡棋子：
      - chong4: 两端都在棋盘内时用对手子堵一端（随机选）
      - 其他: 两端都不能被堵
   f. 收集 attack_positions = 所有开放端索引
   g. 放置 0..max_extra 颗干扰子，避开 protected 集合（进攻位 + 连珠 + 阻挡位）
   h. 用 detect_shapes 校验每个 attack_position 落子后是否形成期望棋型：
      - chong4/huo4 → 连五
      - huo3 → 活四
      - huo2 → 活三
   i. 校验通过 → return (board, attack_positions, player)
3. 全部失败 → return None
```

## 验证指标设计

### 主指标：Hit@N

设样本的进攻位数为 `N = len(attack_positions)`。取 output 的 top-N 索引集合 `topN`，
若 `set(attack_positions) ⊆ topN` 则计一次命中。

### 辅助指标

| 指标 | 公式 | 解读 |
|---|---|---|
| `Hit@N+1` / `Hit@N+2` | 放宽到 top-(N+1/N+2) | 诊断"差几名" |
| `attack_prob_sum` | ∑ output[i] for i ∈ attack_positions | 理想接近 1 |
| `prob_gap` | mean(output[attack]) − max(output[非 attack 合法位]) | 区分度；越大 label 越尖锐 |
| `failed_indices` | Hit@N 未命中样本索引（截断至 100） | 人工复盘用 |

### 失败样例输出

当 `--print-failures > 0` 时，对前 k 个失败样本打印：
- 棋盘 ASCII 图
- `attack_positions` 与 `output` 的 top-5 索引/概率
- 期望棋型 vs 实际 top-1 所在位置的 detect_shapes 结果

## 主流程测试用例

| # | 场景 | 输入 | 期望 |
|---|---|---|---|
| T1 | 冲四生成合法性 | `shape="chong4"`, 10 个样本 | 每个样本 attack_positions 长度 = 1；该位置落子 detect 得"连五" |
| T2 | 活四生成合法性 | `shape="huo4"`, 10 个样本 | 每个样本 attack_positions 长度 = 2；两个位置落子均得"连五" |
| T3 | 活三生成合法性 | `shape="huo3"`, 10 个样本 | 长度 = 2；两个位置落子 detect 都含"活四" |
| T4 | 活二生成合法性 | `shape="huo2"`, 10 个样本 | 长度 = 2；两个位置落子 detect 都含"活三" |
| T5 | 样本 mask 正确性 | 任一棋型 1 个样本 | mask[i] == 0 当且仅当 board[i] != 0 |
| T6 | 样本 output 归一化 | 任一棋型 1 个样本 | `sum(output) ≈ 1.0`；`output[i] == 0` 当 mask[i] == 0 |
| T7 | 未知 shape | `shape="xxx"` | 抛 ValueError |
| T8 | verify_samples 主路径 | 构造 2 个必命中 + 1 个必漏 | `hit_at_N == 2`，failed_indices 包含漏的那个 |
| T9 | 干扰冲突过滤 | `max_extra=0`（禁用干扰子） | 生成速度 ≥ 启用干扰子；且每个样本 detect 都严格只有目标棋型 |
| T10 | CLI 主流程 smoke test | `--shape huo3 --num-samples 20 --output tmp.json` | 退出码 0；tmp.json 含 20 条样本；每条都有 `_attack_positions` |
| T11 | 防守冲四生成 | `shape="defend_chong4"`, 5 个样本 | attack_positions 长度 = 1；**对手**落子该位置 detect 得"连五"；current_player 是**被冲四方** |
| T12 | 防守活三生成 | `shape="defend_huo3"`, 5 个样本 | 长度 = 2；**对手**落子两端 detect 都含"活四"；current_player 是**被活三方** |
| T13 | 防守样本 label 尖锐度 | `defend_chong4` 样本 | `output[attack_pos]` 在 top-1，且 `attack_prob_sum > 0.5` |

测试文件：`tests/ai/test_sample_generator.py`（新建）

## 与现有模块关系

| 模块 | 关系 |
|---|---|
| `ai/reward_calculator.py` | 计算每个空位 raw_reward（依赖，不修改） |
| `ai/shape_detector.py` | 生成合法性校验 + 测试断言（依赖，不修改） |
| `ai/probability_utils.py` | `masked_softmax` 计算 output（依赖，不修改） |
| `ai/sample_creator.py` | 完全独立，互不调用（语义：sample_creator 从对局 record 生产样本，sample_generator 从棋型合成样本） |
| `temp/gen_chong4_samples.py` | 保留不动，不迁移功能，避免影响 git 历史/临时脚本语义 |

## 不纳入本次范围

- 跳棋型（`_X_XX_`、`_X__X_` 等）样本生成 —— 前置依赖：shape_detector + RewardCalculator 需先支持跳棋型识别，属独立项目
- 防守活四 / 防守活二样本 —— 活四已必败（防守无意义），活二威胁弱（label 不尖锐）
- 样本合并到既有 `chong4.json` 的自动化
- 与训练入口的集成（生成完手动 `--train-from-dir`）
