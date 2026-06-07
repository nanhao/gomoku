# PRD：battle.py 支持温度采样参数

## 1. 元信息

| 字段 | 值 |
|------|-----|
| **意图编号** | intention_id=8 |
| **创建日期** | 2026-06-07 |
| **状态** | 待实现 |
| **关联意图** | 无 |

## 2. 目标与背景

### 2.1 目标

`ai/battle.py` 命令行脚本新增温度采样参数，允许用户在启动 AI 对战时控制开局阶段的温度采样行为，与 `create_samples.py` 和 GUI 对战入口保持功能一致。

### 2.2 背景

当前项目中三个 AI 对战入口对温度采样的支持情况：

| 入口 | 温度采样支持 | 默认行为 |
|------|-------------|---------|
| `ai/create_samples.py`（样本生成 CLI） | ✅ 完整 CLI 参数 | 默认启用（T=1.0, moves=12, both） |
| `gui/ai_battle.py`（GUI 对战） | ✅ 硬编码启用 | 默认启用（T=1.0, moves=12, both） |
| **`ai/battle.py`（对战 CLI）** | **❌ 不支持** | **全程 argmax，无法开启温度采样** |

后端基础设施早已就绪：`TempConfig` 数据类、`AIGameManager.start_ai_vs_ai_random()` 的参数签名均已支持 `temp_config`。`battle.py` 是唯一缺失 CLI 参数串联的入口。

温度采样是 AI 自对弈探索的核心机制——在对局前 N 步按温度加权随机采样落子，之后切换为确定性 argmax，能在保持对局质量的同时增加开局多样性。用户在命令行评估模型对战能力时，应有能力控制这一行为。

### 2.3 适用场景

- 用户想测试两个模型在"有探索性"环境下的对战表现（开启温度采样）
- 用户想测试两个模型在纯确定性环境下的对战表现（关闭温度采样，即 argmax）
- 用户想调整温度值来观察不同随机程度对胜率的影响
- 用户想复现某次对战结果（通过 `--temp-seed` 固定随机种子）

## 3. 范围边界

### 3.1 包含范围

| 范围 | 说明 |
|------|------|
| `ai/battle.py` 新增 5 个 CLI 参数 | `--no-temp-sampling`、`--temperature`、`--temp-moves`、`--temp-side`、`--temp-seed` |
| 参数校验 | 非法值（如负温度、非法 temp-side）时报错退出 |
| 构造 TempConfig 并透传 | 将解析后的参数构造为 `TempConfig` 对象，传递给 `AIGameManager.start_ai_vs_ai_random()` |
| 命令行帮助信息更新 | `--help` 输出中显示新增参数及其说明 |
| 默认启用温度采样 | 不传任何温度参数时，默认以 `temperature=1.0, temp_moves=12, temp_side=both` 启用温度采样 |

### 3.2 不包含范围

| 范围 | 说明 |
|------|------|
| `top_k` 参数 | `TempConfig` 虽支持 `top_k` 字段，但本次不暴露为 CLI 参数（与 `create_samples.py` 保持一致） |
| 修改 `AIGameManager` 或 `TempConfig` | 后端代码已就绪，无需改动 |
| `app/battle.py`（BattleSession） | 已支持 `temp_config`，无需改动 |
| `gui/ai_battle.py` | 已硬编码启用温度采样，无需改动 |
| 其他脚本 | 本次仅覆盖 `ai/battle.py` |

## 4. 功能说明

### 4.1 新增参数一览

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--no-temp-sampling` | flag（布尔开关） | 否 | 未设置（即启用） | 禁用开局阶段温度采样，全程走 argmax |
| `--temperature` | `float` | 否 | `1.0` | 温度值 T（必须 > 0），越大越随机；趋近 0 时等价 argmax |
| `--temp-moves` | `int` | 否 | `12` | 前 N 个全局步（黑白合计）启用温度采样，之后切换为 argmax。设为 0 等价于关闭温度采样 |
| `--temp-side` | `str` | 否 | `both` | 启用温度采样的一方，可选值：`both`（双方）、`black`（黑方）、`white`（白方）、`random`（每局随机抽一方） |
| `--temp-seed` | `int` | 否 | `None` | 温度采样随机种子，用于复现对战结果。不指定时每局使用不同随机序列 |

### 4.2 参数行为详述

#### `--no-temp-sampling`

- 不传此 flag 时：默认启用温度采样，使用各参数的默认值
- 传入此 flag 时：禁用温度采样，`--temperature` / `--temp-moves` / `--temp-side` / `--temp-seed` 被忽略，全程走 argmax

#### `--temperature`

- 必须 > 0
- T = 1.0：模型原始概率分布不做变换
- T < 1.0：放大高概率动作的权重，行为趋近 argmax
- T > 1.0：压缩概率差异，行为更随机

#### `--temp-moves`

- 必须 ≥ 0
- 对局前 `temp_moves` 步（黑白合计步数）启用温度采样
- 第 `temp_moves` 步及之后切换为 argmax
- 设为 0 时全程 argmax（等价于 `--no-temp-sampling`）

#### `--temp-side`

- 必须为 `both`、`black`、`white`、`random` 之一
- `both`：双方在前 N 步均使用温度采样
- `black`：仅黑方在前 N 步使用温度采样，白方全程 argmax
- `white`：仅白方在前 N 步使用温度采样，黑方全程 argmax
- `random`：每局开始前随机选择黑方或白方之一启用温度采样，另一方全程 argmax

#### `--temp-seed`

- 可选参数，不传时每局的随机序列不同（不可复现）
- 传入后，同一 seed 值 + 相同对局参数可复现完全一致的落子序列

### 4.3 参数交互规则

| 场景 | 预期行为 |
|------|---------|
| 不传任何温度参数 | 默认启用温度采样（T=1.0, moves=12, both） |
| 仅传 `--temperature 0.5` | 启用温度采样，temperature=0.5，其余使用默认值 |
| 传 `--no-temp-sampling` | 禁用温度采样，忽略其他温度参数值 |
| 传 `--temp-moves 0` | 禁用温度采样（等价于 `--no-temp-sampling`） |
| 传 `--temperature 0` | 报错退出（温度必须 > 0） |
| 传 `--temperature -1` | 报错退出（温度必须 > 0） |
| 传 `--temp-side invalid` | 报错退出（不在可选值列表中） |
| 传 `--temp-moves -5` | 报错退出（必须 ≥ 0） |

### 4.4 命令行示例

```bash
# 默认行为：启用温度采样（T=1.0, moves=12, both）
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50

# 禁用温度采样，全程 argmax（与当前默认行为一致）
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50 --no-temp-sampling

# 自定义温度值：更随机的对战
python ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 20 --temperature 2.0

# 自定义温度步数：前 6 步温度采样
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 30 --temp-moves 6

# 仅黑方启用温度采样
python ai/battle.py --ai1 cnn_v3.1 --ai2 random --games 50 --temp-side black

# 每局随机一方启用
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50 --temp-side random

# 固定随机种子：复现对战结果
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 10 --temp-seed 42

# 组合使用
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 100 --temperature 0.8 --temp-moves 8 --temp-side random --temp-seed 2026
```

## 5. 验收步骤

### AC-1：参数解析验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-1.1 | 不传任何温度参数 | `python ai/battle.py --ai1 random --ai2 random --games 1` | 对战正常运行，内部使用默认 TempConfig（T=1.0, moves=12, both） |
| AC-1.2 | 传入 `--temperature` | `--temperature 2.0` | 对战正常运行，温度值为 2.0 |
| AC-1.3 | 传入 `--temp-moves` | `--temp-moves 6` | 对战正常运行，采样步数为 6 |
| AC-1.4 | 传入 `--temp-side` | `--temp-side black` | 对战正常运行，仅黑方采样 |
| AC-1.5 | 传入 `--temp-side random` | `--temp-side random` | 对战正常运行，每局随机决定采样方 |
| AC-1.6 | 传入 `--temp-seed` | `--temp-seed 42` | 对战正常运行，结果可复现 |
| AC-1.7 | 传入 `--no-temp-sampling` | `--no-temp-sampling` | 对战正常运行，全程 argmax |
| AC-1.8 | 组合多个温度参数 | `--temperature 0.5 --temp-moves 8 --temp-side white` | 对战正常运行，所有参数生效 |
| AC-1.9 | `--temperature 0` | `--temperature 0` | 程序报错退出（ValueError），提示温度必须 > 0 |
| AC-1.10 | `--temperature -1` | `--temperature -1` | 程序报错退出（ValueError），提示温度必须 > 0 |
| AC-1.11 | `--temp-moves -5` | `--temp-moves -5` | 程序报错退出（ValueError），提示必须 ≥ 0 |
| AC-1.12 | `--temp-side invalid` | `--temp-side abc` | argparse 报错退出，提示不在合法值列表中 |
| AC-1.13 | `--help` 输出包含新参数 | `python ai/battle.py --help` | 输出中包含全部 5 个新参数的说明文字 |

### AC-2：默认行为验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-2.1 | 默认启用温度采样 | 不传任何温度参数，运行 1 局对战 | AIGameManager 收到的 temp_config 不为 None，且字段为 T=1.0, moves=12, side=both |
| AC-2.2 | 默认采样行为 | 不传任何温度参数，运行多局对战 | 至少有一局的前几步使用了 `strategy=sampling` 而非 `strategy=argmax` |

### AC-3：关闭温度采样验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-3.1 | `--no-temp-sampling` 关闭 | `--no-temp-sampling`，运行 1 局 | AIGameManager 收到的 temp_config 为 None，全程 argmax |
| AC-3.2 | `--temp-moves 0` 关闭 | `--temp-moves 0`，运行 1 局 | 效果等价于 `--no-temp-sampling`，全程 argmax |
| AC-3.3 | 关闭时温度参数被忽略 | `--no-temp-sampling --temperature 2.0` | 不报错，但温度值不生效（全程 argmax） |

### AC-4：向后兼容验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-4.1 | 现有调用方式不报错 | `python ai/battle.py --ai1 random --ai2 random --games 1` | 正常运行，输出对战结果 |
| AC-4.2 | 现有参数不受影响 | `--ai1` / `--ai2` / `--games` 参数行为不变 | 与改动前行为一致 |
| AC-4.3 | 输出格式不变 | 任意合法输入 | 对战结果输出格式不变（包含每局结果和统计摘要） |
| AC-4.4 | intention_id=6 的调用兼容 | `python ai/battle.py --ai1 cnn_v3.4 --ai2 cnn_v3.5 --games 100` | 不传温度参数时正常运行（按默认温度采样行为执行，对战仍能正常完成并输出胜率） |

### AC-5：温度采样端到端验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-5.1 | 温度值影响落子分布 | 分别以 `--temperature 0.1` 和 `--temperature 5.0` 运行多局 | T=0.1 时落子趋近 argmax（确定性高）；T=5.0 时落子更分散（多样性高） |
| AC-5.2 | temp_moves 切换 | `--temp-moves 4`，运行对局 | 前 4 步使用温度采样，第 5 步起使用 argmax |
| AC-5.3 | temp-side=black | `--temp-side black`，运行对局 | 仅黑方前 N 步使用温度采样，白方始终 argmax |
| AC-5.4 | temp-side=white | `--temp-side white`，运行对局 | 仅白方前 N 步使用温度采样，黑方始终 argmax |
| AC-5.5 | temp-side=random | `--temp-side random`，运行多局 | 每局随机由黑或白一方启用，另一方全程 argmax；多局中两种情况的出现频率趋于均等 |
| AC-5.6 | temp-seed 可复现 | 同一 seed 运行两次 | 两次输出完全一致（每局落子序列相同、胜负结果相同） |
| AC-5.7 | 不指定 temp-seed 不可复现 | 不传 `--temp-seed`，运行两次 | 两次输出大概率不同（随机性导致落子序列差异） |

### AC-6：边缘场景验证

| ID | 验收场景 | 输入 | 预期结果 |
|----|---------|------|---------|
| AC-6.1 | `--temperature` 极小正数（如 0.0001） | `--temperature 0.0001` | 不报错，行为几乎等价于 argmax |
| AC-6.2 | `--temp-moves` 极大值（如 100，超过最大步数 225） | `--temp-moves 100` | 不报错，整局均使用温度采样（因为对局通常在 100 步内结束） |
| AC-6.3 | `--temperature` 极大值（如 100.0） | `--temperature 100.0` | 不报错，落子趋近均匀随机 |
| AC-6.4 | 同时传 `--no-temp-sampling` 和 `--temp-moves 0` | `--no-temp-sampling --temp-moves 0` | 不报错，禁用温度采样 |

## 6. 术语定义

| 术语 | 定义 |
|------|------|
| 温度采样（Temperature Sampling） | 对模型输出的概率分布做 `p^(1/T)` 变换后加权随机抽样，T 控制随机程度 |
| argmax | 始终选择模型输出概率最大的合法位置落子（确定性策略） |
| TempConfig | `ai/move_sampler.py` 中定义的温度采样配置数据类，封装 temperature、temp_moves、temp_side、top_k、seed 字段 |
| 全局步数（move_count） | 对局中黑白双方的合计落子数，从 0 起计 |
| temp_side | 温度采样启用方：`both`（双方）、`black`（黑方）、`white`（白方）、`random`（每局随机） |

## 7. 前置依赖

| 依赖项 | 说明 |
|--------|------|
| `ai/move_sampler.py` | 已存在，提供 `TempConfig` 数据类和 `SUPPORTED_TEMP_SIDES` 常量，无需修改 |
| `ai/game_manager.py` | 已存在，`start_ai_vs_ai_random()` 已支持 `temp_config` 参数，无需修改 |
| `ai/battle.py` | 本次修改的目标文件 |

## 8. 修订记录

| 日期 | 修订内容 | 涉及章节 |
|------|---------|---------|
| 2026-06-07 | 初版创建 | — |
