# 设计文档: create_samples 新增 --sample_games 参数

## 1. 背景

当前 `ai/create_samples.py` 使用 `--games N` 指定固定对局数，经 `should_keep_game` 过滤后可用样本局数不确定。PRD 要求新增 `--sample_games` 参数，让用户直接指定期望的可用样本局数，脚本持续对弈直到达标。

## 2. 设计目标

1. `--sample_games` 与 `--games` 互斥，不可同时指定
2. 逐局对弈并实时判定样本局计数，达标即停
3. 10% 间隔进度输出（含样本局数/目标/百分比/已对弈数）
4. Ctrl+C 优雅退出，保存已收集的部分数据
5. 不修改 `AIGameManager`、`SampleCreator`、`should_keep_game` 的接口和行为

## 3. 设计决策

### D1: 对弈循环方式

**方案**: 在 `create_samples.py` 内部循环调用 `AIGameManager._play_single_game`，逐局执行并判定。

**理由**:
- `_play_single_game` 是静态方法，直接可调用
- 逐局执行可实现精确的样本局计数和进度输出
- 不需要修改 `AIGameManager` 的任何接口
- `--games` 模式保持原有 `start_ai_vs_ai` 调用不变

**补充说明**: 同一 `AIEngine` 实例同时作为 engine_black 和 engine_white 传入，与现有 `start_ai_vs_ai(engine, engine, ...)` 行为一致（自我对弈语义）。

### D2: 样本局判定逻辑

样本局计数判定与 `SampleCreator.create` 的过滤逻辑保持一致，即同时满足：

1. `winner ∈ {1, 2}`（非平局、非未结束）
2. `should_keep_game({"winner": w, "temp_side_this_game": ts})` 返回 True

**理由**: `SampleCreator.create` 先跳过 `winner == 0 or winner == 3`，再调用 `record_filter`。此处复用同一判定链，确保计数与最终样本生成的一致性。

**依赖假设**: `SampleCreator.create` 还会跳过 `history` 为空的对局。`is_sample_game` 未覆盖此条件，因为 `_play_single_game` 保证每局至少记录一步数据，`history` 不可能为空。此假设成立时，计数与样本生成完全一致。

### D3: 进度输出阈值

使用整数百分比判定：`sample_count * 100 // target` 跨越 10 的整数倍时触发输出。触发判定逻辑在调用方（主循环），而非 `print_sample_progress` 函数内部。

**小值保护**: 当 `target` 较小（如 5）时，至少在达标时打印 100% 进度。

**边界说明**: 每次循环增量为一局，百分比增量 ≤ 100/target，对 target ≥ 10 的场景不会跳过中间里程碑。target < 10 时可能跳过部分 10% 整数倍，但达标时必定打印。

### D4: Ctrl+C 处理

在游戏循环外层包裹 `try/except KeyboardInterrupt`，捕获后：
1. 停止后续对弈
2. 打印中断统计
3. 正常走后续的 SampleCreator 样本生成 + 文件写入流程

**理由**: DataRecorder 已按局保存数据，中断后 `round_dir` 下已有完整的对局记录，SampleCreator 可正常从这些记录生成样本。

**补充说明**: 中断后 `results` 列表包含已完成的对弈结果，后续统计输出（黑胜/白胜/平局计数、过滤统计）均基于已有 `results` 正常计算。

## 4. 接口设计

### 4.1 CLI 参数变更

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--sample_games` | int (>0) | 无 | 否 | 目标样本局数，与 `--games` 互斥 |

互斥规则:
- `--games` + `--sample_games` 同时指定 → 报错退出（exit code 1）
- 两者都不指定 → 默认 `--games 100`
- 仅 `--sample_games` → 样本局驱动模式
- 仅 `--games` → 现有行为不变

### 4.2 create_samples.py 内部新增函数

#### `is_sample_game(result: dict) -> bool`

判定单局对弈结果是否为样本局。

- 输入: `{"winner": int, "temp_side_this_game": str|None}`
- 逻辑: `winner ∈ {1,2}` 且 `should_keep_game(result)` 为 True
- 输出: True/False

#### `print_sample_progress(sample_count: int, target: int, total_games: int) -> None`

打印样本局收集进度。纯输出函数，不负责触发判定（触发判定在调用方主循环中）。

- 输入: 当前样本局数、目标样本局数、已对弈总数
- 输出格式: `进度: 样本局 {sample_count}/{target} ({pct}%), 已对弈 {total_games} 局`

### 4.3 results 列表类型与结构

无论 `--games` 还是 `--sample_games` 模式，`results` 均为 `List[dict]`，每项结构：

```
{"game": int, "winner": int, "temp_side_this_game": str|None}
```

`--sample_games` 模式由 `_play_single_game` 返回 `SingleGameResult`(NamedTuple)，经手动构造为 dict 后 append；`--games` 模式由 `start_ai_vs_ai` 直接返回 dict 列表。两者类型一致，后续统计代码无需区分模式。

### 4.4 main() 函数变更

`--sample_games` 模式下，`main()` 的对弈阶段替换为逐局循环，其余阶段（样本生成、文件写入、统计输出）不变。

## 5. 主流程伪代码

```
function main():
    args = parse_args()

    # ---- 互斥校验 ----
    if args.games is set AND args.sample_games is set:
        print("错误: --games 和 --sample_games 不能同时指定")
        exit(1)

    # ---- 确定运行模式 ----
    use_sample_mode = (args.sample_games is not None)

    # ---- 加载模型、创建引擎、创建记录器（与现有逻辑相同）----
    model = get_model(args.model)
    engine = AIEngine(model)
    recorder = DataRecorder(ai_name=...)

    # ---- 执行对弈 ----
    if use_sample_mode:
        target = args.sample_games
        results = []
        sample_count = 0
        total_games = 0
        last_pct = 0

        try:
            while sample_count < target:
                total_games += 1
                result = AIGameManager._play_single_game(
                    engine, engine, recorder, total_games,
                    name_prefix="ai_vs_ai", temp_config=temp_config
                )
                result_dict = {
                    "game": total_games,
                    "winner": result.winner,
                    "temp_side_this_game": result.temp_side_this_game
                }
                results.append(result_dict)

                if is_sample_game(result_dict):
                    sample_count += 1
                    pct = sample_count * 100 // target
                    if pct >= last_pct + 10 or sample_count == target:
                        print_sample_progress(sample_count, target, total_games)
                        last_pct = (pct // 10) * 10

        except KeyboardInterrupt:
            print("\n[中断] 检测到 Ctrl+C，正在整理已有数据...")
            print(f"[中断] 已收集样本局 {sample_count}/{target} ({sample_count*100//target}%), 已完成原始对弈 {total_games} 局")

    else:
        # ---- 现有 --games 逻辑，不变 ----
        results = AIGameManager.start_ai_vs_ai(
            engine, engine, args.games, recorder,
            on_progress=on_progress, temp_config=temp_config
        )

    # ---- 后续统计、样本生成、文件写入（与现有逻辑相同）----
    # 额外: --sample_games 模式下，最终输出增加"样本局数"行
```

## 6. 交互图

详见 [sample_games_flow.puml](../sample_games_flow.puml)

## 7. 涉及文件

| 文件 | 操作 | 变更说明 |
|------|------|----------|
| `ai/create_samples.py` | 修改 | 新增 `--sample_games` 参数；新增 `is_sample_game`、`print_sample_progress` 函数；`main()` 中增加样本局驱动对弈循环 |

## 8. 测试用例

详见 [test_cases.md](test_cases.md)
