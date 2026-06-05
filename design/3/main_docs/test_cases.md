# 测试用例: create_samples --sample_games

## 1. 参数互斥与校验 (A)

| 编号 | 用例名 | 输入 | 期望结果 |
|------|--------|------|----------|
| A-01 | games与sample_games同时指定 | `--games 10 --sample_games 10` | 报错"不能同时指定"，exit code ≠ 0 |
| A-02 | sample_games为0 | `--sample_games 0` | 报错"值必须大于0"，exit code ≠ 0 |
| A-03 | sample_games为负数 | `--sample_games -5` | 报错"值必须大于0"，exit code ≠ 0 |
| A-04 | 仅指定games | `--games 10` | 行为与未修改前完全一致 |
| A-05 | 两者都不指定 | 无games无sample_games | 默认 `--games 100`，行为不变 |
| A-06 | games为0 | `--games 0` | 报错"值必须大于0"，exit code ≠ 0 |
| A-07 | games为负数 | `--games -5` | 报错"值必须大于0"，exit code ≠ 0 |

## 2. is_sample_game 函数 (B)

| 编号 | 用例名 | 输入 | 期望结果 |
|------|--------|------|----------|
| B-01 | 黑胜+无温度采样 | `{winner:1, temp_side_this_game:None}` | True |
| B-02 | 白胜+无温度采样 | `{winner:2, temp_side_this_game:None}` | True |
| B-03 | 平局+无温度采样 | `{winner:3, temp_side_this_game:None}` | False |
| B-04 | 未结束+无温度采样 | `{winner:0, temp_side_this_game:None}` | False |
| B-05 | 采样方黑胜+temp_side=black | `{winner:1, temp_side_this_game:"black"}` | True |
| B-06 | 贪婪方白胜+temp_side=black | `{winner:2, temp_side_this_game:"black"}` | False |
| B-07 | 采样方白胜+temp_side=white | `{winner:2, temp_side_this_game:"white"}` | True |
| B-08 | 贪婪方黑胜+temp_side=white | `{winner:1, temp_side_this_game:"white"}` | False |
| B-09 | both模式+黑胜 | `{winner:1, temp_side_this_game:"both"}` | True |
| B-10 | both模式+平局 | `{winner:3, temp_side_this_game:"both"}` | False |

## 3. print_sample_progress 函数 (C)

| 编号 | 用例名 | 输入 (sample_count, target, total_games) | 期望输出 |
|------|--------|------------------------------------------|----------|
| C-01 | 首次10%进度 | (10, 100, 23) | `进度: 样本局 10/100 (10%), 已对弈 23 局` |
| C-02 | 50%进度 | (50, 100, 114) | `进度: 样本局 50/100 (50%), 已对弈 114 局` |
| C-03 | 100%达标 | (100, 100, 224) | `进度: 样本局 100/100 (100%), 已对弈 224 局` |
| C-04 | 小值达标 | (5, 5, 5) | `进度: 样本局 5/5 (100%), 已对弈 5 局` |

## 4. 对弈循环 (D)

| 编号 | 用例名 | 前置条件 | 期望结果 |
|------|--------|----------|----------|
| D-01 | 无过滤场景达标 | `--sample_games 10 --no-temp-sampling` | 精确10局对弈，样本局=10 |
| D-02 | 单方过滤场景达标 | `--sample_games 10 --temp-side black` | 原始对弈>10局，样本局=10 |
| D-03 | 计数与SampleCreator一致 | `--sample_games 10 --temp-side black` | is_sample_game 计数 = SampleCreator 实际产出样本数（依赖假设：play_single_game 保证 history 非空，见设计文档 D2） |
| D-04 | winning_move+增强 | `--sample_games 50 --mode winning_move` | 最终样本数 = 50×8 |
| D-05 | winning_move+禁用增强 | `--sample_games 50 --no-augment --mode winning_move` | 最终样本数 = 50 |
| D-06 | winner_path模式 | `--sample_games 10 --mode winner_path` | 样本局=10，最终样本数≥10 |

## 5. 进度输出时机 (E)

| 编号 | 用例名 | 场景 | 期望结果 |
|------|--------|------|----------|
| E-01 | 100局标准进度 | `--sample_games 100` | 10%/20%/30%/.../100% 各打印一次 |
| E-02 | 小值仅打印达标 | `--sample_games 5` | 至少打印一次100%进度 |
| E-03 | 非整除场景 | `--sample_games 7` | 按百分比阈值触发，达标时必定打印 |

## 6. Ctrl+C 中断 (F)

| 编号 | 用例名 | 场景 | 期望结果 |
|------|--------|------|----------|
| F-01 | 中断后打印统计 | 运行中按Ctrl+C | 输出中断信息：样本局数/目标/百分比/原始对弈数 |
| F-02 | 中断后样本可生成 | 中断后检查输出文件 | JSON合法，含input/output/mask字段 |
| F-03 | 中断后样本数一致 | 中断后比对 | 输出文件样本条数 与 SampleCreator从已有对局记录生成的结果一致 |

## 7. 向后兼容 (G)

| 编号 | 用例名 | 操作 | 期望结果 |
|------|--------|------|----------|
| G-01 | 不传sample_games | `--games 100` | 行为与未修改前完全一致 |
| G-02 | help输出 | `--help` | 包含 `--sample_games` 参数说明 |

## 8. 验收工具 (H)

### H-1: build_cli_command 互斥逻辑

| 编号 | 用例名 | 输入 (params) | 期望结果 |
|------|--------|---------------|----------|
| H-01 | sample_games有值 | `{model, output, sample_games: 10, games: 100}` | 命令含 `--sample_games 10`，不含 `--games` |
| H-02 | 仅games有值 | `{model, output, games: 100}` | 命令含 `--games 100`，不含 `--sample_games` |
| H-03 | 两者均无值 | `{model, output}` | 命令不含 `--games` 和 `--sample_games` |
| H-04 | sample_games为null | `{model, output, sample_games: null, games: 50}` | 命令含 `--games 50`，不含 `--sample_games` |
| H-05 | sample_games为0 | `{model, output, sample_games: 0}` | 命令含 `--sample_games 0`（由 CLI 侧校验值） |

### H-2: collectParams 字段

| 编号 | 用例名 | 前置状态 | 期望结果 | 测试方式 |
|------|--------|----------|----------|----------|
| H-06 | gamesInput有值 | games=100, sampleGamesInput为空 | `games: 100, sample_games: null` | 手动测试 |
| H-07 | sampleGamesInput有值 | sampleGamesInput=10, gamesInput为空 | `games: null, sample_games: 10` | 手动测试 |
| H-08 | 两者均为空 | gamesInput和sampleGamesInput都为空 | `games: null, sample_games: null` | 手动测试 |

### H-3: 前端互斥交互

| 编号 | 用例名 | 操作 | 期望结果 | 测试方式 |
|------|--------|------|----------|----------|
| H-09 | 输入games清空sample_games | sampleGamesInput=10 → 在gamesInput输入100 | sampleGamesInput 被清空 | 手动测试 |
| H-10 | 输入sample_games清空games | gamesInput=100 → 在sampleGamesInput输入10 | gamesInput 被清空 | 手动测试 |
| H-11 | 清空games不影响sample_games | gamesInput=100, sampleGamesInput为空 → 清空gamesInput | sampleGamesInput 保持为空（不变） | 手动测试 |
