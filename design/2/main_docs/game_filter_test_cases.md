# 对局级样本过滤机制 - 测试用例

## 1. should_keep_game 单元测试

| 用例编号 | 输入 record | 预期返回 | 说明 |
|----------|------------|---------|------|
| GF-01 | `{temp_side_this_game: "black", winner: 1}` | True | 采样方黑胜 → 保留 |
| GF-02 | `{temp_side_this_game: "black", winner: 2}` | False | 贪婪方白胜 → 丢弃 |
| GF-03 | `{temp_side_this_game: "black", winner: 3}` | False | 平局 → 丢弃 |
| GF-04 | `{temp_side_this_game: "white", winner: 2}` | True | 采样方白胜 → 保留 |
| GF-05 | `{temp_side_this_game: "white", winner: 1}` | False | 贪婪方黑胜 → 丢弃 |
| GF-06 | `{temp_side_this_game: "white", winner: 3}` | False | 平局 → 丢弃 |
| GF-07 | `{temp_side_this_game: "both", winner: 1}` | True | 双方采样 → 不过滤 |
| GF-08 | `{temp_side_this_game: None, winner: 1}` | True | 无温度采样 → 不过滤 |
| GF-09 | `{winner: 1}`（无 temp_side_this_game 字段） | True | 旧格式兼容 → 不过滤 |
| GF-10 | `{temp_side_this_game: "black", winner: 0}` | False | 未结束 → 丢弃 |

## 2. DataRecorder 单元测试

### 2.1 new_game 初始化测试

| 用例编号 | 调用 | 预期 current_game_data | 说明 |
|----------|------|----------------------|------|
| DR-01 | `new_game()` | 含 `temp_side_this_game: None` | 初始化为 None |

### 2.2 save_game_result 写入测试

| 用例编号 | 调用参数 | 预期 record.json | 说明 |
|----------|---------|-----------------|------|
| DR-02 | `save_game_result(1, 1, board, temp_side_this_game="black")` | 含 `temp_side_this_game: "black"` | 单方采样 |
| DR-03 | `save_game_result(1, 2, board, temp_side_this_game="white")` | 含 `temp_side_this_game: "white"` | 单方采样 |
| DR-04 | `save_game_result(1, 1, board, temp_side_this_game="both")` | 含 `temp_side_this_game: "both"` | 双方采样 |
| DR-05 | `save_game_result(1, 1, board, temp_side_this_game=None)` | 含 `temp_side_this_game: null` | 无温度采样 |
| DR-06 | `save_game_result(1, 1, board)`（不传 temp_side_this_game） | 含 `temp_side_this_game: null` | 默认值为 None |

## 3. AIGameManager._play_single_game 返回值测试

| 用例编号 | temp_config | 预期返回值 | 说明 |
|----------|-------------|-----------|------|
| GM-01 | `TempConfig(temp_side="black")` | `SingleGameResult(winner=1/2, temp_side_this_game="black")` | 单方采样 |
| GM-02 | `TempConfig(temp_side="white")` | `SingleGameResult(winner=1/2, temp_side_this_game="white")` | 单方采样 |
| GM-03 | `TempConfig(temp_side="both")` | `SingleGameResult(winner=1/2, temp_side_this_game="both")` | 双方采样 |
| GM-04 | None（无 temp_config） | `SingleGameResult(winner=1/2, temp_side_this_game=None)` | 无采样 |
| GM-05 | `TempConfig(temp_side="random")` | `SingleGameResult(winner=1/2, temp_side_this_game="black"/"white")` | random 按局确定 |

## 4. AIGameManager.start_ai_vs_ai 结果列表测试

| 用例编号 | temp_config | 预期结果列表元素键 | 说明 |
|----------|-------------|-------------------|------|
| SA-01 | `TempConfig(temp_side="black")` | 每项含 `temp_side_this_game: "black"` | 透传采样方信息 |
| SA-02 | None | 每项含 `temp_side_this_game: None` | 无采样时为 None |
| SA-03 | `TempConfig(temp_side="both")` | 每项含 `temp_side_this_game: "both"` | 双方采样时为 "both" |
| SA-04 | `TempConfig(temp_side="random")` | 每项含 `temp_side_this_game: "black"/"white"` | 每局可能不同 |

## 5. SampleCreator record_filter 测试

| 用例编号 | record_filter | 输入 records | 预期样本数 | 说明 |
|----------|--------------|-------------|-----------|------|
| SC-01 | `should_keep_game` | 2局：采样方黑胜 + 贪婪方白胜 | 仅黑胜局样本 | 过滤生效 |
| SC-02 | `should_keep_game` | 2局：temp_side_this_game="both" | 2局样本 | 双方采样不过滤 |
| SC-03 | `should_keep_game` | 2局：temp_side_this_game=None | 2局样本 | 无采样不过滤 |
| SC-04 | `None`（默认） | 任意 | 全部样本 | 不传不过滤 |
| SC-05 | `lambda r: False`（全丢弃） | 任意 | 0 | 外部可自定义过滤 |
| SC-06 | `should_keep_game` | 平局（winner=3） | 0 | 平局本身已跳过 + 过滤 |

## 6. SampleCreator.create_from_round_dir record_filter 测试

| 用例编号 | record_filter | round_dir 内容 | 预期样本数 | 说明 |
|----------|--------------|---------------|-----------|------|
| SR-01 | `should_keep_game` | 含 temp_side_this_game="black" 且黑胜 | 正常样本 | 过滤保留 |
| SR-02 | `should_keep_game` | 含 temp_side_this_game="black" 且白胜 | 0 | 过滤丢弃 |
| SR-03 | `should_keep_game` | 含 temp_side_this_game="both" 且白胜 | 正常样本 | 双方采样不过滤 |
| SR-04 | `None` | 同 SR-02 | 正常样本 | 不过滤 |

## 7. create_samples.py CLI 集成测试

| 用例编号 | CLI 参数 | mock results | 预期行为 | 说明 |
|----------|---------|-------------|---------|------|
| CLI-01 | `--temp-side black` | 黑胜 + 白胜各1局 | 仅黑胜局产生样本；stdout 含"样本过滤: 启用" | 单方采样过滤 |
| CLI-02 | `--temp-side white` | 黑胜 + 白胜各1局 | 仅白胜局产生样本；stdout 含"样本过滤: 启用" | 单方采样过滤 |
| CLI-03 | `--temp-side both` | 黑胜 + 白胜各1局 | 两局均产生样本；stdout 含"样本过滤: 未启用" | 非单方采样不过滤 |
| CLI-04 | `--no-temp-sampling` | 黑胜1局 | 正常产生样本；stdout 含"样本过滤: 未启用" | 无采样不过滤 |
| CLI-05 | `--temp-side random` | 2局（采样方各异） | 仅采样方获胜局产生样本 | random 按局判定 |
| CLI-06 | `--temp-side black --mode winner_path` | 黑胜1局（5步） | 仅黑胜局产生 winner_path 样本 | winner_path 模式过滤 |

## 8. 向后兼容回归测试

| 用例编号 | 场景 | 预期行为 | 说明 |
|----------|------|---------|------|
| BC-01 | 旧格式 record.json（无 temp_side_this_game）传给 should_keep_game | 返回 True | 旧数据 get() 返回 None，不过滤 |
| BC-02 | 不传 record_filter 调用 SampleCreator.create | 行为与修改前一致 | 可选参数 |
| BC-03 | 新格式 record.json 含 temp_side_this_game="both" | should_keep_game 返回 True | 双方采样不过滤 |
| BC-04 | train_main.py --cycle 运行 | 行为与修改前一致 | 其他脚本不受影响 |

## 9. SingleGameResult NamedTuple 测试

| 用例编号 | 测试内容 | 预期行为 | 说明 |
|----------|---------|---------|------|
| SN-01 | 属性访问 `result.winner` | 返回 int | 类型安全访问 |
| SN-02 | 属性访问 `result.temp_side_this_game` | 返回 str 或 None | 类型安全访问 |
| SN-03 | 错误属性 `result.typo_field` | 抛出 AttributeError | NamedTuple 防 typo |