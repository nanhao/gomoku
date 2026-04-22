# 🏆 五子棋项目 Bug 猎杀大赛 — 最终报告

> 比赛日期：2026-04-22
> 参赛选手：A号（逻辑与算法猎手）、B号（GUI与交互猎手）、C号（数据与模型猎手）

---

## 📊 比赛积分

| 参赛者 | 独立发现数 | 高严重度 | 中严重度 | 低严重度 | 排名 |
|--------|-----------|---------|---------|---------|------|
| B号 | 10 | 3 | 5 | 2 | 🥇 |
| C号 | 9 | 1 | 3 | 5 | 🥈 |
| A号 | 8 | 0 | 4 | 4 | 🥉 |

> 积分规则：高严重度 = 3分，中严重度 = 2分，低严重度 = 1分
> B号: 3×3 + 5×2 + 2×1 = 21分
> C号: 1×3 + 3×2 + 5×1 = 14分
> A号: 0×3 + 4×2 + 4×1 = 12分

**🎉 冠军：B号选手（GUI与交互猎手）** — 凭借发现 3 个高严重度 bug 脱颖而出！

---

## 🐛 去重合并后的完整 Bug 列表

经三位选手交叉验证，共识别出 **23 个独立有效 Bug**：

---

### 🔴 高严重度（4 个）

#### Bug H1：ReplayBrowser 嵌套 mainloop 导致应用挂死
- **文件**：`gui/components/replay_browser.py`，第146-148行、第295-297行
- **发现者**：B号
- **问题描述**：`RoundSelector.run()` 和 `GameSelector.run()` 在 `self.root` 是 `tk.Toplevel`（有父窗口）时调用 `self.root.mainloop()`。当用户选择完毕后 `_close()` 调用 `self.root.destroy()` 销毁 Toplevel，但嵌套的 `mainloop()` 不会返回——因为父窗口仍然存在，Tcl 事件循环继续运行。`select_round()` 永远不会返回，`browse()` 永远不会继续执行。
- **触发路径**：用户在主游戏窗口点击「回放对局」→ `_open_replay()` → `ReplayBrowser.browse()` → **挂死**
- **正确做法**：使用 `parent.wait_window(self.root)` 代替 `self.root.mainloop()`

#### Bug H2：「保存游戏」按钮不实际保存，存在数据丢失风险
- **文件**：`gui/main.py`，第381-389行
- **发现者**：B号
- **问题描述**：`_save_game()` 方法只弹出一个消息框说"游戏已自动保存"，但从未调用任何保存方法。`make_move()` 不执行保存，只有 `new_game()` 保存了初始状态。所有后续落子仅在内存中，关闭窗口即丢失全部进度。
- **影响**：用户信任"已自动保存"提示，实际数据不安全

#### Bug H3：PvE 模式 AI 回合期间无点击保护，玩家可替 AI 落子
- **文件**：`gui/main.py` 第212-230行；`app/session.py` 第57-96行
- **发现者**：B号
- **问题描述**：玩家落子后，AI 落子通过 `self.root.after(500, self._ai_move)` 延迟 500ms 执行。在此窗口期内，棋盘仍可点击，且 `make_move` 不校验落子方身份。完整触发链：玩家落子 → current_player 切换到 AI → 玩家再次点击 → 为 AI 方落子 → current_player 切回玩家 → 原定 AI 回调触发 → AI 为玩家方落子 → **游戏状态完全错乱**
- **影响**：游戏规则被彻底打破

#### Bug H4：sample_inspector 目标格式假设过时，分析工具完全失效
- **文件**：`ai/models/fc_v1/sample_inspector.py`，第13-19行、58行、70行、73-79行、126-127行、138行、164-181行
- **发现者**：C号
- **问题描述**：`SampleInspector` 假设 `sample.json` 中的 `output`（target）是原始 reward 值（±1,000,000 / 0），但 `Evaluator._compute_all_rewards` 实际输出的是 Softmax 概率分布（0~1 之间的浮点数）。导致：
  - 胜利位置检测始终返回 `None`（没有任何 target 值等于 1,000,000）
  - 已占位/空位分类完全错误（用 -1,000,000 检测已占位，而实际值是 0.0）
  - `_compute_loss` 将 Softmax 概率除以 1,000,000 变成 ~4e-9，loss 计算完全无意义
- **影响**：该分析工具输出的所有结果都是错误的，误导开发者

---

### 🟡 中严重度（11 个）

#### Bug M1：end_board 维度错误（227 维而非注释所述 226 维）
- **文件**：`ai/game_manager.py`，第192-193行、第287-288行
- **发现者**：A号、C号（共同发现）
- **问题描述**：`AIEngine.board_to_vector(board, 0)` 已返回 226 维向量（225 棋盘格 + 1 current_player），再 `append(0)` 变成 227 维，注释声称"使长度为 226"。且传入 `current_player=0` 是无效值（合法值为 1 或 2）。

#### Bug M2：blend_rewards 可产生超过 MAX_REWARD 的值
- **文件**：`ai/evaluator.py`，第726-744行
- **发现者**：A号、C号（共同发现）
- **问题描述**：`blend_rewards` 使用加法融合 `shape_reward + alpha * line_bonus`，当两者都接近 `MAX_REWARD = 1,000,000` 且 `alpha=1.0`（默认）时，结果可达 2,000,000，远超 `MAX_REWARD`。影响 Softmax 分布形态，使概率过度集中。

#### Bug M3：加载存档后 get_state() 返回错误的 start_time
- **文件**：`game/controller.py`，第148-170行
- **发现者**：A号
- **问题描述**：`GomokuGame` 类不存储 `start_time`，通过 `load_game()` 加载存档后，`get_state(short=False)` 检测到 `start_time` 缺失时用当前时间替代，而非保留原始创建时间。

#### Bug M4：get_state() 返回内部可变引用，与构造函数深拷贝策略不一致
- **文件**：`game/game.py`，第58-65行
- **发现者**：A号
- **问题描述**：构造函数通过深拷贝（`[row[:] for row in ...]`）保护内部状态，但 `get_state()` 直接返回 `self.board` 和 `self.history` 的引用。调用者修改返回值会直接篡改游戏内部状态。

#### Bug M5：加载存档后若轮到 AI，游戏卡死
- **文件**：`gui/main.py`，第391-432行
- **发现者**：B号
- **问题描述**：`_load_game()` 加载存档后只调用 `_update_display()` 更新界面，没有检查当前是否轮到 AI。如果加载的存档恰好是 AI 的回合，玩家无法落子（不是自己的回合），也没有机制触发 AI 落子。

#### Bug M6：AIBattleConfigDialog 在模型目录不存在时崩溃
- **文件**：`gui/components/ai_battle.py`，第137-152行
- **发现者**：B号
- **问题描述**：`_get_available_ais()` 直接调用 `os.listdir(ai_dir)` 而未检查目录是否存在。对比 `mode_selector.py` 中的同类方法有 `os.path.exists()` 保护。

#### Bug M7：打开回放销毁主游戏窗口，无法返回当前对局
- **文件**：`gui/main.py`，第448-452行
- **发现者**：B号
- **问题描述**：`_open_replay()` 先 `self.root.destroy()` 销毁主窗口再打开回放窗口。用户查看回放后无法返回当前对局。注：此代码路径因 Bug H1 当前不可达，但修复 H1 后此问题将暴露。

#### Bug M8：CLI show/short 命令会替换当前游戏状态
- **文件**：`cli/main.py`，第174-210行
- **发现者**：B号
- **问题描述**：`handle_show()` 和 `handle_short()` 调用 `controller.load_game(save_name)` 加载指定存档，这会替换当前游戏。`show`/`short` 是只读命令，不应修改游戏状态。

#### Bug M9：after 调度的 AI 落子回调未管理，新建游戏时可能误触发
- **文件**：`gui/main.py`，第228行
- **发现者**：B号
- **问题描述**：`self.root.after(500, self._ai_move)` 的返回值被丢弃，未保存。新建游戏时无法 `after_cancel()` 取消旧回调，可能在新的游戏状态中执行旧的 AI 落子。

#### Bug M10：data_recorder output_vector 类型标注和文档错误
- **文件**：`ai/data_recorder.py`，第78-89行
- **发现者**：C号
- **问题描述**：`output_vector` 参数类型标注为 `List[int]`，实际传入 `List[float]`（Softmax 概率）。文档声称 "675 维输出向量"，实际为 225 维。文档声称 "x, y, reward 列表"，实际为 225 个位置的概率值。

#### Bug M11：BaseModel 抽象基类缺少 save/load/network 接口定义
- **文件**：`ai/base_model.py`，全文
- **发现者**：C号
- **问题描述**：`BaseModel` 只定义了 `predict` 和 `name` 两个抽象接口，但 `BaseTrainer` 依赖 `self.model.network`、`self.model.save()` 等未声明的接口。`RandomModel` 未实现这些方法，传入 `BaseTrainer` 会抛出 `AttributeError`。

---

### 🟢 低严重度（8 个）

#### Bug L1：battle.py 帮助示例使用无效 AI 名称 "cnn"
- **文件**：`ai/battle.py`，第8-10行、第181-182行
- **发现者**：A号、C号（共同发现）
- **问题描述**：文档示例使用 `--ai1 cnn`，合法名称为 `cnn_v1`。按示例执行会收到 `ValueError`。

#### Bug L2：train_main.py 文档示例使用无效模型名称 "cnn"
- **文件**：`ai/train_main.py`，第12行
- **发现者**：C号
- **问题描述**：文档示例使用 `--model cnn`，合法名称为 `cnn_v1`。按示例执行会收到 `ValueError`。

#### Bug L3：start_ai_vs_ai_random 用 game_num 代替 recorder.current_game_id
- **文件**：`ai/game_manager.py`，第279行、第289行
- **发现者**：A号、C号（共同发现）
- **问题描述**：`start_ai_vs_ai` 正确使用 `recorder.current_game_id`，但 `start_ai_vs_ai_random` 使用循环变量 `game_num`。当前恰好因两者同步而不报错，但耦合脆弱。

#### Bug L4：save_game 修改传入的 game_state 字典（副作用）
- **文件**：`game/save.py`，第19-20行
- **发现者**：A号
- **问题描述**：`save_game` 方法在 `game_state` 中添加 `"name"` 键，修改了调用者的字典，违反最小惊讶原则。

#### Bug L5：detect_shapes 中"独一"在混合棋型时被完全丢弃
- **文件**：`ai/evaluator.py`，第284行
- **发现者**：A号
- **问题描述**：条件 `if "独一" in raw_shapes and len(raw_shapes) == 1` 导致"独一"在同时有其他棋型时被丢弃。当前因"独一"奖励为 0 影响有限，但输出不完整。

#### Bug L6：回放查看器步骤归零时热力图不清除，显示过期数据
- **文件**：`gui/components/replay_viewer.py`，第230-254行
- **发现者**：B号
- **问题描述**：`_update_display()` 只在 `step > 0` 时调用 `_draw_heatmaps()`。步骤 0 应显示空白热力图，但实际显示上一步的数据。

#### Bug L7：棋盘边界外点击可能映射到有效位置
- **文件**：`gui/components/game_board.py`，第154-158行
- **发现者**：B号
- **问题描述**：坐标计算使用 `int(x / self.CELL_SIZE)` 对负数向零截断（`int(-1/42) = 0`），导致棋盘左/上边缘外侧约 21 像素内的点击被错误映射到第 0 行/列。应使用 `//` 运算符（`-1 // 42 = -1`）。

#### Bug L8：train_cycle 末尾冗余的 model.save() 调用
- **文件**：`ai/base_trainer.py`，第472行
- **发现者**：C号
- **问题描述**：循环体末尾（第463行）已在每轮结束后保存模型，循环结束后第472行又调用了一次 `self.model.save()`，此时模型参数与上次保存完全相同，是无意义的重复磁盘 I/O。

---

## 🔗 Bug 关联分析

| 关联组 | 涉及 Bug | 说明 |
|--------|---------|------|
| 回放功能缺陷链 | H1, M7 | H1 导致 M7 的代码路径不可达；修复 H1 后 M7 将暴露 |
| AI 回合交互问题链 | H3, M9 | 同源于 `after` 回调缺少生命周期管理和状态保护 |
| 数据格式不一致链 | H4, M10, M1 | Evaluator 改为 Softmax 输出后，inspector(H4) 和 recorder 文档(M10)未同步更新；end_board 维度(M1)也与数据格式相关 |
| 文档名称错误链 | L1, L2 | battle.py 和 train_main.py 都使用了过时的 "cnn" 名称 |
| game_id 传递不一致 | L3 | 两个功能相似方法使用了不同的 game_id 传递方式 |

---

## 🏅 特别奖项

| 奖项 | 获奖者 | Bug | 理由 |
|------|--------|-----|------|
| 💀 最致命 Bug | B号 | H1（ReplayBrowser 挂死） | 直接导致功能完全不可用 |
| 🔥 最隐蔽 Bug | C号 | H4（inspector 完全失效） | 工具运行不报错但输出全错 |
| 💥 最危险 Bug | B号 | H3（AI 回合无保护） | 可导致游戏状态彻底错乱 |
| 🤝 最佳协作发现 | A号+C号 | M1（end_board 维度错误） | 两位选手独立发现同一 bug |
