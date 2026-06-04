# 对局级样本过滤机制 — 质疑与挑战（第一轮）

> 审查立场：全方位、多维度深度质疑，目的是揭示设计中的潜在缺陷、风险和遗漏，说服读者拒绝此方案。

---

## 一、架构设计：过滤职责错配，侵入式改造代价过高

### 1.1 过滤逻辑注入层级不当——在 SampleCreator 做对局级过滤是职责错位

设计在 `SampleCreator.create()` 中插入 `record_filter`，但 `SampleCreator` 的核心职责是 **从 record.json 生成训练样本**（input/output/mask 构造），而非对局级筛选。将过滤逻辑注入到样本生成器内部，造成：

- **语义混淆**：`record_filter` 的操作粒度是"整局 record"，而 `SampleCreator` 的操作粒度是"单步 sample"。一个管"局"的过滤器被塞进管"步"的模块，两层抽象被强制耦合。
- **过滤时机不优**：过滤发生在 `json.load(f)` 之后、mode 分派之前。这意味着即使对局将被丢弃，文件 I/O + JSON 反序列化的成本已经付出了。对大规模对局数据（成千上万局），浪费不可忽视。

**更合理的做法**：过滤应发生在文件发现阶段（`create_from_round_dir` 扫描子目录时），或在 `Evaluator.evaluate_round` 层面——在产生 record.json 的文件路径列表时即剔除不需要的对局，避免无谓的 I/O。

### 1.2 新增 game_filter.py 模块过轻——一个纯函数不值得一个文件

`should_keep_game` 是一个 4 行条件判断的纯函数。为它单独创建 `ai/game_filter.py` 模块，违反了"模块应有独立且充实的职责"原则：

- 模块只有 1 个函数，未来也几乎不可能扩展（过滤规则由 PRD 锁定，没有"多种过滤策略"的扩展场景）。
- 增加了 import 路径的复杂度和项目文件组织的碎片化。
- 调用方（`create_samples.py`）只需要 `from ai.game_filter import should_keep_game`，这一行完全可以被 `from ai.move_sampler import should_keep_game` 替代——因为过滤判定本质上就是温度采样语义的一部分。

**建议**：将 `should_keep_game` 放入 `move_sampler.py`，与 `should_sample`、`resolve_temp_side` 形成内聚的"温度采样策略"函数族。

### 1.3 双轨样本生成管线（Evaluator vs SampleCreator）未被统一考虑

当前项目中存在 **两条并行的样本生成管线**：

1. **Evaluator.evaluate_round** → 生成 `sample.json`（被 `SelfPlayCycle`、`SelfPlayGenerator`、`BattleSession` 使用）
2. **SampleCreator.create_from_round_dir** → 直接返回样本列表（被 `create_samples.py` 使用）

设计方案只在 SampleCreator 管线上加了 `record_filter`，**完全忽略了 Evaluator 管线**。这意味着：

- `SelfPlayCycle.run()` → `run_self_play_and_evaluate()` → `Evaluator.evaluate_round()` 这条主要训练管线 **永远不会过滤**，即使 `temp_side="black"` 且贪婪方赢了，这些"坏样本"仍会被用于训练。
- `BattleSession.run()` → `Evaluator.evaluate_round()` 同理。

方案声称"train_main.py（cycle 模式）：不传 record_filter，行为不变"是"兼容性"——但 PRD 的需求是 **所有单方采样场景** 都应过滤，而非仅 CLI 脚本。如果 Evaluator 管线不做过滤，那训练周期产出的样本质量就无法保障，PRD 需求只被满足了一半。

这不是兼容性，而是**功能缺失**。

---

## 二、API 设计：返回值类型变更破坏性被低估

### 2.1 `_play_single_game` 返回值从 `int` 改为 `dict` 是破坏性变更

设计将 `_play_single_game` 的返回值从 `int` 改为 `dict`。虽然声称"调用方用 `r["winner"]` 不受影响"，但这只考虑了 **直接的调用方**（`start_ai_vs_ai` 和 `start_ai_vs_ai_random`）。实际问题包括：

- **所有测试用例需要重写**：现有测试 `test_game_manager_temp.py` 中大量 mock 的返回值是 `int`，必须全部改为 `dict`。这不只是"适配"，而是 **整个测试套件的返工**。
- **调用方解构模式变更**：当前 `start_ai_vs_ai` 直接 `winner = AIGameManager._play_single_game(...)`，变更后必须 `result = ...; winner = result["winner"]`。同样 `start_ai_vs_ai_random` 中的 `ai1_won = (winner == ai1_color)` 也必须改写。
- **类型安全退化**：从强类型的 `int`（winner）退化为弱类型的 `dict`，调用方必须用字符串键 `"winner"` 访问，typo 风险无处防护。没有 `TypedDict` 或 `NamedTuple` 或 `dataclass` 约束返回结构。

**更合理的做法**：保持 `_play_single_game` 返回 `int`，将 `temp_side_this_game` 通过 `DataRecorder` 的写入接口独立传递。`_play_single_game` 已持有 `recorder` 引用，完全可以在函数内部直接调用 `recorder.save_game_result(..., temp_side_this_game=temp_side_this_game)`——事实上设计文档正是这么写的（第93行），但同时又把 `temp_side_this_game` 塞进返回值，导致 **信息被重复传递**：一次通过 `recorder.save_game_result` 持久化，一次通过返回值透传给 `start_ai_vs_ai` 再塞进 `results` 列表。

### 2.2 `start_ai_vs_ai` 返回值结构被静默扩展

当前 `start_ai_vs_ai` 返回 `List[dict]`，每项为 `{"game": int, "winner": int}`。设计新增 `temp_side_this_game` 键。

问题是：**这个返回值被多少调用方消费？**经审查：

- `create_samples.py` 第 165-168 行：只读 `r["winner"]`
- `self_play_generator.py` 第 95-98 行：只读 `r["winner"]`
- `battle.py` 第 85-89 行：整个 `results` 列表仅被返回，未消费内部字段

目前确实"兼容"，但这是一个 **脆弱的隐性契约**。没有任何类型注解或文档约束返回值 schema，任何调用方随时可能开始依赖新字段，却没有迁移指南。方案缺少对返回值 schema 的正式定义（如 TypedDict）。

### 2.3 `record_filter: Callable[[dict], bool]` 过于宽泛

`record_filter` 参数类型为 `Callable[[dict], bool]`，没有对输入 `dict` 的 schema 做任何约束。这意味着：

- 调用方可以传入任何函数，包括有副作用的、抛异常的、依赖外部状态的。
- `record_filter` 接收的 `record` 是 `json.load` 的原始字典，其 schema 没有文档化——调用方如何知道该检查哪些字段？
- 过滤函数的返回语义（True=保留/False=丢弃）只靠文档约定，没有枚举或常量。

当项目只有 `should_keep_game` 一个过滤函数时，这个抽象是过度设计；当未来有更多过滤需求时，这个抽象又不够严谨。两头不讨好。

---

## 三、数据模型：schema 变更的不一致性和冗余

### 3.1 `temp_side_this_game` 始终写入 record.json 是噪声污染

设计要求"始终写入 `temp_side_this_game`（含 None 值，JSON 中为 null）"。这意味着 **每一局** 的 `record.json` 都会多一个字段，即使 99% 的场景下该字段为 `null`：

- GUI AI 内战不传 `temp_config` → 写 `null`
- `train_main.py --cycle` 默认 `temp_side="both"` → 写 `"both"`（但双方采样不过滤，写这个字段纯属冗余）
- `create_samples.py --no-temp-sampling` → 写 `null`

在 `DataRecorder.new_game()` 初始化 `current_game_data` 时，`winner` 和 `end_board` 是每局必有且有语义的字段，而 `temp_side_this_game` 在大多数场景下无意义。**无条件写入** 违反了"字段存在应有语义必要性"原则，也增加了旧数据与新代码之间的认知负担（看到 `null` 要理解"这是未启用温度采样"）。

更合理的做法：**条件写入**——仅在 `temp_side_this_game` 非 None 时写入，与 `strategy`/`temperature`/`top_k` 的条件写入模式（见 `record_move` 第 128-138 行）保持一致。方案对 `record_move` 做了条件写入，对 `save_game_result` 却做了无条件写入，**同一个类的同一个概念（温度采样元数据）采用了两种不同的写入策略**，这是明显的不一致。

### 3.2 `temp_side_this_game` 与 `history` 中的 `strategy`/`temperature` 存在信息冗余

`record.json` 的 `history` 中每一步已有 `strategy` 字段（值为 `"sampling"` 或 `"argmax"`），以及 `temperature` 字段。这些信息 **已经隐含了** "哪一方在做温度采样"：

- 如果偶数步（黑方）有 `strategy="sampling"`，而奇数步（白方）没有 → 隐含 `temp_side_this_game="black"`
- 如果所有步都是 `strategy="argmax"` → 隐含 `temp_side_this_game=None`

将 `temp_side_this_game` 作为顶层字段冗余存储，破坏了数据模型的信息归一化原则。如果未来某天 history 中的策略信息与顶层字段不一致（例如代码 bug 导致写入矛盾），将无从判断哪个是"真相源"。

### 3.3 record.json schema 变更缺少版本化机制

`record.json` 没有版本号字段。每次 schema 变更（从最初的 `winner`+`end_board`+`history`，到追加 `step_id`、`strategy`/`temperature`/`top_k`/`sampling_probs`，再到现在的 `temp_side_this_game`）都是直接添加字段，无 `schema_version` 或 `format_version` 标识。

方案用 `record.get("temp_side_this_game")` 做向后兼容，但这只是一个约定，不是机制。随着 schema 持续膨胀，未来任何字段重命名或语义变更都将无法可靠地检测和处理。

---

## 四、兼容性：被忽视的破坏面

### 4.1 `start_ai_vs_ai_random` 的结果列表结构冲突

当前 `start_ai_vs_ai_random` 返回的每项 dict 包含 6 个键：`game`、`winner`、`ai1_color`、`ai2_color`、`ai1_won`、`ai2_won`。设计方案只说"结果列表新增 `temp_side_this_game` 键"，但未分析 `temp_side_this_game` 与 `ai1_color`/`ai2_color` 之间的语义关系：

- 在 `start_ai_vs_ai_random` 中，引擎执方是随机的。`temp_side_this_game="black"` 时，采样方是"黑方"，但黑方可能是 engine1 也可能是 engine2。
- 结果列表中已有 `ai1_color` 可以确定哪个引擎是黑方。`temp_side_this_game` 与 `ai1_color` 组合才能确定"哪个引擎在做采样"——但设计完全没有提到 `start_ai_vs_ai_random` 的结果列表需要新增"哪个引擎在做采样"的语义。

如果有人想统计"采样方引擎 vs 贪婪方引擎的胜率"，仅靠 `temp_side_this_game` + `ai1_color` 可以推导，但这属于隐含契约，未被显式建模。

### 4.2 Evaluator.evaluate_round 不感知过滤——已生成的"坏样本"无法追溯清除

`Evaluator.evaluate_round()` 在 `run_self_play_and_evaluate()` 中被调用，它的输入是 `round_dir`，输出是 `sample.json` 文件。这个过程 **完全没有过滤**。

假设 `SelfPlayCycle` 运行了 10 轮，每轮 100 局，`temp_side="black"`。根据 PRD 需求，只有采样方黑胜的对局应保留。但 Evaluator 会对 **全部 1000 局** 都生成 `sample.json`，其中大约一半是"坏样本"（贪婪方白胜）。这些坏样本被 `trainer.train_auto(sample_dirs)` 吃掉。

设计文档声称"train_main.py（cycle 模式）：不传 record_filter，行为不变"是兼容的——但"行为不变"意味着 **训练周期永远不会做对局级过滤**，除非在 `SelfPlayCycle` 或 `run_self_play_and_evaluate` 中也注入 `record_filter`，而这是设计完全未考虑的。

### 4.3 GUI AI 内战路径缺少 temp_side_this_game 的生成来源

方案声称"GUI AI 内战：不传 temp_config，save_game_result 写入 temp_side_this_game=null"。但审查 `gui/components/ai_battle.py` 和 `app/battle.py` 的调用链：

```python
# battle.py 第 85-89 行
results = AIGameManager.start_ai_vs_ai(
    engine1=engine1, engine2=engine2, ...,
    temp_config=self._temp_config,  # GUI路径中是 None
)
```

当 `temp_config=None` 时，`_play_single_game` 内部 `temp_side_this_game` 为 `None`，这是正确的。但问题在于：**GUI 的 `AIBattleProgressWindow` 不传 `temp_config` 参数给 `BattleSession`**（见 `ai_battle.py` 第 346 行 `progress = AIBattleProgressWindow(num_games, ai1_name, ai2_name)`——注意没传 `temp_config`）。这条路径是安全的。

然而，方案修改后 `_play_single_game` 返回 dict，`start_ai_vs_ai` 的结果列表结构变了，`BattleSession.run()` 第 98 行 `return results` 的消费者（GUI）是否依赖 `results` 的具体结构？审查 `ai_battle.py` 发现 `_on_battle_complete` 方法不使用 `results`，仅使用 `recorder`。目前安全，但同样是隐性契约。

---

## 五、测试覆盖：严重遗漏

### 5.1 缺少 Evaluator 管线的过滤测试

测试用例文档共 8 组 44 个用例，**没有一个** 针对 `Evaluator.evaluate_round` 在单方采样模式下的行为。这是最大的遗漏——因为 `SelfPlayCycle` 和 `SelfPlayGenerator` 使用的正是 Evaluator 管线，而方案没有给这条管线加过滤。

### 5.2 缺少 `run_self_play_and_evaluate` 的集成测试

方案未测试 `run_self_play_and_evaluate()` 在 `temp_side="black"` 时是否只生成采样方获胜对局的样本。这恰恰是最核心的端到端场景。

### 5.3 缺少 `SelfPlayCycle.run()` 的回归测试

方案声称"train_main.py --cycle 运行：行为与修改前一致"，但没有测试用例验证这一点。`BC-04` 说"train_main.py --cycle 运行，行为与修改前一致"，但这只是一个描述性断言，没有定义"一致"的具体检查项（哪些输出不变？哪些文件不变？loss 行为不变？）。

### 5.4 缺少并发/中断场景测试

`run_self_play_and_evaluate` 有 `KeyboardInterrupt` 处理逻辑。在方案修改后，`_play_single_game` 返回值变为 dict，中断时 `results` 列表可能包含部分 dict、部分尚未修改的旧格式（如果部分对局在代码变更前已完成）。**中断恢复场景下的类型一致性** 未被测试。

### 5.5 `should_keep_game` 缺少异常输入测试

`should_keep_game` 接收任意 `dict`，但测试用例只覆盖了"正常的 record 字典"。缺少：

- `should_keep_game({})` — 完全空字典
- `should_keep_game({"temp_side_this_game": "invalid_value", "winner": 1})` — 非法 temp_side 值
- `should_keep_game({"temp_side_this_game": "black", "winner": "1"})` — winner 类型错误（字符串而非整数）

这些异常输入在真实场景中完全可能出现（手动编辑 record.json、数据损坏等），但方案假设输入总是合法的。

---

## 六、边界条件与正确性

### 6.1 过滤统计输出的计算逻辑有 bug

设计方案在 `create_samples.py` 中的过滤统计：

```python
filter_active = (
    temp_config is not None
    and temp_config.temp_side in ("black", "white", "random")
)
if filter_active:
    kept = sum(1 for r in results if r.get("temp_side_this_game") == r["winner"])
```

**`kept` 的计算逻辑是错误的**。`r.get("temp_side_this_game")` 是 `"black"` 或 `"white"`（字符串），而 `r["winner"]` 是 `1` 或 `2`（整数）。字符串永远不等于整数，所以 `kept` 永远是 0！

正确逻辑应该是：

```python
kept = sum(
    1 for r in results
    if (r.get("temp_side_this_game") == "black" and r["winner"] == 1)
    or (r.get("temp_side_this_game") == "white" and r["winner"] == 2)
)
```

或者复用 `should_keep_game` 函数——但 `should_keep_game` 接收的是 `record`（record.json 格式），而 `results` 中的元素是 `start_ai_vs_ai` 返回的格式，两者 schema 不同。这暴露了方案中 **数据格式不统一** 的深层问题。

### 6.2 `temp_config is None` 时 `filter_active` 的判定依赖隐式条件

当 `args.no_temp_sampling` 为 `True` 时，`temp_config = None`，`filter_active = False`。这没问题。

但当用户传 `--temp-side both` 时，`temp_config.temp_side = "both"`，`filter_active = False`——因为 `"both"` 不在 `("black", "white", "random")` 中。输出"样本过滤: 未启用 (非单方采样模式)"。**这是正确行为**。

但考虑 `--temp-side random` 场景：`temp_config.temp_side = "random"`，`filter_active = True`。此时每局 `temp_side_this_game` 是 `"black"` 或 `"white"`（由 `resolve_temp_side` 抽签决定），但 `create_samples.py` 在构造 `filter_active` 时只知道原始配置是 `"random"`，不知道每局的实际值。**统计输出能否正确反映按局过滤的结果** 取决于 `results` 中每项的 `temp_side_this_game` 值是否正确——而这又取决于 `_play_single_game` → `start_ai_vs_ai` 的透传是否无 bug。一条隐蔽的依赖链。

### 6.3 `should_keep_game` 对 `winner=0`（未结束）的处理

`GF-10` 测试用例：`{temp_side_this_game: "black", winner: 0}` → 返回 `False`。但 `SampleCreator.create()` 第 80 行已经跳过了 `winner == 0` 的情况：

```python
if winner == 0 or winner == 3:
    continue
```

所以 `should_keep_game` 永远不会看到 `winner=0` 的 record。`GF-10` 是一个不可达路径的测试——看似完善，实为虚假覆盖率。真正需要担心的是 `winner=0` 如何进入 `should_keep_game`（如果有人单独调用它），但这个边界条件并未在 `should_keep_game` 的文档中明确说明前置条件。

### 6.4 单方采样 + 平局的交互

`should_keep_game` 对 `temp_side_this_game="black"` + `winner=3`（平局）返回 `False`。但 `SampleCreator.create()` 第 80 行也会跳过平局。所以过滤函数的平局处理是冗余的——同样，两层过滤逻辑的职责边界模糊：**平局到底该由谁过滤？** 如果两个地方都过滤，任何一个漏掉都不会出错，但这掩盖了职责不清的问题。

---

## 七、可维护性：隐性耦合和技术债

### 7.1 过滤规则的"真相源"分散在三处

判定"是否保留对局"的逻辑分布在：

1. `should_keep_game`（纯函数，基于 record dict）
2. `SampleCreator.create()` 第 80-81 行（硬编码跳过平局和未结束）
3. `create_samples.py` 的 `filter_active` 判定（基于 `temp_config.temp_side`）

如果未来需要修改过滤规则（例如"平局也要保留一部分"或"加入步数阈值"），开发者需要同时修改三处，且没有机制保证一致性。

### 7.2 `temp_side_this_game` 的语义重载

`temp_side_this_game` 同时承载两个语义：
- **记录语义**：该局实际上哪一方做了温度采样（事实描述）
- **过滤语义**：是否应该保留该对局用于训练（决策依据）

这两个语义应分离。`should_keep_game` 的逻辑 `"black" 且 winner==1 → True` 实际上是将记录语义隐式转换为过滤决策。如果未来过滤规则变了（例如"采样方输了的对局也保留一部分用于负样本"），`temp_side_this_game` 的记录语义仍不变，但 `should_keep_game` 需要改。这种耦合目前没问题，但会随需求演进越来越紧。

### 7.3 `create_samples.py` 与 `train_main.py` 的行为分化

修改后：
- `create_samples.py`：**会** 过滤（传入 `record_filter=should_keep_game`）
- `train_main.py --cycle`：**不会** 过滤（`SelfPlayCycle` 不传 `record_filter`）

两条命令都是"自我对弈 + 生成样本"，但样本质量保障策略不同。用户完全可能困惑："为什么 `create_samples.py` 生成的样本比 `train_main.py` 少？"这在文档中未解释，也无运行时警告。

---

## 八、可扩展性：设计短视

### 8.1 `record_filter` 是闭包黑盒，无法组合

`record_filter: Callable[[dict], bool]` 无法表达"且"/"或"逻辑组合。如果未来 PRD 新增"步数少于 10 步的对局也丢弃"，无法用现有的 `record_filter` 机制组合两个过滤条件——除非在 `create_samples.py` 中手写一个新的 lambda 来组合，而这个 lambda 又无法复用。

更合理的做法：定义 `RecordFilter` 协议类或抽象基类，支持 `AndFilter`/`OrFilter` 组合，或至少用 `List[Callable]` + all/any。

### 8.2 只支持对局级过滤，未来可能需要步级过滤

当前设计只做对局级过滤（整局保留/丢弃）。但 PRD 的演进方向完全可能是"保留对局，但只取采样方的步骤作为训练样本"——这正是 `winner_path` 模式的思路。设计没有为步级过滤预留扩展点，`record_filter` 的粒度是整局 record，无法适配步级需求。

### 8.3 Evaluator 管线完全无法扩展

`Evaluator.evaluate_round()` 和 `Evaluator.evaluate_record()` 的签名和逻辑完全不受此次修改影响。如果未来要在 Evaluator 管线上加过滤，需要修改 `Evaluator` 本身——而 `Evaluator` 是一个复杂的多进程并行评价器，修改它的风险和成本远高于修改 `SampleCreator`。方案选择只改 `SampleCreator`，实际上是把技术债留给了 Evaluator 管线。

---

## 九、设计遗漏

### 9.1 未考虑 `start_ai_vs_ai_random` + 单方采样的交互

当 `start_ai_vs_ai_random` 与 `temp_side="black"` 组合时：

- 每局随机决定 engine1 执黑还是白
- 采样方是"黑方"，但黑方可能是 engine1 也可能是 engine2

此时 `temp_side_this_game="black"` 只是说"采样方是黑方"，但没有说"采样方是哪个引擎"。如果 `engine1` 和 `engine2` 是不同模型（这正是 `start_ai_vs_ai_random` 的设计意图），那么"采样方引擎"和"贪婪方引擎"的区分就至关重要——它决定了训练样本来自哪个模型的自对弈。但 `results` 列表中的 `temp_side_this_game` + `ai1_color` 可以推导出这个信息，方案未文档化这一推导关系。

### 9.2 未考虑 `should_keep_game` 在 `Evaluator` 管线中的替代方案

方案完全未讨论是否应该/如何在 Evaluator 管线中实现过滤。可能的替代方案：

1. 在 `run_self_play_and_evaluate` 中，对弈完成后、评价之前，先删除不需要的 record.json 文件
2. 在 `Evaluator.evaluate_round` 中加入 `record_filter` 参数
3. 在 `Evaluator._evaluate_files` 中加入文件级过滤

方案未评估这些替代方案的利弊，直接跳到了"只改 SampleCreator"的结论。

### 9.3 未处理 `create_samples.py` 的 `--temp-side random` 场景下的统计准确性

当 `--temp-side random` 时，`temp_config.temp_side = "random"`。`filter_active` 为 True。但 `create_samples.py` 中的统计：

```python
kept = sum(1 for r in results if r.get("temp_side_this_game") == r["winner"])
```

在 `random` 模式下，`r["temp_side_this_game"]` 可能是 `"black"` 或 `"white"`，而 `r["winner"]` 是 `1` 或 `2`。如前所述（6.1），字符串 != 整数，统计永远为 0。这不是统计"不准确"的问题，而是 **根本无法工作**。

---

## 十、总结：核心问题清单

| # | 问题 | 严重性 | 类别 |
|---|------|--------|------|
| 1 | Evaluator 管线（训练周期主力管线）无过滤，PRD 需求只满足一半 | 🔴 致命 | 功能缺失 |
| 2 | 过滤统计 `kept` 计算逻辑有 bug（字符串 vs 整数比较） | 🔴 致命 | 正确性 |
| 3 | `_play_single_game` 返回值类型变更破坏面被低估 | 🟠 严重 | API 设计 |
| 4 | `temp_side_this_game` 无条件写入 vs `strategy` 条件写入不一致 | 🟠 严重 | 数据模型 |
| 5 | 过滤职责错配：应在文件发现阶段或 Evaluator 层面过滤 | 🟠 严重 | 架构 |
| 6 | `game_filter.py` 过轻，应归入 `move_sampler.py` | 🟡 中等 | 模块组织 |
| 7 | 两条管线行为分化（create_samples 过滤 / train_cycle 不过滤）未文档化 | 🟡 中等 | 可维护性 |
| 8 | `record_filter` 闭包黑盒无法组合 | 🟡 中等 | 可扩展性 |
| 9 | 测试覆盖遗漏 Evaluator 管线、异常输入、中断恢复 | 🟡 中等 | 测试 |
| 10 | `should_keep_game` 与 `SampleCreator.create()` 职责边界模糊（平局过滤冗余） | 🟢 低 | 可维护性 |

**结论**：方案在核心需求上存在致命缺陷（Evaluator 管线无过滤、统计逻辑 bug），API 变更的破坏性被低估，数据模型设计存在不一致，应退回重新设计后再提交评审。
