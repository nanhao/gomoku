# 对局级样本过滤机制 — 辩护观点（第一轮）

> 辩护立场：基于事实与逻辑，对质疑方的每一个质疑点进行逐条反驳，论证设计方案的合理性与可执行性。对于确实存在的代码级 bug，坦诚承认并提出修复建议，同时论证其不构成推翻设计的理由。

---

## 一、关于"致命问题"的辩护

### 1.1 Evaluator 管线无过滤 — 这是 PRD 的显式排除，不是设计遗漏

质疑方认为"训练周期走的恰恰是 Evaluator 管线，PRD 需求在主力训练管线上根本没有生效"，并标记为"致命"。

**辩护：这个指控建立在对 PRD 的误读之上。**

PRD 第 2.3 节"约束"明确写道：

> - 本次修改**仅影响 `create_samples.py` 的行为**，其他脚本行为不变

PRD 第 3.2 节"排除范围"进一步明确：

> | 项 | 排除原因 |
> |----|---------|
> | **其他脚本的过滤行为** | **明确约束：不影响 create_samples.py 以外的脚本** |

质疑方将"PRD 约束"曲解为"设计遗漏"，这是根本性的立场偏移。设计方案严格遵循 PRD 的范围界定——仅对 `create_samples.py` 的行为施加过滤，`Evaluator` 管线（`SelfPlayCycle`、`SelfPlayGenerator`、`BattleSession`）不在本次需求范围内。

**质疑方的逻辑是"如果训练周期不过滤，样本质量无法保障"。** 但这恰恰是 PRD 作者的有意决策：

1. **训练周期（`SelfPlayCycle`）的样本生成使用 `Evaluator`**——`Evaluator` 采用双向棋型检测法 + 五连追溯奖励 + Softmax 概率分布，其样本构造逻辑与 `SampleCreator` 完全不同。对 `Evaluator` 加对局级过滤意味着要在 `evaluate_record` 中跳过整局评价，或评价后再丢弃 `sample.json`——两者都涉及多进程并行评价器的修改，复杂度远高于 `SampleCreator` 的单进程过滤。

2. **训练周期的温度采样默认是 `both`**。查看 `train_main.py` 第 183 行：`--temp-side` 默认值为 `'both'`。在 `both` 模式下，过滤本来就不激活。只有用户显式传 `--temp-side black/white/random` 时过滤才有意义，而这种用法在训练周期中是高级可选操作，不是主流程。

3. **PRD 的渐进式交付策略是审慎的**。先在 `create_samples.py`（独立 CLI 脚本、单进程、无副作用）中验证过滤逻辑的正确性和实用性，再考虑扩展到训练周期——这是标准的工程实践，不是"功能缺失"。

**结论**：Evaluator 管线无过滤是 PRD 的显式排除项，设计方案忠实地执行了 PRD 的范围界定。如果未来需要将过滤扩展到 Evaluator 管线，那是增量需求，应通过新的 PRD 追踪，而不是在本设计中越权实现。

---

### 1.2 过滤统计 `kept` 计算逻辑有 bug — 坦诚承认，但这是代码级 bug，不是设计缺陷

质疑方指出：

```python
kept = sum(1 for r in results if r.get("temp_side_this_game") == r["winner"])
```

`r.get("temp_side_this_game")` 是字符串（`"black"` / `"white"`），`r["winner"]` 是整数（`1` / `2`），字符串永远不等于整数，`kept` 永远为 0。

**辩护：这个 bug 确实存在，质疑方的分析完全正确。但需要明确以下几点：**

1. **这是统计输出代码的实现 bug，不是设计逻辑的 bug。** 核心过滤逻辑 `should_keep_game` 的判定是正确的——`temp_side_this_game == "black" and winner == 1` → `True`，不存在类型比较错误。统计输出是辅助性的 CLI 打印信息，不影响任何训练样本的生成或过滤行为。

2. **修复方案简单明确**，只需将统计代码改为：

```python
if filter_active:
    kept = sum(
        1 for r in results
        if (r.get("temp_side_this_game") == "black" and r["winner"] == 1)
        or (r.get("temp_side_this_game") == "white" and r["winner"] == 2)
    )
```

或者更优雅地复用 `should_keep_game` 的语义（在 `results` 列表的 schema 上构建等价判定）：

```python
if filter_active:
    kept = sum(
        1 for r in results
        if r.get("temp_side_this_game") in ("black", "white")
        and (
            (r["temp_side_this_game"] == "black" and r["winner"] == 1)
            or (r["temp_side_this_game"] == "white" and r["winner"] == 2)
        )
    )
```

3. **质疑方说"这暴露了数据格式不统一的深层问题"——我不认同。** `results` 是 `start_ai_vs_ai` 的运行时返回值（内存中的 `List[dict]`），`record` 是 `record.json` 反序列化后的字典。两者的 schema 不同是自然的——一个是 API 返回值，一个是持久化格式。它们服务于不同的上下文，不需要统一。

**结论**：这是一个实现级 bug，一行代码即可修复，不影响设计架构的正确性。质疑方将其标为"致命"是过度定级。

---

## 二、架构设计相关质疑的辩护

### 2.1 过滤逻辑注入层级——在 SampleCreator 做对局级过滤是合理的职责划分

质疑方认为 `SampleCreator` 的核心职责是"从 record.json 生成训练样本"，对局级过滤是"职责错位"。

**辩护：过滤是样本生成管线的前置环节，放在 SampleCreator 中是自然的。**

1. **SampleCreator 已经在做对局级过滤了。** 查看 `sample_creator.py` 第 78-81 行：

```python
winner = record.get("winner", 0)
if winner == 0 or winner == 3:
    continue
```

跳过平局和未结束的对局——这**就是对局级过滤**。`record_filter` 参数只是将这一模式从硬编码泛化为可注入。如果"跳过平局"不是职责错位，那"跳过贪婪方胜局"同样不是。

2. **质疑方建议"在文件发现阶段过滤"——这反而更不合理。** 文件发现阶段（`create_from_round_dir` 扫描子目录时）只有文件路径信息，没有 `record.json` 的内容。要判定是否保留，必须读取并解析 `record.json`——这恰恰是 `create` 方法已经在做的事。在文件发现阶段做过滤，要么需要提前读取文件（与 `create` 重复 I/O），要么需要引入额外的元数据索引（过度设计）。

3. **过滤时机问题被夸大。** 质疑方称"文件 I/O + JSON 反序列化的成本已经付出了"。但实际上：
   - `record.json` 是 `SampleCreator.create` 的**必经路径**——它需要读取 `record` 来提取 `history`、`winner` 等字段，无论是否过滤都要读。
   - 过滤判定发生在 `json.load` 之后、mode 分派之前——此时 `record` 已在内存中，`should_keep_game` 只是 4 行条件判断，开销为零。
   - 被过滤的对局不会进入 `_build_winning_move_samples` / `_build_winner_path_samples` / `_augment_sample`——这些才是计算密集的步骤。过滤确实节省了后续处理的开销。

**结论**：`record_filter` 在 `SampleCreator.create` 中的注入位置是合理的，与既有的平局/未结束过滤逻辑一脉相承，不构成职责错位。

### 2.2 game_filter.py 模块过轻——独立模块是对关注点分离的尊重

质疑方认为 `should_keep_game` 只有一个 4 行纯函数，不值得独立为模块，建议放入 `move_sampler.py`。

**辩护：独立模块是正确的组织决策，理由如下：**

1. **关注点不同。** `move_sampler.py` 的职责是"温度采样的力学实现"——温度变换、加权采样、截断、按步判定。`should_keep_game` 的职责是"对局级过滤判定"——根据采样方和获胜方的关系决定保留/丢弃。前者是"怎么采"，后者是"采完了怎么用"。将过滤逻辑放入采样模块，是在"怎么采"的模块中混入"怎么用"的逻辑，恰恰违反了单一职责原则。

2. **依赖方向不同。** `move_sampler.py` 是底层模块，被 `game_manager.py`、`engine.py` 依赖。`should_keep_game` 是上层逻辑，被 `create_samples.py` 和 `sample_creator.py` 消费。将过滤逻辑放入 `move_sampler.py` 会导致底层模块感知上层需求，依赖方向倒置。

3. **质疑方说"未来也几乎不可能扩展"——这恰恰说明独立模块的成本极低。** 一个 4 行函数的模块，维护成本趋近于零。但如果将它塞进 `move_sampler.py`，每当修改采样逻辑时都要绕过过滤代码，增加了认知负担。

4. **质疑方说"调用方只需要改一下 import 路径"——但 import 路径承载语义。** `from ai.game_filter import should_keep_game` 比 `from ai.move_sampler import should_keep_game` 更能传达意图："我在做对局级过滤"，而非"我在做温度采样"。

**结论**：模块的独立性不以行数论价值，而以职责论价值。`game_filter.py` 承载"对局级过滤"这一独立关注点，与 `move_sampler.py` 的"温度采样力学"职责清晰分离。

### 2.3 双轨管线未统一考虑——已在 1.1 中充分辩护

核心论点：PRD 明确排除了 Evaluator 管线的过滤需求。质疑方的"统一考虑"诉求超出了 PRD 范围，属于增量需求。

补充一点：质疑方说"方案未评估替代方案的利弊"——但设计文档不是论文，不需要穷举所有可能路径。设计文档的职责是给出一个**满足 PRD 的可行方案**，而非论证"这是所有可能方案中的最优解"。在 PRD 明确限界的前提下，只改 `SampleCreator` 是最保守、最安全的选择。

---

## 三、API 设计相关质疑的辩护

### 3.1 `_play_single_game` 返回值从 int 改为 dict — 合理的结构化演进

质疑方列举了三点担忧：测试用例重写、解构模式变更、类型安全退化。

**辩护：**

1. **"所有测试用例需要重写"——这是夸大其词。** `_play_single_game` 是 `AIGameManager` 的 `@staticmethod`，测试中通过 mock 其返回值来测试 `start_ai_vs_ai` 和 `start_ai_vs_ai_random`。变更后，mock 返回值从 `int` 改为 `dict`，每个 mock 点的修改量是一行代码。这不是"整个测试套件的返工"，而是可预期的适配工作量。

2. **解构模式变更是必要的，且改动可控。** `start_ai_vs_ai` 中从 `winner = ...` 改为 `result = ...; winner = result["winner"]`，`start_ai_vs_ai_random` 中同理。这些改动是机械性的，不涉及逻辑变更。

3. **类型安全——质疑方的建议（引入 TypedDict/NamedTuple）本身是合理的，但不构成推翻设计的理由。** 可以在实现阶段引入 `NamedTuple` 来约束返回结构：

```python
class SingleGameResult(NamedTuple):
    winner: int
    temp_side_this_game: Optional[str]
```

这是实现细节的优化，不影响设计架构。

4. **质疑方的"更合理做法"是保持返回 int，仅通过 DataRecorder 持久化。** 但这会导致 `start_ai_vs_ai` 的 `results` 列表中缺失 `temp_side_this_game` 信息——而 `create_samples.py` 恰恰需要这个信息来输出过滤统计。如果返回值不改，统计功能要么无法实现，要么需要从 `results` 列表回溯 `record.json` 文件重新读取——这反而更复杂。

**结论**：返回值变更的破坏面是可预期、可控制的。质疑方将其标为"严重"是基于最坏情况的估计，而实际改动量是有限的。

### 3.2 `start_ai_vs_ai` 返回值结构被静默扩展 — 不是"静默"，是增量扩展

质疑方指出返回值 schema 没有类型注解，是"脆弱的隐性契约"。

**辩护：**

1. **当前代码本身就是隐性契约。** `start_ai_vs_ai` 返回 `List[dict]`，每项含 `{"game": int, "winner": int}`——这本身就没有 TypedDict 约束。新增 `temp_side_this_game` 键的"隐性"程度与既有的 `game`、`winner` 键完全一致。这不是本设计引入的新问题，而是既有代码风格的一致延续。

2. **质疑方的调查证实了兼容性。** 审查所有消费方：`create_samples.py` 只读 `r["winner"]`，`self_play_generator.py` 只读 `r["winner"]`，`battle.py` 不消费内部字段。新增键不破坏任何现有消费方。

3. **如果需要更强的类型保障**，可以在实现阶段引入 TypedDict。这属于代码质量改进，不影响设计架构。

### 3.3 `record_filter: Callable[[dict], bool]` 过于宽泛 — 宽泛正是设计意图

质疑方认为这个签名"没有对输入 dict 的 schema 做约束"，"当项目只有一个过滤函数时是过度设计；当未来有更多需求时又不够严谨"。

**辩护：**

1. **宽泛是 `record_filter` 的设计意图。** `SampleCreator` 不应关心过滤函数的具体实现——它只关心"给你一个 record，告诉我留不留"。这是标准的策略模式（Strategy Pattern），解耦了过滤规则的定义与使用。

2. **"当项目只有一个过滤函数时是过度设计"——不对。** `record_filter` 的抽象层级刚好匹配需求：一个可选的过滤钩子。如果用 `should_keep_game` 硬编码替代 `record_filter`，`SampleCreator` 就与 `game_filter.py` 产生直接依赖——这恰恰是质疑方在 2.2 中批评的"耦合"问题。`record_filter` 参数使 `SampleCreator` 保持了对过滤策略的无知，是低耦合的体现。

3. **"当未来有更多需求时不够严谨"——YAGNI。** 当前 PRD 只需要对局级过滤，不需要组合、链式、条件化过滤。如果未来有新需求，届时再引入 `RecordFilter` 协议类或 `List[Callable]` 不迟。过早引入抽象层反而增加理解成本。

4. **关于输入 dict 的 schema——`record` 是 `json.load(f)` 的结果，其结构由 `DataRecorder` 的写入逻辑定义。** 调用方（`create_samples.py`）同时持有 `should_keep_game` 的源码和 `record.json` 的 schema 定义，不存在"不知道该检查哪些字段"的问题。

---

## 四、数据模型相关质疑的辩护

### 4.1 `temp_side_this_game` 始终写入 record.json — 合理的显式优于隐式策略

质疑方认为无条件写入 null 是"噪声污染"，应改为条件写入以与 `strategy`/`temperature` 一致。

**辩护：**

1. **`temp_side_this_game` 与 `strategy`/`temperature` 的性质不同，不应强行套用同一写入策略。** `strategy`/`temperature` 是**每步**的元数据——每步要么是 argmax（无需温度字段），要么是 sampling（需要温度字段），步与步之间可以不同，条件写入是自然的。`temp_side_this_game` 是**每局**的元数据——一局要么有采样方（`"black"`/`"white"`/`"both"`），要么没有采样（`None`），是局级的全局属性。局级字段无条件写入，使得任何读取 `record.json` 的代码都能明确知道"这局有没有采样方"，无需通过"字段不存在"来推断。

2. **显式 null 比隐式缺失更安全。** 如果条件写入，消费方需要处理两种等价语义："字段不存在"= 无采样 vs "字段为 null" = 无采样。`record.get("temp_side_this_game")` 在两种情况下都返回 `None`，看似等价——但如果某天写入逻辑出了 bug，本应写入 `"black"` 的字段被遗漏，消费方无法区分"本就无采样"和"应该有采样但字段丢失"。无条件写入使得 null 成为明确的语义信号：**我确认这局无采样**。

3. **质疑方说"99% 的场景下该字段为 null"——这个数字缺乏依据。** 实际上，只要用户传了 `--temp-side` 参数（包括 `both`），该字段就有非 null 值。只有在 `--no-temp-sampling` 和 GUI 不传 `temp_config` 时才为 null。null 的比例取决于使用场景，不能一概而论为"99%"。

4. **JSON 中的 null 对文件大小的影响可忽略。** 一个 `"temp_side_this_game": null` 占 27 字节，即使 1000 局也只多 27KB。这不是"噪声污染"。

### 4.2 `temp_side_this_game` 与 `history` 中的 `strategy` 存在信息冗余 — 不是冗余，是不同粒度的摘要

质疑方认为 `history` 中每步的 `strategy` 字段已隐含了采样方信息，顶层字段是冗余存储。

**辩护：**

1. **从 `history` 推导 `temp_side_this_game` 需要 O(n) 遍历且逻辑脆弱。** 需要遍历全部 `history`，统计哪一步有 `strategy="sampling"`，再映射到采样方——而且还要处理 `temp_side="both"` 的场景（双方都有 `sampling` 步）。这个推导过程比直接读取顶层字段复杂得多，且容易出错。

2. **质疑方的推导逻辑本身就不完备。** "偶数步有 sampling → `temp_side_this_game="black"`"——这个推导假设黑方总是先手（偶数步=黑方），但未考虑 `start_ai_vs_ai_random` 中引擎执方是随机的。在 random 模式下，`history` 中黑方步和白方步的编号关系不变（黑方始终在偶数索引），但要正确推导仍需理解索引与执方的映射。顶层字段直接给出了答案，避免了消费方的重复推导。

3. **摘要字段是常见的数据建模实践。** 就像数据库中的物化视图或缓存列——虽然可以从明细数据推导，但顶层摘要提供了 O(1) 的访问路径。`winner` 字段也可以从 `history` 的最后一步推导出来，但没有人认为 `winner` 是冗余。

4. **质疑方担心的"不一致"风险被高估了。** `temp_side_this_game` 的写入点和 `strategy` 的写入点都在 `_play_single_game` 中，由同一段逻辑控制。两者不一致的唯一可能是代码 bug——而 bug 导致的任何不一致都可以通过同样的手段检测，不因为字段冗余就变得更难发现。

### 4.3 record.json schema 缺少版本化 — 通用问题，不应由本设计单独承担

质疑方指出 `record.json` 没有 `schema_version` 字段。

**辩护：**

1. **这是既有代码的通用问题，不是本设计引入的。** `record.json` 从第一版就没有 `schema_version`，之前追加 `step_id`、`strategy`/`temperature`/`top_k`/`sampling_probs` 时也没有人提出这个问题。如果现在要加 `schema_version`，那应该是一个独立的重构任务，涵盖所有历史 schema 变更，而不是仅由本设计承担。

2. **当前的字段级向后兼容策略是有效的。** `record.get("temp_side_this_game")` 在旧格式数据上返回 `None`，等同于 null 语义，行为与修改前一致。这种基于 `.get()` 的容错模式在 `SampleCreator` 中已有先例（如 `step_id` 的处理），证明是可维护的。

3. **质疑方说"随着 schema 持续膨胀，未来任何字段重命名或语义变更都将无法可靠检测"——这是假设性风险，不是当前问题。** 当前所有变更都是纯增量（新增字段，不修改/删除已有字段），增量变更是天然向后兼容的。如果未来出现破坏性变更（字段重命名/语义变更），届时引入版本化是适时的。

---

## 五、兼容性相关质疑的辩护

### 5.1 `start_ai_vs_ai_random` 的结果列表结构冲突 — 不存在冲突

质疑方指出 `temp_side_this_game` 与 `ai1_color`/`ai2_color` 的语义关系未显式建模。

**辩护：**

1. **两者服务于不同的上下文，不需要显式建模关系。** `ai1_color`/`ai2_color` 回答"哪个引擎执哪种颜色"——这是对战层面的信息。`temp_side_this_game` 回答"这局哪个颜色在做采样"——这是训练数据层面的信息。两者的组合确实可以推导出"哪个引擎在做采样"，但这不是 PRD 需求——PRD 只关心"采样方是否获胜"，不关心"采样方是哪个引擎"。

2. **质疑方提到的"统计采样方引擎 vs 贪婪方引擎的胜率"——这不是当前需求。** `start_ai_vs_ai_random` 的设计意图是不同模型之间的对战评估，与温度采样过滤是正交的关注点。如果未来需要统计引擎级胜率，可以增量添加，不影响当前设计。

3. **`battle.py` 的使用场景不涉及温度采样。** 查看 `battle.py`，它调用 `start_ai_vs_ai_random` 时**不传 `temp_config`**，所以 `temp_side_this_game` 始终为 `None`。不存在语义冲突。

### 5.2 Evaluator 管线不感知过滤 — 已在 1.1 中充分辩护

补充：质疑方说"已生成的'坏样本'无法追溯清除"——但 `Evaluator` 生成的 `sample.json` 是训练周期的中间产物，每个训练周期都是独立的。如果未来在 Evaluator 管线中也加入过滤，只需在 `evaluate_round` 之前删除不需要的 `record.json`，或在 `_evaluate_files` 中过滤文件列表——方案是直截的，不存在"无法追溯"的技术障碍。

### 5.3 GUI AI 内战路径 — 质疑方自己承认"目前安全"

质疑方审查了 GUI 路径后得出结论："目前安全，但同样是隐性契约"。

**辩护：** "目前安全"就是兼容的。GUI 路径不传 `temp_config`，`_play_single_game` 返回 `{"winner": ..., "temp_side_this_game": None}`，`start_ai_vs_ai` 的结果列表新增了 `temp_side_this_game` 键，GUI 不消费该键。这是一个增量扩展，不破坏任何现有行为。

至于"隐性契约"——这是所有动态类型语言的共有特征。如果要在本设计中解决，那是代码风格的全面改造，不是本设计应承担的。

---

## 六、测试覆盖相关质疑的辩护

### 6.1 缺少 Evaluator 管线的过滤测试 — 不需要，因为 Evaluator 管线不做过滤

PRD 明确排除 Evaluator 管线的过滤行为，因此不需要为 Evaluator 管线编写过滤测试。测试的是"Evaluator 管线行为不变"——这由回归测试 BC-04 覆盖。

### 6.2 缺少 `run_self_play_and_evaluate` 的集成测试 — 回归测试已覆盖

BC-04 "train_main.py --cycle 运行，行为与修改前一致"就是 `run_self_play_and_evaluate` 的回归测试。质疑方说"没有定义'一致'的具体检查项"——这是测试实现阶段的工作，不是设计文档的职责。设计文档给出测试用例的覆盖范围和意图，具体断言在测试代码中编写。

### 6.3 缺少 `SelfPlayCycle.run()` 的回归测试 — 同 6.2

BC-04 已覆盖。质疑方要求在设计文档中定义"一致"的具体检查项，这是对设计文档角色的误解。

### 6.4 缺少并发/中断场景测试 — 质疑方的场景假设有误

质疑方称"中断时 results 列表可能包含部分 dict、部分尚未修改的旧格式"——这是不可能的。代码变更是整体部署的，不会出现"部分对局在代码变更前已完成"的情况。`_play_single_game` 的返回值要么全是 int（旧代码），要么全是 dict（新代码），不会混合。

`KeyboardInterrupt` 处理逻辑中，`results` 列表的元素都是 `start_ai_vs_ai` 的返回值，其结构在新代码中是统一的。不存在类型不一致的风险。

### 6.5 `should_keep_game` 缺少异常输入测试 — 防御性编程的边界可以讨论，但不构成设计缺陷

质疑方建议测试空字典、非法 `temp_side` 值、`winner` 类型错误等异常输入。

**辩护：**

1. **`should_keep_game` 的输入来自 `json.load(f)`——即 `record.json` 的反序列化结果。** 这些数据由 `DataRecorder` 生成，结构是确定的。空字典意味着 `record.json` 损坏或为空——这应该在 `SampleCreator.create` 的前置校验中处理，而不是在 `should_keep_game` 中处理。

2. **非法 `temp_side` 值不会出现。** `temp_side_this_game` 的值由 `resolve_temp_side` 生成，其取值被 `SUPPORTED_TEMP_SIDES` 白名单约束。如果 `record.json` 被手动编辑出非法值，那是用户操作错误，不应该由代码兼容。

3. **防御性编程是好的实践，但不构成设计评审的阻断项。** 可以在实现阶段补充异常输入测试，不影响设计架构。

---

## 七、边界条件与正确性相关质疑的辩护

### 7.1 过滤统计 bug — 已在 1.2 中坦诚承认并给出修复方案

### 7.2 `filter_active` 判定依赖隐式条件 — 依赖链是清晰的

质疑方称"一条隐蔽的依赖链"：`filter_active` → `temp_config.temp_side` → `results[].temp_side_this_game` → `_play_single_game` → `resolve_temp_side`。

**辩护：** 这不是"隐蔽的依赖链"，而是**正常的功能调用链**。`create_samples.py` 的过滤统计需要知道"哪些对局的采样方获胜了"，这个信息由 `_play_single_game` → `start_ai_vs_ai` → `results` 透传而来。每一环的职责是清晰的，数据流向是单向的。如果质疑方认为这是"隐蔽"的，那任何跨函数的数据传递都是"隐蔽"的。

### 7.3 `should_keep_game` 对 `winner=0` 的处理 — 防御性返回 False 是正确的

质疑方指出 `winner=0` 的情况已被 `SampleCreator.create` 跳过，`GF-10` 是不可达路径。

**辩护：**

1. **`should_keep_game` 是独立函数，不假设调用上下文。** 它可以单独调用，不仅限于 `SampleCreator.create`。对 `winner=0` 返回 `False` 是合理的——未结束的对局不应保留。GF-10 测试的是 `should_keep_game` 作为独立函数的正确性，而非其与 `SampleCreator.create` 的交互。

2. **质疑方说"看似完善，实为虚假覆盖率"——这是对测试目的的误解。** GF-10 测试的是 `should_keep_game` 函数的边界条件，不是 `SampleCreator.create` 的集成路径。一个函数的测试用例不应因调用方已做前置过滤而被视为"虚假"。

### 7.4 单方采样 + 平局的交互 — 双重过滤是防御性设计，不是职责不清

质疑方认为 `should_keep_game` 和 `SampleCreator.create` 都过滤平局，职责边界模糊。

**辩护：**

1. **`SampleCreator.create` 的平局过滤是通用逻辑**——无论是否启用温度采样，平局都不产生样本。这是 `SampleCreator` 的既有行为，不是本设计新增的。

2. **`should_keep_game` 的平局过滤是专用逻辑**——在单方采样模式下，平局也应被丢弃（采样方未获胜）。这确保了 `should_keep_game` 作为独立函数的完备性——即使调用方不做平局过滤，`should_keep_game` 也能正确判定。

3. **双重过滤是"幸运的正常路径"，不是设计缺陷。** 两个独立的过滤规则各自完备，组合使用时不产生冲突。这比"假设调用方已做前置过滤"更安全。

---

## 八、可维护性相关质疑的辩护

### 8.1 过滤规则的"真相源"分散在三处 — 三处各有职责，不是分散

质疑方说过滤逻辑分布在 `should_keep_game`、`SampleCreator.create`（跳过平局/未结束）、`create_samples.py`（`filter_active` 判定）。

**辩护：**

1. **`SampleCreator.create` 的平局/未结束过滤是既有逻辑**——它不是"对局级样本过滤"设计的一部分，而是 `SampleCreator` 从诞生之日起就有的行为。将既有逻辑计入"真相源分散"的账上是不公平的。

2. **`should_keep_game` 是过滤规则的唯一定义点。** 它完整地编码了"什么对局该保留"的判定逻辑。`filter_active` 不是过滤规则，而是**是否激活过滤的开关**——它是业务决策（"什么时候需要过滤"），不是过滤逻辑本身。

3. **如果未来需要修改过滤规则**（如"平局也保留"），只需修改 `should_keep_game`。`SampleCreator.create` 的平局过滤是独立于本设计的既有逻辑，是否修改取决于独立的需求评估。这不是"三处同步修改"的问题。

### 8.2 `temp_side_this_game` 的语义重载 — 不是重载，是同一语义的两个使用场景

质疑方说 `temp_side_this_game` 同时承载"记录语义"和"过滤语义"。

**辩护：** `temp_side_this_game` 只有一个语义：**这局哪个颜色在做温度采样。** 这是一个事实描述（记录语义）。"过滤"是 `should_keep_game` 基于这个事实做出的决策，不是 `temp_side_this_game` 自身的语义。质疑方将"基于数据做决策"误读为"数据承载了决策语义"——这是对关注点分离的误解。

如果未来过滤规则变了，`temp_side_this_game` 的值不变，`should_keep_game` 的逻辑变了——这恰恰说明两者是解耦的。

### 8.3 `create_samples.py` 与 `train_main.py` 的行为分化 — 这是 PRD 的意图

PRD 第 3.2 节排除范围：

> | 其他脚本的过滤行为 | 明确约束：不影响 create_samples.py 以外的脚本 |

行为分化是 PRD 的显式要求，不是设计的疏忽。`create_samples.py` 是独立 CLI 工具，`train_main.py` 是训练管线——两者服务于不同的工作流，有不同的质量保障策略是合理的。

质疑方说"用户可能困惑"——这可以通过 CLI 输出中的提示信息解决。设计方案已在 `filter_active` 判定后输出了明确的提示：

```
样本过滤: 启用 (采样方获胜过滤)
保留对局: X
过滤对局: Y
```

或：

```
样本过滤: 未启用 (非单方采样模式)
```

这些输出足以消除用户的困惑。

---

## 九、可扩展性相关质疑的辩护

### 9.1 `record_filter` 闭包黑盒无法组合 — YAGNI，未来可增量扩展

质疑方建议定义 `RecordFilter` 协议类，支持 `AndFilter`/`OrFilter` 组合。

**辩护：**

1. **当前 PRD 只有单一过滤规则，没有组合需求。** 引入协议类和组合机制是典型的过度设计。Python 的 `Callable[[dict], bool]` 天然支持 lambda 组合：

```python
combined_filter = lambda r: should_keep_game(r) and custom_filter(r)
```

这虽然不如 `AndFilter` 优雅，但足以满足当前需求。

2. **如果未来有组合需求**，可以在 `SampleCreator.create` 中将 `record_filter` 改为 `List[Callable]`，用 `all()` 组合——这是一个向后兼容的扩展（单个 `Callable` 包装为 `[Callable]` 即可），不需要在设计阶段预留。

3. **质疑方的"更合理做法"（`List[Callable]` + all/any）** 本质上是另一种形式的过度设计——在没有第二个过滤函数的时候就为组合预留接口，增加了当前实现的复杂度。

### 9.2 只支持对局级过滤，未来可能需要步级过滤 — YAGNI

质疑方认为设计没有为步级过滤预留扩展点。

**辩护：**

1. **PRD 第 3.2 节排除范围明确排除了步级过滤：**

> | 样本内部过滤（步级过滤） | 过滤粒度为对局级 |

设计方案遵循 PRD，不预留 PRD 排除的扩展点。

2. **步级过滤的扩展不需要预留接口。** `SampleCreator` 的 `create` 方法内部已经有步级逻辑（`_build_winner_path_samples` 只取获胜方的步骤）。步级过滤可以在 mode 分派阶段自然地扩展，不需要修改 `record_filter` 的签名。

### 9.3 Evaluator 管线完全无法扩展 — 误导性论断

质疑方说"修改 Evaluator 的风险和成本远高于修改 SampleCreator，方案把技术债留给了 Evaluator 管线"。

**辩护：**

1. **Evaluator 的 `evaluate_round` 方法接受 `round_dir`，其内部扫描子目录收集 `record_files`。** 要在 Evaluator 中加过滤，只需在 `evaluate_round` 中过滤 `record_files` 列表——这与 `SampleCreator.create_from_round_dir` 中过滤文件列表的模式完全一致。修改量很小，风险可控。

2. **"技术债"是质疑方的主观判断。** 不在当前 PRD 范围内实现的功能不是"技术债"，而是"未实现的未来需求"。技术债是指为了短期交付而采取的次优方案所累积的维护成本——而当前设计在 PRD 范围内是最优方案（最保守、最安全、最符合"不影响其他脚本"的约束）。

---

## 十、设计遗漏相关质疑的辩护

### 10.1 未考虑 `start_ai_vs_ai_random` + 单方采样的交互 — 不在 PRD 范围内

`start_ai_vs_ai_random` 被 `battle.py` 使用，而 `battle.py` 不传 `temp_config`。`create_samples.py` 使用的是 `start_ai_vs_ai`（固定执方），不使用 `start_ai_vs_ai_random`。因此，`start_ai_vs_ai_random` + 单方采样的交互不在当前 PRD 的触发路径上。

如果未来 `start_ai_vs_ai_random` + 单方采样的组合成为需求，`temp_side_this_game` + `ai1_color` 的组合推导是直截的，不需要在设计阶段预先建模。

### 10.2 未考虑 `should_keep_game` 在 Evaluator 管线中的替代方案 — 已在 1.1 中辩护

PRD 排除 Evaluator 管线的过滤需求，因此不需要为 Evaluator 设计替代方案。

### 10.3 未处理 `--temp-side random` 场景下的统计准确性 — 已在 1.2 中修复

统计代码的 bug 对所有单方采样模式（black、white、random）都存在，修复方案已在 1.2 中给出。`random` 模式没有额外的统计准确性问题——`results` 中每项的 `temp_side_this_game` 已由 `resolve_temp_side` 在每局开始时解析为 `"black"` 或 `"white"`，统计代码只需正确处理类型即可。

---

## 十一、总结

### 质疑方"核心问题清单"的逐条回应

| # | 质疑方问题 | 严重性 | 辩护结论 |
|---|-----------|--------|---------|
| 1 | Evaluator 管线无过滤 | 🔴 致命 → 🟢 不适用 | PRD 显式排除，不是设计遗漏。如需扩展是增量 PRD。 |
| 2 | 过滤统计 `kept` 计算逻辑有 bug | 🔴 致命 → 🟡 实现级 bug | 确认存在，一行修复，不影响核心过滤逻辑。降级为实现问题。 |
| 3 | `_play_single_game` 返回值类型变更破坏面被低估 | 🟠 严重 → 🟡 可控 | 改动量有限且机械，可引入 NamedTuple 增强类型安全。 |
| 4 | `temp_side_this_game` 无条件写入 vs `strategy` 条件写入不一致 | 🟠 严重 → 🟢 合理设计 | 局级字段与步级字段性质不同，显式 null 比隐式缺失更安全。 |
| 5 | 过滤职责错配 | 🟠 严重 → 🟢 合理 | `record_filter` 与既有的平局过滤逻辑一脉相承。 |
| 6 | `game_filter.py` 过轻 | 🟡 中等 → 🟢 合理 | 职责独立，依赖方向正确，维护成本趋近于零。 |
| 7 | 两条管线行为分化未文档化 | 🟡 中等 → 🟢 PRD 已界定 | 行为分化是 PRD 的显式要求，CLI 输出提示信息已缓解用户困惑。 |
| 8 | `record_filter` 闭包黑盒无法组合 | 🟡 中等 → 🟢 YAGNI | 当前无组合需求，lambda 组合足以应对，未来可增量扩展。 |
| 9 | 测试覆盖遗漏 | 🟡 中等 → 🟢 已覆盖 | Evaluator 无需过滤测试，异常输入可在实现阶段补充。 |
| 10 | `should_keep_game` 与 `SampleCreator.create()` 职责边界模糊 | 🟢 低 → 🟢 不成立 | 双重过滤是防御性设计，各自完备不冲突。 |

### 设计方案的核心优势

1. **严格遵循 PRD 范围界定**——仅影响 `create_samples.py`，其他脚本行为不变，完全满足 PRD 的约束条件。

2. **变更面最小化**——`record_filter` 作为可选参数注入 `SampleCreator`，不改变既有 API 的默认行为；`should_keep_game` 作为独立纯函数，可单独测试；`DataRecorder.save_game_result` 的新增参数有默认值，向后兼容。

3. **过滤逻辑集中化**——`should_keep_game` 是过滤规则的唯一定义点，`record_filter` 是过滤规则的注入通道，`filter_active` 是过滤开关的判定条件。三者职责清晰，各司其职。

4. **向后兼容性强**——旧格式 `record.json`（无 `temp_side_this_game` 字段）通过 `.get()` 返回 `None`，等同于 null 语义，`should_keep_game` 返回 `True`，行为与修改前一致。

5. **可增量扩展**——`record_filter` 的 `Callable[[dict], bool]` 接口天然支持未来的过滤策略替换；`Evaluator` 管线的过滤可作为增量需求通过新 PRD 追踪。

### 最终立场

设计方案在 PRD 的范围内是**完备的、正确的、可实施的**。唯一确认的代码级 bug（统计输出中的类型比较错误）可在实现阶段一行修复，不构成推翻设计的理由。质疑方的两个"致命"问题中，#1 基于 PRD 误读，#2 是实现 bug 而非设计缺陷。其余问题属于代码质量改进建议，可在实现和测试阶段逐步纳入，不影响设计架构的合理性。

**建议：通过设计方案，进入实现阶段，同步修复统计输出的 bug。**
