# C++ 模块接口设计

## 1. GameSample — 训练样本结构体

新增数据类，表示一条 AGZ 格式训练样本。

```cpp
struct GameSample {
    std::vector<float> state;   // 226 维：[225 棋盘编码 + 1 当前玩家标识]
    std::vector<float> pi;      // 225 维：MCTS visit counts 归一化后的概率分布
    float z;                    // 终局标签：+1（当前玩家胜）、-1（当前玩家负）、0（平局）
};
```

### 行为约定

- `state` 使用 cnn_v11 的相对视角编码（我方棋子 / 对方棋子 / 我方颜色）
- `pi` 由 visit_counts 归一化得到（`pi[i] = N[i] / ΣN`），sum ≈ 1.0
- `z` 的视角与 `state[225]` 表示的当前玩家一致

---

## 2. SampleAugmentor — D4 数据增强

新增工具类，对单条 `GameSample` 执行 8 种棋盘对称变换。

### 对外接口

```cpp
class SampleAugmentor {
public:
    /** 对单条样本执行 D4 增强，返回 8 条变体样本 */
    static std::vector<GameSample> augment(const GameSample& sample);

    /** 对 225 维向量执行指定变换（tid: 0~7） */
    static std::vector<float> transform_225(const std::vector<float>& vec, int tid);

    /** 返回变换 tid 将原始索引映射到的新索引 */
    static int transform_index(int idx, int tid);
};
```

### 变换定义

| tid | 变换 | 映射公式 |
|-----|------|---------|
| 0 | 恒等 | `(r, c)` |
| 1 | 顺时针 90° | `(c, 14 - r)` |
| 2 | 180° | `(14 - r, 14 - c)` |
| 3 | 顺时针 270° | `(14 - c, r)` |
| 4 | 左右镜像 | `(r, 14 - c)` |
| 5 | 上下镜像 | `(14 - r, c)` |
| 6 | 主对角线翻转 | `(c, r)` |
| 7 | 反对角线翻转 | `(14 - c, 14 - r)` |

### 行为约定

- `augment` 返回的 8 条样本中 `z` 值不变（对称变换不影响对局结果）
- `state[225]`（玩家标识）保持不变
- `state[0:225]` 和 `pi[0:225]` 按变换映射

---

## 3. GameResult — 扩展

在现有 `GameResult` 中新增样本字段。

### 对外接口（追加）

```cpp
struct GameResult {
    int game_index;
    int winner;           // 0=平局, 1=黑胜, 2=白胜
    int model1_color;     // model1 执方为 1=黑 或 2=白（自对弈时 model1 == model2）
    int model2_color;
    int move_count;
    std::vector<GameSample> samples;  // 新增：本局产生的训练样本（已 D4 增强）
};
```

### 行为约定

- `samples` 在 C++ 游戏线程完成对局后填充（`run_single_game` 内部）
- 样本已包含 D4 增强：若对局产生 N 步原始样本，`samples.size() == 8 * N`
- 平局对局的 `samples` 为空（`z == 0` 的样本无训练价值）

---

## 4. GameRunner — 自对弈模式改造

在现有 `GameRunner` 基础上新增自对弈模式和样本采集能力。

### 对外接口（追加）

```cpp
class GameRunner {
public:
    // --- 现有接口不变 ---
    GameRunner(const std::string& model1_path,
               const std::string& model2_path,
               int num_games,
               const GameConfig& config,
               int num_workers = 0);

    // --- 新增：自对弈模式 + 样本采集构造器 ---
    GameRunner(const std::string& model_path,     // 单一模型路径
               int num_games,
               const GameConfig& config,
               int num_workers,
               bool collect_samples);             // 是否开启样本采集

    // --- 现有接口不变 ---
    void start();
    int get_completed_count() const;
    int get_total_count() const;
    bool is_finished() const;
    void wait_completion();
    std::vector<GameResult> get_results() const;
};
```

### 行为约定

- 自对弈模式：`model1_path == model2_path`，双方使用同一模型
- 执方分配为**交替模式**（非随机）：`game_index % 2 == 0` 时 model1 执黑，`game_index % 2 == 1` 时 model1 执白
- `collect_samples == true` 时，`run_single_game()` 内部：
  1. 每步 MCTS 搜索后，记录 `(state_before_search, visit_counts)`
  2. 对局结束后，若 `winner != 0`（非平局），为每步计算 z 值（从落子方视角）
  3. 调用 `SampleAugmentor::augment()` 生成 8 条变体
  4. 将增强后样本填入 `GameResult.samples`
- `collect_samples == false` 时行为与现有完全一致（向后兼容）

### 样本采集流程（`run_single_game` 内）

```
对局开始
  → 初始化 samples_raw = []（存储原始样本 {state, visit_counts, player}）
  → 对局循环：
      → 记录 current_state（226 维，落子前）
      → MCTS 搜索 → 得到 visit_counts
      → samples_raw.push_back({current_state, visit_counts, current_player})
      → 温度采样选动作 → 落子 → 终局判定
  → 对局结束
  → 若 winner == 0（平局）：不生成样本
  → 否则：
      for each (state, visit_counts, player) in samples_raw:
          z = +1 if winner == player else -1
          pi = normalize(visit_counts)
          raw_sample = {state, pi, z}
          augmented = SampleAugmentor::augment(raw_sample)  // 8 条
          result.samples.insert(augmented)
```

---

## 5. bindings.cpp — pybind11 绑定变更

### 新增绑定

```cpp
// GameSample 结构体
py::class_<GameSample>(m, "GameSample", "单条训练样本")
    .def_readonly("state", &GameSample::state, "226 维棋盘状态")
    .def_readonly("pi", &GameSample::pi, "225 维 MCTS 概率分布")
    .def_readonly("z", &GameSample::z, "终局标签 ±1/0");

// GameResult.samples 字段
// 在现有 GameResult 绑定中追加：
// .def_readonly("samples", &GameResult::samples, "本局训练样本列表")

// GameRunner 新增构造器重载
// 在现有 GameRunner 绑定中追加第二个 __init__ 签名
```

### 行为约定

- `GameSample` 的 `std::vector<float>` 自动转换为 Python `list[float]`
- `GameResult.samples` 的 `std::vector<GameSample>` 自动转换为 Python `list[GameSample]`
- 新增构造器通过 `py::overload_cast` 或手动 lambda 区分
