# C++ 模块接口设计

## 1. MctsThreadPool — 任务队列模式改造

### 改造原因

现有线程池为"单批次屏障"模式：`run_parallel(task_fn)` 阻塞等待所有 worker 完成同一批任务后才返回。此模式只允许同时运行一次 MCTS 搜索。并行对局需要多个 MctsSearcher 同时向线程池提交任务。

### 对外接口

```cpp
class MctsThreadPool {
public:
    // --- 单例生命周期（不变）---
    static MctsThreadPool& instance();
    static void initialize(int num_threads);
    static void shutdown();
    int size() const;

    // --- 新增：任务队列模式 ---
    // 向队列提交一个任务，worker 空闲时自动取走执行
    void submit(std::function<void()> task);

    // 阻塞直到队列中所有已提交的任务执行完毕
    void wait_all();

    // 返回当前队列中待处理任务数
    int pending_count() const;
};
```

### 行为约定

- worker 线程在 `submit` 之前已全部启动（由 `initialize` 保证）
- `submit` 是线程安全的，多个游戏线程可并发调用
- `wait_all` 阻塞调用线程直到队列清空且所有 worker 空闲
- 禁止在 worker 执行的任务中再次调用 `submit`（防止死锁）

---

## 2. NeuralNet — 全局单例化

### 改造原因

当前每个 MctsSearcher 拥有独立的 NeuralNet 实例。并行对局场景下，N 个实例各自持有 GPU 推理线程会导致 GPU 资源争抢。改为全局单例后，所有对局共享一个 GPU 推理线程，攒批效率更高。

### 对外接口

```cpp
class NeuralNet {
public:
    // 单例
    static NeuralNet& instance();

    // 初始化（仅首次调用生效，call_once 保护）
    static void initialize(const std::string& model_path,
                           int batch_size, bool use_gpu);

    // 切换模型（用于对战双方不同模型场景）
    void set_model(const std::string& model_path);

    // 异步批量推理：提交 states，通过 future 获取结果
    std::future<std::pair<std::vector<float>, std::vector<float>>>
    predict_async(const std::vector<float>& state);

    // 同步批量推理（内部攒批）
    std::pair<std::vector<float>, std::vector<float>>
    predict(const std::vector<float>& state);
};
```

### 行为约定

- `initialize` 必须在任何 `predict` 调用之前执行
- `predict_async` 返回 future，GPU 线程完成推理后设置值
- 多线程并发调用 `predict` 安全，内部自动攒批
- `set_model` 切换模型权重；对战中每个游戏线程独立持有 MctsSearcher，模型路径在 MctsSearcher 创建时确定

---

## 3. MctsSearcher — 共享资源注入

### 改造原因

改为接受外部线程池和 NeuralNet 引用，不再自行创建。多个 MctsSearcher 共享同一套资源。

### 对外接口

```cpp
class MctsSearcher {
public:
    MctsSearcher(const std::string& model_path,
                 MctsThreadPool& pool,
                 NeuralNet& nn,
                 int num_simulations,
                 double c_puct,
                 double dirichlet_epsilon,
                 double dirichlet_alpha,
                 double virtual_loss);

    // search 和 advance_root 接口不变
    MctsResult search(const std::vector<int>& state);
    bool advance_root(int action);
};
```

### 行为约定

- 构造时不初始化线程池和 NeuralNet（它们已由 GameRunner 初始化）
- `search` 内部通过 `pool.submit()` 分发 simulation 任务
- 不同 MctsSearcher 实例的 `search` 可并发调用，线程安全由 MctsThreadPool 保证
- `model_path` 仅用于 NeuralNet 模型切换提示；实际推理使用共享 NeuralNet

---

## 4. GameConfig — 配置结构体

### 对外接口

```cpp
struct GameConfig {
    int num_simulations = 800;
    double c_puct = 2.0;
    double dirichlet_epsilon = 0.25;
    double dirichlet_alpha = 0.03;
    int batch_size = 16;
    double virtual_loss = 3.0;
    double temperature = 1.0;
    int temp_steps = 8;
    bool use_gpu = true;
};
```

---

## 5. GameResult — 对局结果

### 对外接口

```cpp
struct GameResult {
    int game_index;       // 局编号 0..N-1
    int winner;           // 0=平局, 1=黑胜, 2=白胜
    int model1_color;     // model1 执方 1=黑, 2=白
    int model2_color;     // model2 执方
    int move_count;       // 本局总步数
};
```

`model1_won` 和 `model2_won` 由 Python 端根据 `winner` 和执方计算，避免 C++ 侧冗余。

---

## 6. GameRunner — 并行对局管理器

### 职责

- 接收两个模型路径、对局数量、配置参数
- 初始化全局资源（线程池、NeuralNet）
- 启动 N 个游戏线程，每个线程运行一局完整对局
- 收集结果，提供进度查询

### 对外接口

```cpp
class GameRunner {
public:
    GameRunner(const std::string& model1_path,
               const std::string& model2_path,
               int num_games,
               const GameConfig& config);

    // 启动所有游戏线程（非阻塞，立即返回）
    void start();

    // 进度查询（线程安全）
    int get_completed_count() const;
    int get_total_count() const;

    // 是否全部完成
    bool is_finished() const;

    // 阻塞直到所有游戏线程完成
    void wait_completion();

    // 获取所有结果（仅在 is_finished() 为 true 后调用）
    std::vector<GameResult> get_results() const;
};
```

### 行为约定

- `start()` 非阻塞：调用后立即返回，游戏线程在后台运行
- `get_completed_count()` 线程安全，可在 Python 主线程中轮询
- `wait_completion()` 阻塞当前线程直到 N 个游戏线程全部 join
- `get_results()` 仅在完成后调用；完成前调用行为未定义
- 若初始化全局资源失败（如模型文件不存在），构造器抛出 `std::runtime_error`
- 每个游戏线程的对局逻辑：
  1. 随机确定 model1 执方（1=黑 / 2=白），model2 执相反方
  2. 创建两个 MctsSearcher（model1 对应执方, model2 对应执方）
  3. 初始化空棋盘 `BoardState`，黑方先手
  4. 循环：当前方 Searcher.search() → 温度采样选动作 → BoardState.apply_action() → 终局判定 → Searcher.advance_root()
  5. 对局结束，产出 `GameResult`，原子追加到结果列表

### 温度采样逻辑

```
前 temp_steps 步：τ = temperature，从 visit_counts 按 N^(1/τ) 概率采样
temp_steps 之后：τ → 0，纯贪婪选择（visit_counts 最大者）
```

与现有 `PolicyValueAiEngine._get_temperature()` 行为一致。
