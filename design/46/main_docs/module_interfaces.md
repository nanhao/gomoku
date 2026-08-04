# C++ 模块对外接口规范

> **面向接口设计**：以下仅列出各模块的公开方法和构造器。不包含私有方法、内部数据成员。

---

## 1. BoardState — 棋盘状态

提供 15×15 五子棋盘的胜负判定与状态查询。棋盘编码：0=空, 1=黑, 2=白，225 维平坦向量。

```
class BoardState
```

### 公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `BoardState(std::vector<int> board, int current_player)` | 从 225 维向量和当前玩家构造 |
| has_winner | `bool has_winner() const` | 棋盘上是否存在五连 |
| get_winner | `int get_winner() const` | 返回胜者 (1/2)，无胜者返回 0 |
| is_full | `bool is_full() const` | 棋盘是否已满（平局） |
| is_terminal | `bool is_terminal() const` | 是否终局（有人胜或满棋盘） |
| terminal_value | `float terminal_value() const` | 终局价值：胜=1.0, 负=-1.0, 平=0.0（从当前玩家视角）。非终局调用结果未定义 |
| get_board | `const std::vector<int>& get_board() const` | 获取当前棋盘向量 |
| get_legal_actions | `std::vector<int> get_legal_actions() const` | 获取所有空位索引列表 |
| apply_action | `BoardState apply_action(int action) const` | 返回落子后的新 BoardState（不修改自身，玩家翻转） |
| encode_for_network | `std::vector<float> encode_for_network() const` | 输出 3×15×15 相对视角编码张量（与 Python `board_vector_to_cnn_tensor_relative` 一致） |

---

## 2. MctsNode — 搜索树节点

并发安全的 MCTS 树节点。使用 `std::atomic` + 细粒度互斥锁。

```
class MctsNode
```

### 公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `MctsNode(MctsNode* parent, double prior_prob)` | 创建子节点，parent 为父节点指针；根节点传 nullptr |
| select_child | `int select_child(double c_puct, double c_virtual_loss)` | PUCT 选择最优子节点，返回 action 索引。同时施加 virtual loss。无合法子节点时返回 -1 |
| expand | `void expand(const std::vector<float>& action_priors, const std::vector<int>& legal_actions)` | 为当前节点创建子节点。仅叶节点可展开，幂等（重复调用无效） |
| backup | `void backup(float leaf_value)` | 沿父链向上回传，每层取反 value，更新 N/W/Q。自动撤销 virtual loss |
| is_leaf | `bool is_leaf() const` | 是否为叶节点（未展开） |
| get_visit_counts | `std::vector<int> get_visit_counts() const` | 获取所有子节点的 N 值，225 维 |
| get_root_value | `float get_root_value() const` | 获取根节点平均价值 Q |
| get_root_policy | `std::vector<float> get_root_policy() const` | 获取根节点原始策略（噪声施加前的模型输出），225 维 |

---

## 3. NeuralNet — 神经网络推理引擎

使用 LibTorch 加载 TorchScript 模型，提供异步批量推理接口。内部维护一个专用 GPU 推理线程 + 请求队列。

```
class NeuralNet
```

### 公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `NeuralNet(std::string model_path, int batch_size, bool use_gpu)` | 加载 TorchScript 模型文件，启动 GPU 推理线程 |
| 析构 | `~NeuralNet()` | 停止推理线程并等待其退出 |
| predict | `std::future<NeuralNetResult> predict(const std::vector<float>& encoded_board)` | 异步投递推理请求，立即返回 future。调用方通过 `future.get()` 阻塞等待结果 |

### NeuralNetResult 结构体

```
struct NeuralNetResult {
    std::vector<float> policy;  // 225 维 softmax 概率
    float value;                // 局面价值 [-1, 1]
};
```

---

## 4. MctsSearcher — MCTS 搜索器

多线程 MCTS 搜索器主类。封装搜索树管理、Virtual Loss 协调、狄利克雷噪声、子树复用。对应 Python 端 `MCTSSearcher`。

```
class MctsSearcher
```

### 公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `MctsSearcher(std::string model_path, int num_simulations, double c_puct, double dirichlet_epsilon, double dirichlet_alpha, int num_threads, int batch_size, double virtual_loss, bool use_gpu)` | 创建搜索器：加载模型、配置线程池大小 |
| search | `MctsResult search(std::vector<int> state)` | 对给定状态执行 `num_simulations` 次模拟，返回搜索结果 |
| advance_root | `bool advance_root(int action)` | 将 action 对应子节点提升为新根，丢弃兄弟子树。成功返回 true |

### MctsResult 结构体

```
struct MctsResult {
    std::vector<int> visit_counts;    // 225 维访问计数
    float root_value;                 // 根节点平均价值
    std::vector<float> root_policy;   // 根节点原始策略（噪声施加前）
};
```

### 构造参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| model_path | string | TorchScript 模型文件路径 (.pt) |
| num_simulations | int | 每次搜索的总模拟次数，默认 200 |
| c_puct | double | PUCT 探索系数，默认 2.0 |
| dirichlet_epsilon | double | 狄利克雷噪声混合比例，默认 0.25 |
| dirichlet_alpha | double | 狄利克雷分布浓度参数，默认 0.03 |
| num_threads | int | MCTS 工作线程数，默认 4 |
| batch_size | int | GPU 推理攒批大小，默认 16 |
| virtual_loss | double | 虚拟损失值，默认 3.0 |
| use_gpu | bool | 是否使用 GPU 推理，默认 true |
