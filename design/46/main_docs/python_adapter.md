# pybind11 桥接层 + Python 端适配

## 1. pybind11 绑定 (bindings.cpp)

将 C++ 类暴露为 Python 模块 `cpp_mcts`，模块包含一个类 `MctsSearcher`。

### 绑定映射

| C++ 类/结构体 | Python 暴露 | 说明 |
|---------------|-------------|------|
| `MctsSearcher` | `cpp_mcts.MctsSearcher` | 搜索器主类 |
| `MctsResult` | 自动转换为 Python `tuple` | `(visit_counts, root_value, root_policy)` |

### 绑定接口规范

```
模块名: cpp_mcts
    └── class MctsSearcher
         ├── __init__(self, model_path: str, num_simulations: int = 200,
         │            c_puct: float = 2.0, dirichlet_epsilon: float = 0.25,
         │            dirichlet_alpha: float = 0.03, num_threads: int = 4,
         │            batch_size: int = 16, virtual_loss: float = 3.0,
         │            use_gpu: bool = True)
         ├── search(self, state: List[int]) -> Tuple[List[int], float, List[float]]
         │     state: 226 维 int 列表（225 board + 1 player）
         │     return: (visit_counts[225], root_value, root_policy[225])
         └── advance_root(self, action: int) -> bool
```

### 自动类型转换

| C++ 类型 | Python 类型 |
|----------|-------------|
| `std::vector<int>` | `List[int]` |
| `std::vector<float>` | `List[float]` |
| `std::string` | `str` |
| `bool` | `bool` |
| `MctsResult` | `Tuple[List[int], float, List[float]]` |

---

## 2. Python 适配层 (cpp_mcts_adapter.py)

位于 `cpp/python/cpp_mcts_adapter.py`。提供一个工厂函数 `create_cpp_searcher()`，将 C++ MctsSearcher 包装为与现有 `MCTSSearcher` 兼容的接口，供 `PolicyValueAiEngine` 直接使用。

### CppMctsAdapter 公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造 | `__init__(self, model_path: str, num_simulations: int = 200, c_puct: float = 2.0, dirichlet_epsilon: float = 0.25, dirichlet_alpha: float = 0.03, num_threads: int = 4, batch_size: int = 16, virtual_loss: float = 3.0, use_gpu: bool = True)` | 内部创建 `cpp_mcts.MctsSearcher` 实例 |
| search | `(self, state: List[int], log_key: Optional[str] = None) -> Tuple[List[int], float, List[float]]` | 委托 C++ searcher.search()。log_key 参数被忽略（C++ 端暂不支持日志） |
| advance_root | `(self, action: int) -> bool` | 委托 C++ searcher.advance_root() |

### create_cpp_searcher 工厂函数

```
def create_cpp_searcher(model_path: str, **kwargs) -> CppMctsAdapter
```

根据模型路径后缀 `.pt` 判定为 TorchScript 模型，自动选择 C++ 搜索器。

---

## 3. PolicyValueAiEngine 适配方案

无需修改 `PolicyValueAiEngine` 源码。通过依赖注入：在创建 `PolicyValueAiEngine` 时传入 `CppMctsAdapter` 实例替代原 Python `MCTSSearcher`。

### 使用示例

```python
from cpp.python.cpp_mcts_adapter import CppMctsAdapter
from ai.engine.policy_value import PolicyValueAiEngine

# 创建 C++ MCTS 搜索器
searcher = CppMctsAdapter(
    model_path="src/ai/models/cnn_v11/cnn_v11_torchscript.pt",
    num_simulations=200,
    num_threads=64,
    batch_size=32,
    use_gpu=True,
)

# 以与原有完全相同的方式使用
engine = PolicyValueAiEngine(searcher, temperature=1.0, temp_steps=8)
move = engine.get_move(board, current_player, strategy="sampling")
engine.after_move(move.x, move.y)
```

接口完全兼容，`get_move`、`after_move` 等上层方法无需任何改动。

---

## 4. TorchScript 模型导出

在 Python 端，训练完成后导出 TorchScript 格式供 C++ 加载。

### 导出要求

- 使用 `torch.jit.script`（非 `torch.jit.trace`），保留完整控制流
- 输入：`(batch, 3, 15, 15)` 相对视角编码张量
- 输出：`(policy_log_probs, value)` 元组，policy 为 LogSoftmax
- 保存路径：`src/ai/models/cnn_v11/cnn_v11_torchscript.pt`

### CNNV11Model 需新增方法

| 方法 | 签名 | 说明 |
|------|------|------|
| export_torchscript | `(self, output_path: str = None) -> str` | 将 `self.network` 导出为 TorchScript 文件。output_path 为 None 时使用默认路径 |

---

## 5. 接口契约

### search() 行为一致性要求

C++ `MctsSearcher.search()` 的输出必须与 Python `MCTSSearcher.search()` 等价：

| 项目 | 规范 |
|------|------|
| 输入 `state` | `[board[0..224], current_player]`，226 维 int |
| 输出 `visit_counts` | 225 维 int，仅记录本搜索新增的访问计数增量 |
| 输出 `root_value` | float，根节点 Q 值（从当前玩家视角） |
| 输出 `root_policy` | 225 维 float，狄利克雷噪声施加前模型原始 softmax |
| 狄利克雷噪声 | 仅施加于根节点合法动作，混合比例 ε=0.25, α=0.03 |
| PUCT 公式 | \(a = \mathrm{argmax}(Q_a + c_{puct} \cdot P'_a \cdot \frac{\sqrt{\sum N_b}}{1 + N_a})\) |
| Virtual Loss | 选择阶段对每条边 N+=VL, W-=VL；回传阶段撤销 |
| 回传取反 | 每向上一层，value = -value |
| 终局处理 | 胜局 value=-1.0，平局 value=0.0（从落子后方视角）；不展开子节点 |
| 子树复用 | advance_root 成功后，后续 search 保留子树 N 值，仅报告增量 |
