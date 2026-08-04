# Python 适配层与 CLI 脚本

## 1. CppMctsBattleAdapter — C++ GameRunner 包装器

### 文件

`cpp/python/cpp_mcts_battle_adapter.py`

### 职责

将 C++ `GameRunner` 包装为 Python 友好的接口，负责：
- 从 registry spec 解析模型路径并查找 TorchScript 文件
- 创建 C++ GameRunner 实例
- 提供进度查询和结果获取

### 对外接口

```python
class CppMctsBattleAdapter:
    def __init__(
        self,
        model1_spec: str,        # registry spec，如 "cnn_v11.1"
        model2_spec: str,        # registry spec，如 "cnn_v11.2"
        num_games: int,          # 对战局数
        config: MctsSelfPlayConfig,  # 复用现有配置类
    ):
        """初始化适配器。构造时完成：
        - 通过 registry 解析两个模型 spec → 查找 TorchScript 模型路径
        - 初始化全局 MCTS 线程池
        - 构建 GameConfig 并创建 C++ GameRunner
        """

    def start(self) -> None:
        """启动所有游戏线程，非阻塞，立即返回"""

    def get_progress(self) -> tuple[int, int]:
        """返回 (已完成局数, 总局数)"""

    def is_finished(self) -> bool:
        """所有对局是否已完成"""

    def wait_completion(self) -> None:
        """阻塞直到所有对局完成"""

    def get_results(self) -> list[CppBattleGameResult]:
        """获取所有对局结果，仅在完成时调用"""
```

### 内部逻辑：模型路径解析

```
输入 spec "cnn_v11.1"
  → registry.get_model("cnn_v11.1") 获取模型实例
  → 从 model.DEFAULT_MODEL_PATH 获取 .pth 路径
  → 同目录下查找 cnn_v11_torchscript.pt
  → 返回 .pt 绝对路径
```

若 TorchScript 文件不存在，构造器抛出 `FileNotFoundError`。

### 数据类

```python
@dataclass
class CppBattleGameResult:
    """单局对战结果"""
    game_index: int
    winner: int         # 0=平局, 1=黑胜, 2=白胜
    model1_color: int   # model1 执方 1=黑, 2=白
    model2_color: int   # model2 执方
    model1_won: bool    # adapter 层计算
    model2_won: bool    # adapter 层计算
    move_count: int
```

---

## 2. mcts_battle_cpp.py — CLI 入口脚本

### 文件

`src/ai/scripts/mcts_battle_cpp.py`

### CLI 参数

```
--ai1           必需  模型1 registry spec（如 cnn_v11.1）
--ai2           必需  模型2 registry spec（如 cnn_v11.2）
--games         必需  对战局数（≥1）
--simulations   可选  每步 MCTS 模拟次数（默认 800）
--c-puct        可选  PUCT 探索系数（默认 2.0）
--temp-steps    可选  温度探索步数（默认 8）
--temperature   可选  温度参数（默认 1.0）
--dirichlet-epsilon  可选  狄利克雷噪声比例（默认 0.25）
--dirichlet-alpha    可选  狄利克雷分布参数（默认 0.03）
--batch-size    可选  GPU 推理攒批大小（默认读 mcts_config.json）
--virtual-loss  可选  virtual loss 值（默认读 mcts_config.json）
```

不支持 `--workers` 参数。并发度由游戏数 N 决定（N 个 C++ 线程）。

### 用法示例

```bash
# 基本对战
python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 50

# 自定义 MCTS 参数
python src/ai/scripts/mcts_battle_cpp.py \
    --ai1 cnn_v11.3 --ai2 cnn_v11.4 \
    --games 100 --simulations 400 --temperature 0.5 --temp-steps 12

# 同模型自对弈评估（测试随机性）
python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.1 --games 20
```

### 脚本结构

遵循入口脚本可定义函数的例外规则：

- `parse_args()` — 解析 CLI 参数
- `_build_config(args)` — 构建 MctsSelfPlayConfig
- `run_battle(args)` — 主流程入口
- `_print_per_game_results(results, model1, model2)` — 打印每局结果
- `_print_statistics(model1, model2, results)` — 打印胜率统计
- `if __name__ == "__main__"` — 入口

### 主流程

1. 解析参数，验证 `--games ≥ 1`
2. 构建 `MctsSelfPlayConfig`
3. 创建 `CppMctsBattleAdapter`
4. 调用 `adapter.start()` 启动所有 C++ 游戏线程
5. 轮询 `adapter.get_progress()` 实时打印进度
6. 支持 Ctrl+C 中断：收到 SIGINT 后调用 `adapter.wait_completion()` 等待进行中的对局完成
7. 调用 `adapter.get_results()` 获取结果
8. 打印每局结果和最终统计

### 输出格式

与现有 `mcts_battle.py` 一致：

```
[进度] 3/10

[局 1/10] cnn_v11.1(黑) vs cnn_v11.2(白) → 黑方胜 (45步)
[局 2/10] cnn_v11.1(白) vs cnn_v11.2(黑) → 白方胜 (52步)
[局 3/10] cnn_v11.1(黑) vs cnn_v11.2(白) → 平局 (225步)
...

=== 对战结果 ===
总局数: 10
cnn_v11.1 胜: 5 局 (50.0%)
cnn_v11.2 胜: 4 局 (40.0%)
平局: 1 局 (10.0%)
```

---

## 3. bindings.cpp — C++/Python 桥接扩展

### 新增绑定

```cpp
// GameConfig
py::class_<GameConfig>(m, "GameConfig")
    .def(py::init<>())
    .def_readwrite("num_simulations", &GameConfig::num_simulations)
    // ... 其他字段 ...

// GameResult
py::class_<GameResult>(m, "GameResult")
    .def_readonly("game_index", &GameResult::game_index)
    // ... 其他字段 ...

// GameRunner
py::class_<GameRunner>(m, "GameRunner")
    .def(py::init<std::string, std::string, int, GameConfig>())
    .def("start", &GameRunner::start)
    .def("get_completed_count", &GameRunner::get_completed_count)
    .def("get_total_count", &GameRunner::get_total_count)
    .def("is_finished", &GameRunner::is_finished)
    .def("wait_completion", &GameRunner::wait_completion)
    .def("get_results", &GameRunner::get_results);
```
