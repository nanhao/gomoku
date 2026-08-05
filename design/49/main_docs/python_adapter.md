# Python 适配层与 CLI 脚本设计

## 1. SampleBufferManager — 样本缓冲区管理器

管理 `sample_buffer/` 目录下的 FIFO 滑动窗口缓冲区。以 10 局 chunk 为原子单位进出，总样本数上限 80000。

### 对外接口

```python
class SampleBufferManager:
    def __init__(self, buffer_dir: str, max_size: int = 80000, chunk_games: int = 10):
        """初始化缓冲区管理器

        Args:
            buffer_dir: 缓冲区目录路径（如 "sample_buffer"）
            max_size: 缓冲区最大样本数（按局面计），默认 80000
            chunk_games: 每个 chunk 包含的对局数，默认 10
        """

    def ingest(self, game_start: int, game_end: int, samples: List[dict]) -> int:
        """写入一个 chunk 到缓冲区，必要时淘汰旧 chunk

        将 game_start 到 game_end（含）的对局产生的样本写入新 chunk 文件。
        若写入后缓冲区总样本数 >= max_size，从最旧 chunk 开始逐 chunk 淘汰，
        直到总样本数 < max_size。

        Args:
            game_start: 本 chunk 起始局编号（1-based）
            game_end: 本 chunk 结束局编号（含）
            samples: 样本列表，每项 {"state": [...], "pi": [...], "z": float}

        Returns:
            本次淘汰的样本总数（0 表示未触发淘汰）
        """

    def get_total_samples(self) -> int:
        """返回缓冲区当前样本总数"""

    def get_chunk_count(self) -> int:
        """返回当前 chunk 数量"""

    def get_summary(self) -> dict:
        """返回缓冲区摘要信息
        Returns:
            {"total_samples": N, "chunk_count": N, "max_size": N,
             "chunks": [{"game_range": [start, end], "sample_count": N, "file": path}, ...]}
        """
```

### 行为约定

- 构造时自动创建 `buffer_dir`（若不存在）
- 若 `buffer_dir/manifest.json` 存在，加载已有状态（支持断点续跑）
- 每 chunk 写入独立 JSON 文件，命名格式：`chunk_{start:06d}_{end:06d}.json`
- 文件内容：样本列表的 JSON 数组，每项 `{"state": [...], "pi": [...], "z": float}`
- `manifest.json` 记录缓冲区元数据：chunk 列表、总样本数、配置参数
- 淘汰操作：删除对应 chunk 文件并从 manifest 中移除记录
- 写文件采用"先写 `.tmp` 再 `os.replace`"的原子写入策略

### 文件结构

```
sample_buffer/
├── manifest.json                 # 元数据
├── chunk_000001_000010.json     # 第 1-10 局样本
├── chunk_000011_000020.json     # 第 11-20 局样本
├── chunk_000021_000030.json     # 第 21-30 局样本
└── ...
```

### manifest.json 格式

```json
{
    "max_size": 80000,
    "chunk_games": 10,
    "total_samples": 15234,
    "chunks": [
        {"game_start": 1, "game_end": 10, "sample_count": 5123, "file": "chunk_000001_000010.json"},
        {"game_start": 11, "game_end": 20, "sample_count": 4987, "file": "chunk_000011_000020.json"},
        {"game_start": 21, "game_end": 30, "sample_count": 5124, "file": "chunk_000021_000030.json"}
    ]
}
```

---

## 2. CppMctsSampleBufferAdapter — C++ 自对弈适配器

包装 C++ `GameRunner` 为 Python 友好接口，负责模型路径解析、进度轮询和样本缓冲管理。

### 对外接口

```python
class CppMctsSampleBufferAdapter:
    def __init__(
        self,
        model_spec: str,
        num_games: int,
        config: MctsSelfPlayConfig,
        num_workers: int = 0,
        buffer_dir: str = "sample_buffer",
        buffer_max_size: int = 80000,
        chunk_games: int = 10,
    ):
        """初始化适配器

        构造时完成：
        - 从 registry spec 解析 TorchScript 模型路径
        - 创建 C++ GameRunner（自对弈模式，collect_samples=true）
        - 创建 SampleBufferManager

        Args:
            model_spec: registry spec，如 "cnn_v11.1"
            num_games: 自对弈总局数（≥1）
            config: MCTS 超参数配置
            num_workers: 游戏线程数（0=每局一线程）
            buffer_dir: 样本缓冲区目录
            buffer_max_size: 缓冲区最大样本数
            chunk_games: 每 chunk 包含的对局数

        Raises:
            FileNotFoundError: TorchScript 模型文件不存在
            ValueError: registry spec 无效
        """

    def start(self) -> None:
        """启动所有 C++ 游戏线程（非阻塞）"""

    def get_progress(self) -> Tuple[int, int]:
        """返回 (已完成局数, 总局数)"""

    def get_sample_stats(self) -> dict:
        """返回实时样本统计
        Returns:
            {"completed_games": N, "total_games": N,
             "buffer_samples": N, "buffer_chunks": N,
             "evicted_samples": N}
        """

    def is_finished(self) -> bool:
        """所有对局是否已完成并全部落盘"""

    def wait_completion(self) -> None:
        """阻塞直到所有对局完成、所有样本落盘"""

    def get_buffer_summary(self) -> dict:
        """返回缓冲区最终摘要（同 SampleBufferManager.get_summary()）"""
```

### 行为约定

- 构造时复用 `_resolve_torchscript_path`（与 `CppMctsBattleAdapter` 共用逻辑）
- 自对弈模式：利用 C++ GameRunner 的新增构造器（单模型路径 + collect_samples=true）
- `start()` 后内部启动进度轮询线程，每检测到 `chunk_games` 局新完成即触发 `buffer_manager.ingest()`
- 进度轮询间隔 ≤ 0.1 秒
- 每完成一个 chunk 的落盘后，实时打印缓冲区状态
- Ctrl+C 时优雅退出：等待进行中对局完成，将已完局样本全部落盘
- `model_spec` 路径解析使用 `ai.registry` 模块（同 `CppMctsBattleAdapter`），不依赖 `.pth` 文件

---

## 3. mcts_sample_buffer.py — CLI 脚本

### 参数

| 参数 | 类型 | 必需 | 默认 | 说明 |
|------|------|------|------|------|
| `--model` | str | ✓ | — | 模型 registry spec（如 `cnn_v11.1`） |
| `--games` | int | ✓ | — | 自对弈总局数（≥1） |
| `--workers` | int | | 0 | 游戏工作线程数（0=每局一线程） |
| `--simulations` | int | | 800 | 每步 MCTS 模拟次数 |
| `--c-puct` | float | | 2.0 | PUCT 探索系数 |
| `--temp-steps` | int | | 8 | 温度探索步数 |
| `--temperature` | float | | 1.0 | 温度参数 |
| `--dirichlet-epsilon` | float | | 0.25 | 狄利克雷噪声比例 |
| `--dirichlet-alpha` | float | | 0.03 | 狄利克雷分布参数 |
| `--virtual-loss` | float | | 配置值 | virtual loss 值 |
| `--buffer-max-size` | int | | 80000 | 缓冲区最大样本数 |
| `--chunk-games` | int | | 10 | 每 chunk 对局数 |
| `--output-dir` | str | | `sample_buffer` | 缓冲区目录 |

### 实时打印格式

```
[超参数] simulations=800, c_puct=2.0, temperature=1.0, temp_steps=8, dirichlet_epsilon=0.25, virtual_loss=3.0
[模型] cnn_v11.1
[模式] 自对弈（双方交替执方），总局数=1000

[进度] 对局: 15/1000 | 缓冲区: 6234 样本, 1 chunk | 淘汰: 0
[进度] 对局: 25/1000 | 缓冲区: 10421 样本, 2 chunk | 淘汰: 0
...
[进度] 对局: 215/1000 | 缓冲区: 80234 样本, 21 chunk | 淘汰: 5123
[进度] 对局: 225/1000 | 缓冲区: 79812 样本, 21 chunk | 淘汰: 10234

=== 对弈进度统计 ===
  已完成: 1000/1000
  黑胜: 487 (48.7%)
  白胜: 456 (45.6%)
  平局: 57 (5.7%)
  总样本: 401234 (已淘汰: 321567)
  当前缓冲区: 78667 样本, 9 chunk

[耗时] 总耗时 2345.6s
[GPU 监控] 总批次: ... (同 mcts_battle_cpp.py 格式)

[完成] 缓冲区状态: 78667 样本, 9 chunk, 目录: sample_buffer/
```

### 行为

- 入口函数 `main()` 解析参数 → 构建 `MctsSelfPlayConfig` → 创建 `CppMctsSampleBufferAdapter` → `start()` → 轮询打印 → `wait_completion()` → 打印最终统计
- Ctrl+C 终止：`_signal_handler` 标记中断 → 等待进行中对局完成 → 将已完局未落盘的样本写入最后一个 chunk → 打印中断摘要 → 退出
- 若 `--games` 不是 `--chunk-games` 的整数倍，最后一个 chunk 的对局数不足 10 局，照常落盘
