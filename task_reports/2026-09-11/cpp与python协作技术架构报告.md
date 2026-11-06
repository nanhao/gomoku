# C++ 与 Python 协作技术架构报告

**日期**: 2026-09-11
**调研范围**: `cpp/`（C++ MCTS 核心 + Python 适配层）、`src/ai/`（训练闭环与脚本）、`build_cpp.bat`、相关设计文档（design/49、design/59）

---

## 1. 架构总览

本项目是一个 AlphaGo Zero 风格的五子棋自我对弈训练系统，采用 **"Python 管训练与 I/O，C++ 管搜索与对局，GPU 推理全部下沉到 C++ 端"** 的分工。C++ 与 Python 的数据桥梁只有一个：**TorchScript 模型文件（.pt）**，权重单向流动（Python 训练 → 导出 → C++ 推理），运行时无双向权重同步。

```
┌─────────────────────────── Python 应用层 (src/) ───────────────────────────┐
│  GUI/引擎        AIEngineBuilder.create_engine()                            │
│                  └─ PolicyValueAiEngine(CppMctsAdapter)  ← 智能路由          │
│  CLI 对战        mcts_battle_cpp.py      → CppMctsBattleAdapter             │
│  自对弈采样      mcts_sample_buffer.py   → CppMctsSampleBufferAdapter        │
│  训练闭环        mcts_train_cnn_v11.py   → checkpoint_evaluator（直接用      │
│                    cpp_mcts.GameRunner）+ off_policy_train_loop.py          │
├────────────────────────── Python 适配层 (cpp/python/) ──────────────────────┤
│  cpp_mcts_adapter.py / cpp_mcts_battle_adapter.py /                        │
│  cpp_sample_buffer_adapter.py / mcts_config.py（与 C++ 共享同一 JSON）      │
├──────────────────────── pybind11 绑定层 (cpp/src/bindings.cpp) ─────────────┤
│                    cpp_mcts.pyd（模块导入时读 mcts_config.json）            │
├──────────────────────────── C++ 核心层 (mcts_core 静态库) ──────────────────┤
│  MctsSearcher（多线程搜索+Virtual Loss+子树复用）                            │
│  GameRunner（N 线程并行对局+样本采集+D4 增强）                               │
│  NeuralNet 全局单例（LibTorch 推理线程 + 请求队列攒批）                       │
│  MctsThreadPool 全局单例（batch/queue 双模式）                               │
│  BoardState（规则+相对视角编码）  MctsNode（atomic 并发树）                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 三种协作模式

| 模式 | 调用方 | C++ 对象 | 交互方式 |
|------|--------|----------|----------|
| **A. 单次搜索** | GUI / `PolicyValueAiEngine` | `MctsSearcher` | Python 逐次调 `search(state226)` → 同步拿结果；落子后调 `advance_root(action)` 复用子树 |
| **B. 批量对战** | `mcts_battle_cpp.py` / `checkpoint_evaluator` | `GameRunner`（双模型构造） | Python 创建后 `start()` 非阻塞启动，轮询 `get_progress()`，结束后 `get_results()` |
| **C. 自对弈采样** | `mcts_sample_buffer.py` | `GameRunner`（自对弈构造，`collect_samples=true`） | 模式 B 基础上，Python 侧按 chunk（10 局）轮询拉取 `GameSample` 并落盘到 `sample_buffer/` |

### 1.2 模块边界一览

| 目录 | 职责 |
|------|------|
| `cpp/src/*.cpp/.h` | C++ 核心：搜索、对局、规则、推理、线程池 |
| `cpp/src/bindings.cpp` | pybind11 绑定（唯一入口），约 280 行 |
| `cpp/python/*.py` | Python 适配层（Adapter 模式），屏蔽 C++ 类型与路径解析 |
| `cpp/python/cpp_mcts.pyd` | 构建产物，由 `build_cpp.bat` 从 `cpp/build/` 拷入 |
| `cpp/python/mcts_config.json` | **双端共享**的运行时配置（Python 与 C++ 各读一份，参数保持一致） |

---

## 2. 构建系统关键技术

### 2.1 CMake 配置要点（cpp/CMakeLists.txt）

- **LibTorch 来源**：不使用 `find_package(Torch)`（那需要完整 CUDA Toolkit），而是用 `execute_process` 执行 `python -c "import torch; print(os.path.dirname(torch.__file__))"` 自动探测 pip 安装的 PyTorch 路径，直接拼 include 和 lib 路径。这是"pip PyTorch 即 LibTorch"的关键技巧。
- **链接库**：`torch.lib / torch_cpu.lib / torch_cuda.lib / c10.lib / c10_cuda.lib`。链接 torch_cuda.lib 使 CUDA 符号编译期解析，运行时靠 PATH 找 DLL。
- **MSVC 强制 `/utf-8`**：否则中文注释/字符串字面量触发 C2001/C4819 编译错误。
- **pybind11 是可选依赖**：`find_package(pybind11 QUIET)`，缺失时仅构建 `mcts_core` 静态库和 C++ 测试，跳过 .pyd。
- 产物命名：`PREFIX ""` + `SUFFIX ".pyd"`，生成 `cpp_mcts.pyd`（Python 可直接 import 的扩展模块）。
- **单元/集成测试**（ctest）通过 `ENVIRONMENT_MODIFICATION` 把 `${TORCH_LIB_DIR}` 前置到 PATH，保证测试可执行文件能找到 DLL。

### 2.2 build_cpp.bat 一键构建

自动探测 Python / cmake / ctest（含 pip 安装位置）→ CMake configure/build → **两步 DLL 拷贝**：

1. `cpp_mcts.pyd` → `cpp/python/`
2. torch CPU 依赖 DLL（`torch.dll torch_cpu.dll c10.dll libiomp5md.dll torch_global_deps.dll shm.dll uv.dll zlibwapi.dll`）→ `cpp/python/`（pyd 导入时即需要）和 build 输出目录（供测试程序用）

### 2.3 绑定层细节（bindings.cpp）

- **配置加载借用 Python**：模块初始化时通过 `py::module_::import("os"/"json")` 定位 .pyd 同目录的 `mcts_config.json`，填充全局 `g_mcts_config`；文件缺失时静默回退内置默认值。**必须在绑定定义之前加载**，因为 `MctsSearcher` 构造参数的 pybind 默认值直接引用了 `g_mcts_config` 字段。
- `SetConsoleOutputCP(CP_UTF8)`：Windows 控制台默认 GBK(936)，C++ 端 UTF-8 输出会乱码。
- 结构体绑定模式：数据类用 `def_readonly`（GameSample/GameResult/GpuInferenceStats）或 `def_readwrite`（GameConfig）直接暴露字段；`MctsSearcher.search()` 返回值手工转换为 `py::tuple (visit_counts[225], root_value, root_policy[225])`。
- 暴露全局函数：`init_thread_pool / shutdown_thread_pool / thread_pool_size / get_gpu_stats`。

---

## 3. C++ 端核心组件设计

### 3.1 NeuralNet 全局单例（LibTorch 推理）

- **专用推理线程 + 请求队列攒批**：MCTS worker 调 `predict(675维编码, model_id)` → 请求入队 → 阻塞等待 `std::promise` 结果；推理线程收集请求后按 `model_id` 分组批量 forward。
- 攒批参数（来自 mcts_config.json）：`batch_lower_bound=16`（不足时再等，首请求最多等 500μs、攒批窗口 2ms、100μs 轮询），`batch_upper_bound=500`。
- **模型缓存**：`unordered_map<路径, Module*>`，支持双模型对战；`preload_model()` 用 `shared_mutex` 独占写锁（会阻塞推理线程下一 batch），forward 持 shared 锁。
- **GPU 初始化策略**：加载时强制 `map_location=kCPU`；`module->to(kCUDA)` 延迟到推理线程首次 forward 时执行——避免主线程 CUDA context 与 worker 线程冲突。
- 输出处理：policy 为 LogSoftmax 输出 → exp → 再归一化（防数值精度问题）；value 为 Tanh 输出。
- 运行时可观测：`GpuInferenceStats`（全周期累计 + 最近 10 batch / 5 秒滑动窗口），通过 `cpp_mcts.get_gpu_stats()` 在 Python 端读取，`wait_ratio()` 是调优核心指标。

### 3.2 MctsThreadPool 全局单例

进程内唯一，`initialize()` 用 call_once 幂等。双模式：
- **batch 模式**（`run_parallel`）：所有 worker 执行同一任务函数（单搜索器 800 次模拟按 atomic 计数器分发）。
- **queue 模式**（`submit + wait_all`）：多对局并发时多个搜索器共享线程池。

线程池复用的动机：避免每局 30+ 步 × 32 线程 ≈ 每局上千次线程创建销毁。

### 3.3 MctsSearcher（搜索器）

- 输入 226 维 int（225 棋盘 + 1 当前玩家），内部由 `BoardState::encode_for_network()` 转成 3×15×15=675 维**相对视角**编码（通道 0=我方棋子、1=对方棋子、2=我方颜色指示器），与 Python 端 `board_vector_to_cnn_tensor_relative` 语义严格一致。
- 子树复用三分支（design/59 修复点 1）：局面一致 → 复用；恰差 1 子且目标分支已展开 → 内部推进复用；其余 → 整树销毁重建（`rebuild_count_` 计数，GameRunner 每局打印）。
- 树规模限幅 `max_tree_expands=20000`（约 24MB/树），防止多 worker 长对局内存累积到 GB 级。

### 3.4 GameRunner（并行对局）

- N 个游戏线程轮询分发对局（worker w 负责 w, w+K, w+2K…），共享全局线程池和 NeuralNet。
- 自对弈模式交替执方（偶数局 model1 执黑）；双模型对战可 `alternate_colors` 或随机。
- 样本采集：每步记录 (state, visit_counts)，终局回填 z，C++ 端完成 D4 增强（8 种对称变换，`SampleAugmentor`），产出 AGZ 格式 `GameSample{state[226], pi[225], z}`。

---

## 4. Python 适配层设计

| 适配器 | 包装的 C++ 对象 | 关键职责 |
|--------|----------------|----------|
| `CppMctsAdapter` | `MctsSearcher` | GUI/引擎单搜索模式；`init_thread_pool` 幂等初始化；`search/advance_root` 透传 |
| `CppMctsBattleAdapter` | `GameRunner`（双模型） | registry spec（如 `cnn_v11.1`）→ TorchScript 路径解析；结果转换时在 Python 侧计算 `model1_won/model2_won` |
| `CppMctsSampleBufferAdapter` | `GameRunner`（自对弈） | 局号翻译（C++ 局部 game_index 从 0 起 → 全局局号）；每 10 局 chunk 落盘；`GameSample` → dict |

**智能路由**（`AIEngineBuilder.create_engine`）：模型为 cnn_v11 且同目录存在 `_torchscript.pt` 且 `cpp_mcts` 可导入 → 创建 C++ 引擎；任一条件失败 → 静默回退 Python MCTS 实现。两条实现的 `search/advance_root` 接口签名完全一致，上层无感知。

**模型导出**：训练闭环（`off_policy_train_loop.py`、`cnn_v11_train_loop.py`）用 `torch.jit.script(network)` + `torch.jit.save` 导出；checkpoint 评估阶段也导出临时 `best.pt`/`checkpoint.pt` 给 C++ 对战用（checkpoint 不在 registry 中，故 `checkpoint_evaluator.py` 绕过适配器直接用 `cpp_mcts.GameRunner`）。

---

## 5. 必须注意的坑（实战经验）

### 5.1 构建与加载类

| # | 坑 | 说明 / 对策 |
|---|-----|------------|
| 1 | **MSVC 中文编码** | 无 `/utf-8` 时中文注释/字符串触发 C2001/C4819。CMakeLists 已加；新增 C++ 文件无需额外处理，但不要用本地编码保存源文件 |
| 2 | **pyd 的 DLL 依赖链** | `cpp_mcts.pyd` 隐式链接 torch DLL，**import 时即需要** `torch.dll/torch_cpu.dll/c10.dll/libiomp5md.dll` 等。`build_cpp.bat` 已拷贝到 `cpp/python/`；若手工换 Python 环境（torch 版本变化）必须重新跑构建脚本拷 DLL |
| 3 | **CUDA DLL 不在拷贝列表** | cudnn/cublas 等只在 `torch\lib` 中。**必须在 `import cpp_mcts` 之前把 `torch\lib` 加入 PATH**，否则 import 或首次 GPU 推理时报 DLL 找不到。项目内所有入口脚本都遵循 `import torch → 改 PATH → import cpp_mcts` 顺序 |
| 4 | **sys.path 三件套** | `cpp_mcts.pyd` 位于平铺目录 `cpp/python/`（不是标准包结构），脚本需手动把 项目根、src、cpp\python 三条路径插入 sys.path（见各 `src/ai/scripts/*.py` 头部） |
| 5 | **CMake 找 Torch** | 不要用 `find_package(Torch)`（要求 CUDA Toolkit）；依赖 build 脚本先验证 `python -c "import torch"`，或手动 `-DTORCH_ROOT=...` |
| 6 | **pybind11 缺失静默跳过** | `find_package(QUIET)` 下 pybind11 缺失时构建"成功"但**没有产出 .pyd**。`build_cpp.bat` 用 `python -c "import pybind11; print(pybind11.get_cmake_dir())"` 探测并显式传 `-Dpybind11_DIR`；构建后注意确认 `[ OK ] cpp_mcts.pyd ready` |
| 7 | **控制台中文乱码** | 绑定层已 `SetConsoleOutputCP(CP_UTF8)`；若在 C++ 端新增 printf/std::cout 中文输出，无需额外处理，但 PowerShell 端观察时注意代码页 |

### 5.2 运行时与线程类

| # | 坑 | 说明 / 对策 |
|---|-----|------------|
| 8 | **CUDA context 跨线程** | 模型加载固定 `map_location=kCPU` 在主线程完成；`to(kCUDA)` 只能在推理线程内做。绝不能在 pybind 调用线程（Python 主线程）初始化 CUDA 后又在别的线程用同一模型 |
| 9 | **GPU tensor 直写 UB** | 必须先在 CPU 上创建张量并 `data_ptr` 填充，再 `.to(device)`；直接往 GPU tensor 的 `data_ptr()` 写数据是未定义行为，会 segfault（`neural_net.cpp` forward_batch 有明确注释） |
| 10 | **全局单例生命周期** | `NeuralNet::shutdown()` 与 `instance()` 并发会 use-after-free，需持锁；`MctsThreadPool::initialize` 是 call_once 幂等，Python 侧多个 Adapter 各自调用 `init_thread_pool` 是安全的，但线程数以**首次调用**为准（后传不同值不会生效，打印的 actual_pool_size 可核实） |
| 11 | **predict 阻塞语义** | worker 提交请求后阻塞等 promise；若推理线程意外退出（如模型加载抛异常未被捕获），所有 worker 挂死。攒批参数配置过大时延迟放大，需结合 `get_gpu_stats()` 的 wait_ratio 判断瓶颈在 GPU（<0.3）还是 MCTS 生产端（>0.6） |

### 5.3 语义对齐类（design/59，血泪教训）

| # | 坑 | 说明 / 对策 |
|---|-----|------------|
| 12 | **子树复用必须校验树根局面** | C++ 端复用分支若只判 `root_ != nullptr`，交替对弈中树根会滞后对手 1 手，搜索在错位树上进行，棋力崩塌（曾造成 CLI 对战"五五开"假象）。现以 `root_board_/root_player_` 三分支校验修复，对齐 Python `root_hash` 语义 |
| 13 | **promote 目标必须是已展开节点** | 把未访问叶节点提升为根后无子边，800 次模拟全部空转 → visit_counts 全 0 → 弃局。目标为叶时必须走整树重建分支 |
| 14 | **落子必须过滤真实空位** | `sample_action` 候选条件是 `visit_counts>0 && board[i]==0`；只看 visit_counts 会把对手刚落的子位重复落子（覆盖对方棋子），产生非法棋局。另有兜底链：候选空 → 全空位按 N 最大；满盘 → -1 |
| 15 | **节点提升为根需 flip_value** | 非根节点 q 存父视角、根节点存自身视角，推进后不翻转会导致 root_value 向 0 漂移。每次 promote 后立即 `flip_value()` |
| 16 | **展开与策略存储必须同一临界区** | 两个 worker 可同时判定同一 child 为叶并双双进入展开流程；锁外写 225 维策略 vector 是数据竞争。必须在 `expand_lock_` 内原子完成（`expand_with_policy`） |
| 17 | **树内存限幅** | GameRunner 跨步保留搜索树，展开节点单调累积，225 步对局单树可达数万节点；多 worker × 2 树可达 GB 级。`max_tree_expands=20000` 触发重建，仅损复用不损正确性 |
| 18 | **污染数据识别** | design/59 修复前 C++ 自对弈采集的样本（错位树 pi、覆盖落子）不可信；renjunet_samples（RIF 棋谱转换）不经 GameRunner，不受影响。排查训练质量问题时要先确认样本产自修复后的版本 |
| 19 | **重建次数的解读** | 强弱悬殊对局中弱方重建次数接近步数属正常（其树浅层 N>0 边极少），不要误判为回归；同强度对局重建次数应显著小于步数 |

### 5.4 接口约定类

| # | 坑 | 说明 / 对策 |
|---|-----|------------|
| 20 | **Adapter 参数静默忽略** | `CppMctsAdapter.search(state, num_simulations, log_key)` 的后两个参数被忽略——模拟次数在构造时固定，C++ 端不支持搜索日志。改参数需重建 Adapter 或改 mcts_config.json |
| 21 | **局号坐标系** | C++ `GameRunner` 的 `game_index` 每批次从 0 开始（局部坐标），Python 缓冲区用全局局号，靠 `_global_base_game_index` 翻译（全局局号 = 基准 + 局部 index + 1）。跨批次续接时从 manifest 恢复基准，防止 chunk 文件名冲突 |
| 22 | **先验证再导入** | Adapter 构造时先解析/校验模型路径再 `import cpp_mcts`，确保 `FileNotFoundError` 能正确抛出而非被 ImportError 掩盖 |
| 23 | **双端配置一致性** | `mcts_config.json` 是唯一配置源，Python（`mcts_config.py`）与 C++（`bindings.cpp` 的 `load_mcts_config`）各自解析同一文件；新增配置键必须**三处同步**（JSON、mcts_config.py、bindings.cpp 解析行），漏一处会静默回退默认值 |
| 24 | **registry 多代模型约束** | spec 如 `cnn_v11.2` 必须指向存在的 `gen2/` 目录和 `<name>_torchscript.pt`；非多代模型不能带代数。TorchScript 文件缺失时在 Python 侧提前报错，不会走到 C++ |

---

## 6. 测试体系

- **C++ 侧**（ctest，`build_cpp.bat --test`）：单元测试覆盖 board_state / mcts_node / mcts_searcher（含资源注入、状态同步）/ neural_net（单例、双模型）/ thread_pool / game_runner（workers、colors）/ sample_augmentor；集成测试 `test_mcts_integration`、`test_game_runner_legal_moves`（对齐 design/59 修复点 2 的合法性验证）。
- **Python 侧**：`tests/` 下的 pytest 套件覆盖适配层与训练组件；C++ 相关冒烟通过 CLI 脚本（`mcts_battle_cpp.py --games 2`）人工验证。
- 测试可执行文件的 DLL 路径由 ctest `ENVIRONMENT_MODIFICATION` 解决，不需要手工设 PATH。

---

## 7. 架构评价与注意事项小结

**优点**：
- 边界干净：跨语言只传 JSON 配置 + TorchScript 模型 + 扁平数值数组，无复杂对象图穿越边界；
- 适配器模式让 Python MCTS / C++ MCTS 可互换，GUI 路径有自动回退；
- 全链路可观测（GpuInferenceStats、rebuild_count、样本统计），调优有数据支撑。

**结构性约束（改动前必读）**：
1. 修改 C++ 接口 → 必须同步 `bindings.cpp` + Python 适配层 + CMakeLists 测试，并跑 `build_cpp.bat --test`；
2. 新增配置 → JSON、`mcts_config.py`、`bindings.cpp` 三处同步；
3. 更换 torch 版本 → 重新 `build_cpp.bat`（重链 + 重拷 DLL）；
4. 涉及搜索正确性的改动 → 对照 design/59 的六条修复点与测试用例回归；
5. C++ 端任何新增输出路径（如模型缓存、日志）都要考虑 Python 侧工作目录差异——C++ 单例的默认模型路径是相对路径 `src/ai/models/cnn_v11/...`，依赖进程 CWD 为项目根。
