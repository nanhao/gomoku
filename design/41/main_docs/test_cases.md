# 测试用例集

## 一、MctsSelfPlayWorker 单元测试

### TC1.1 构造器 — 正常加载模型

- **前置**: 存在有效的 cnn_v7.pth 权重文件
- **操作**: 创建 `MctsSelfPlayWorker(model_path, model_name, device="cpu", config=默认配置)`
- **验证**: worker 实例创建成功，内部模型已加载到 CPU

### TC1.2 构造器 — 模型文件不存在

- **前置**: 不存在的权重路径
- **操作**: 创建 worker
- **验证**: 抛出 `FileNotFoundError`

### TC1.3 run_games — 单局执行

- **前置**: worker 已创建
- **操作**: `worker.run_games(game_indices=[0], run_id="test")`
- **验证**: 返回非空样本列表，每条样本包含 state（226 维）、pi（225 维浮点）、z（∈ {-1, 0, 1}）；pi 总和 ≈ 1.0

### TC1.4 run_games — 多局执行

- **前置**: worker 已创建
- **操作**: `worker.run_games(game_indices=[0, 1, 2], run_id="test")`
- **验证**: 返回样本总数 = 3 局步数之和；每条样本的 state[225] 记录正确落子方

### TC1.5 run_games — z 值回填正确性

- **操作**: worker 执行 1 局
- **验证**: 若黑胜，所有黑方步的 z=1，白方步的 z=-1；若白胜则相反；若平局全部 z=0

### TC1.6 run_games — engine 不复用

- **操作**: worker 执行 2 局
- **验证**: 两局产生的样本中 game_idx 不同但独立完整，第二局的初始棋盘为空（未被上一局的 MCTS 树污染）

### TC1.7 run_games — 设备确认

- **操作**: worker 以 `device="cpu"` 创建并执行
- **验证**: 模型网络的所有参数张量在 CPU 上

---

## 二、多进程并行测试

### TC2.1 单 worker 与多 worker 结果一致性

- **前置**: 固定随机种子
- **操作**: 
  - 运行 `--games=4 --workers=1`，记录样本总数和每条样本内容
  - 运行 `--games=4 --workers=2`，记录样本总数
- **验证**: 两次运行的样本总数相等；`--workers=2` 时各 worker 分配的 game_indices 无重叠、无遗漏、覆盖 [0, 1, 2, 3]

### TC2.2 分配均匀性

- **操作**: `--games=10 --workers=3`
- **验证**: game_indices 分配为三个 chunk，大小分别为 4、3、3（或类似均匀分布），所有 10 个编号恰好出现一次

### TC2.3 样本按 game_idx 排序

- **操作**: `--games=8 --workers=4`，检查最终输出的 sample.json
- **验证**: 样本按 state 中的 game_idx 升序排列（各 worker 返回的结果被正确合并排序）

### TC2.4 大任务量

- **操作**: `--games=100 --workers=4`
- **验证**: 正常完成无死锁；输出 sample.json 包含 100 局样本；总耗时明显少于单进程

---

## 三、中断与保存测试

### TC3.1 Ctrl+C 部分保存

- **操作**: 启动 `--games=100 --workers=2`，中途发送 SIGINT
- **验证**: 
  - 主进程打印中断信息
  - sample.json 包含已完成局的全部样本
  - summary.json 中 `completed_games` 字段 ≤ 100
  - 文件未损坏（JSON 格式正确）

### TC3.2 周期保存

- **操作**: `--games=60`（_SAVE_INTERVAL=50）
- **验证**: 第 50 局完成时触发一次磁盘写入；第 60 局完成后再次写入

### TC3.3 复用输出目录

- **前置**: 输出目录已有 sample.json（含 30 条样本）
- **操作**: 同目录再跑 `--games=50`
- **验证**: 最终 sample.json 含 30 + 50 局的样本

---

## 四、配置与参数测试

### TC4.1 MctsSelfPlayConfig 默认值

- **操作**: `MctsSelfPlayConfig()`
- **验证**: simulations=800, c_puct=2.0, temp_steps=8, temperature=1.0, batch_size=225, virtual_loss=3.0, use_prior=True

### TC4.2 --workers 默认值

- **操作**: 不传 `--workers`
- **验证**: 使用 `os.cpu_count()` 作为 worker 数

### TC4.3 --workers 大于 --games

- **操作**: `--games=2 --workers=8`
- **验证**: 仅有 2 个 worker 被分配任务，其余 worker 的 game_indices 为空列表（不报错，不执行对局）

### TC4.4 超参数透传

- **操作**: `--simulations=400 --c-puct=1.5`
- **验证**: worker 内部创建的 engine 使用 simulations=400、c_puct=1.5

---

## 五、回归测试

### TC5.1 样本格式兼容性

- **操作**: 多进程方案输出的 sample.json
- **验证**: 可被 `mcts_train.py` 正常加载并开始训练（格式与当前单进程版本完全一致）

### TC5.2 MCTSSearcher 不受影响

- **操作**: 运行现有 MCTS 搜索器单元测试
- **验证**: 全部通过
