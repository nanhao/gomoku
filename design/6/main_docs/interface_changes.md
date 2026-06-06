# 接口变更详细设计

本次为纯新增，无已有接口变更。

---

## 1. 新增模块：ai/iterative_train.py

### 1.1 模块级接口

| 接口 | 签名 | 说明 |
|------|------|------|
| `main()` | `() -> None` | CLI 入口函数，创建 IterativeTrainer 实例并调用 `run()`，打印最终报告。`main()` 捕获 `FileNotFoundError`、`RuntimeError`、`ValueError`、`IOError` 等异常，打印友好错误消息后以非零退出码退出。`KeyboardInterrupt` 由 `run()` 内部处理，`main()` 不重复捕获 |

### 1.2 类：IterativeTrainer

#### 构造器

```
IterativeTrainer(project_root: str = None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project_root` | str | None | 项目根目录路径，None 时自动推导（取 `Path(__file__).resolve().parent.parent`）。注意：依赖 `__file__` 推导，在 `python -m` 调用或符号链接场景下可能需显式传入 |

**前置校验**：构造时检查 `gen4/cnn_v3.pth` 是否存在，不存在则抛 `FileNotFoundError`。同时检查锁文件 `ai/models/cnn_v3/.iterative_lock` 是否存在，存在则抛 `RuntimeError` 拒绝启动（防止多实例并发运行）。锁文件在 `run()` 开始时创建、结束时删除。

#### 公共方法

```
run() -> dict
```

**返回值结构**：

```
{
    "total_iterations": int,   # 成功晋升的轮次数
    "final_round": int,        # 停止时的轮次编号
    "rounds": [                # 每轮明细列表
        {
            "round_num": int,  # 轮次编号
            "gen4_wins": int,  # 父代（cnn_v3.4）胜场
            "gen5_wins": int,  # 子代（cnn_v3.5）胜场
            "draws": int,      # 平局数
            "gen5_win_rate": float,  # 子代胜率（百分比，如 48.0，一位小数精度）
            "promoted": bool,  # 本轮是否晋升
        },
        ...
    ],
    "stop_reason": str,        # 停止原因文本，格式: "第 {round_num} 轮 cnn_v3.5 胜率 {rate}% ≤ 50.0%，停止迭代"
    "final_model_path": str,   # 最终 gen4 权重文件的绝对路径
}
```

#### 异常

| 异常类型 | 抛出场景 |
|----------|----------|
| `FileNotFoundError` | gen4/cnn_v3.pth 不存在 |
| `RuntimeError` | 子进程返回非零退出码、生成的样本数据为空、JSON 解析失败、或锁文件存在（多实例并发） |
| `ValueError` | 胜率解析失败 |
| `IOError` | 文件复制失败（如磁盘满，晋升 copy2 或初始化 copy2） |

---

## 2. 已有模块变更

**无**。本次不修改任何已有文件。

---

## 3. 文件系统约定

| 路径（相对项目根） | 说明 | 模式 |
|----|------|------|
| `ai/models/cnn_v3/gen4/cnn_v3.pth` | 父代权重（必须存在） | 读 + 被覆盖（晋升时） |
| `ai/models/cnn_v3/gen5/cnn_v3.pth` | 子代权重 | 每轮被覆盖 |
| `ai/merge_samples/v4/path{round_num}.json` | 第 round_num 轮训练样本 | 每轮新建 |
| `ai/models/cnn_v3/.iterative_lock` | 锁文件（防多实例并发） | run() 开始创建、结束时删除 |
