# mcts_battle.py 入口脚本设计

## 概述

CLI 入口脚本，启动两个 MCTS 模型的对战。支持单进程和多进程模式，参数体系与 `mcts_self_play.py` 对齐。

## 文件

`src/ai/scripts/mcts_battle.py`

## CLI 参数

```
--model1        必需  模型1 权重文件路径
--model1-name   可选  模型1 注册名.代数（默认从路径自动解析）
--model2        必需  模型2 权重文件路径
--model2-name   可选  模型2 注册名.代数（默认从路径自动解析）
--games         必需  对战局数
--simulations   可选  每步 MCTS 模拟次数（默认 800）
--c-puct        可选  PUCT 探索系数（默认 2.0）
--temp-steps    可选  温度探索步数（默认 8）
--temperature   可选  温度参数（默认 1.0）
--dirichlet-epsilon  可选  狄利克雷噪声比例（默认 0.25）
--dirichlet-alpha    可选  狄利克雷分布参数（默认 0.03）
--batch-size    可选  每批 simulation 数（默认读 MCTS_BATCH_SIZE 配置）
--virtual-loss  可选  virtual loss 值（默认读 MCTS_VIRTUAL_LOSS 配置）
--no-prior      可选  禁用模型先验概率
--device        可选  推理设备（默认 cuda）
--workers       可选  并行 worker 进程数（默认 CPU 核数，1 为单进程串行）
```

用法示例：
```
python src/ai/scripts/mcts_battle.py --model1 models/cnn_v8/gen10/cnn_v8.pth --model2 models/cnn_v8/gen11/cnn_v8.pth --games 100
python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10.pth --model1-name cnn_v8.10 --model2 cnn_v8.11.pth --model2-name cnn_v8.11 --games 200 --workers 4 --simulations 400
```

## 脚本结构

脚本文件仅包含以下内容（遵循"入口脚本可定义函数"的例外规则）：

- `parse_args()` — 解析 CLI 参数，返回 argparse.Namespace
- `_build_hyperparams(args)` — 从 args 构建超参数字典（与 `mcts_self_play._build_hyperparams` 复用）
- `_distribute_indices(total, num_chunks)` — 均匀分配 game_indices（与 `mcts_self_play._distribute_indices` 复用）
- `run_battle(args)` — 主流程入口
- `_run_single_process(args, hyperparams, run_id, workers)` — 单进程模式
- `_run_multi_process(args, hyperparams, run_id, workers)` — 多进程模式
- `_print_statistics(model1_name, model2_name, results)` — 打印胜率统计
- `if __name__ == "__main__"` — 入口

## 主流程

1. 解析参数，确定 worker 数
2. 多进程模式下强制 device="cpu"
3. 构建 hyperparams → `MctsSelfPlayConfig`
4. 生成本次运行 run_id
5. 根据 workers 选择单进程或多进程模式
6. 收集所有 `BattleGameResult`
7. 打印统计信息

## 输出格式

### 每局结果

```
[局 1/100] model1(黑) vs model2(白) → 黑方胜 (45步)
[局 2/100] model1(白) vs model2(黑) → 白方胜 (52步)
[局 3/100] model1(黑) vs model2(白) → 平局 (225步)
```

### 最终统计

```
=== 对战结果 ===
总局数: 100
cnn_v8.10 胜: 53 局 (53.0%)
cnn_v8.11 胜: 42 局 (42.0%)
平局: 5 局 (5.0%)
```

## 与 mcts_self_play.py 的差异

| 项 | mcts_self_play.py | mcts_battle.py |
|----|-------------------|----------------|
| 模型 | 单个 `--model` | 两个 `--model1` `--model2` |
| 输出 | sample.json + summary.json | 终端打印统计信息 |
| 周期保存 | 每 50 局写入磁盘 | 无 |
| SIGINT 处理 | 中断后保存已有样本 | 中断后打印已有统计 |
| Worker 类 | MctsSelfPlayWorker | MctsBattleWorker |
| 对局循环 | MctsGameRunner + on_move 收集样本 | MctsGameRunner（无回调） |
