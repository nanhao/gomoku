# 测试文件完整清单

## tests/ 根目录

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `conftest.py` | 125 | 配置 |
| `run_tests.py` | 60 | 运行器 |
| `test_controller.py` | ~300 | 游戏控制器 |
| `test_game.py` | ~220 | 游戏逻辑 |
| `test_save.py` | ~280 | 存档管理 |
| `test_write_guard.py` | ~220 | 写入保护 |

## tests/ai/

| 文件 | 行数 | 测试类型 | 训练 epochs |
|------|------|----------|-------------|
| `conftest.py` | 46 | 配置 | - |
| `helpers.py` | ~80 | 工具函数 | - |
| `test_base_model_multi_gen.py` | ~220 | 多代模型 | - |
| `test_base_trainer.py` | 277 | 训练器基类（收敛性验证） | 400, 200 |
| `test_base_trainer_train_auto.py` | 217 | 训练分发（需 mock） | 50, 1000 |
| `test_base_trainer_val.py` | ~250 | 验证集 | 200×2 |
| `test_battle.py` | ~360 | 对战 | - |
| `test_bugfix_dense_log_and_val_ratio.py` | ~260 | Bug 修复 | - |
| `test_cnn_models.py` | 643 | CNN 模型（v1~v6） | - |
| `test_cnn_trainers.py` | 234 | CNN 训练器 | 5 |
| `test_cnn_v7_models.py` | 363 | CNN v7 | - |
| `test_create_samples.py` | 426 | 样本创建 | - |
| `test_data_recorder.py` | 469 | 数据记录 | - |
| `test_engine.py` | 336 | AI 引擎 | - |
| `test_engine_no_fallback.py` | ~60 | 引擎无回退 | - |
| `test_evaluator.py` | 305 | 评估器 | - |
| `test_fc_v1_model.py` | 360 | FC v1 模型 | - |
| `test_fc_v1_softmax_input.py` | ~180 | softmax | 50 |
| `test_fc_v1_trainer.py` | 386 | FC v1 训练器 | 200 |
| `test_fc_v1_trainer_integration.py` | ~70 | FC v1 集成 | - |
| `test_game_filter.py` | ~140 | 游戏过滤 | - |
| `test_game_manager.py` | 356 | 游戏管理 | - |
| `test_game_manager_record.py` | 349 | 记录管理 | - |
| `test_game_manager_temp.py` | ~300 | 临时目录 | - |
| `test_graceful_shutdown.py` | ~180 | 优雅退出（需 mock） | 10×4 |
| `test_is_variant_move.py` | ~100 | 变体检测 | - |
| `test_kl_div_migration.py` | ~300 | KL 散度 | - |
| `test_loss_logger.py` | 437 | 损失日志 | 1000×2 |
| `test_loss_logger_val.py` | ~140 | 验证日志 | - |
| `test_merge_samples.py` | ~250 | 样本合并 | - |
| `test_mini_batch_training.py` | ~330 | mini batch | 200 |
| `test_mini_batch_training_edge.py` | ~280 | mini batch 边界 | 5 |
| `test_move_sampler.py` | ~210 | 走法采样 | - |
| `test_neural_network_model.py` | 432 | 神经网络模型 | - |
| `test_performance_level1.py` | 337 | 优化正确性（Mock+predict+JSON+tensor） | - |
| `test_performance_level2_numba.py` | ~370 | 优化正确性（JIT 函数 + 1 基准类） | - |
| `test_performance_level2_parallel.py` | ~350 | 优化正确性（并行一致性） | - |
| `test_random_model.py` | ~170 | 随机模型 | - |
| `test_registry.py` | ~310 | 注册表 | - |
| `test_registry_multi_gen.py` | ~380 | 多代注册表 | - |
| `test_reward_calculator.py` | ~300 | 奖励计算 | - |
| `test_runners.py` | ~150 | 运行器 | 1 |
| `test_sample_creator.py` | ~350 | 样本创建器 | - |
| `test_sample_creator_variant_move.py` | 368 | 变体走法 | - |
| `test_sample_creator_winner_path.py` | ~330 | 胜者路径 | - |
| `test_sample_creator_winner_path_soft_label.py` | 341 | 软标签 | - |
| `test_sample_generator.py` | ~260 | 样本生成器 | - |
| `test_sample_loader.py` | ~330 | 样本加载器 | - |
| `test_sample_visualizer.py` | ~280 | 样本可视化 | - |
| `test_self_play_generator.py` | 276 | 自我对弈 | - |
| `test_shape_detector.py` | 410 | 形状检测 | - |
| `test_train_main.py` | ~300 | 训练主入口 | - |

## tests/ai/integration/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_cnn_v6_integration.py` | 149 | CNN v6 集成 |
| `test_iterative_train.py` | 363 | 迭代训练 |
| `test_sample_games_cli.py` | ~450 | CLI 样本生成 |
| `test_sample_games_cli_progress.py` | ~280 | 进度显示 |
| `test_train_main_lr.py` | ~270 | 学习率 |

## tests/ai/unit/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_registry.py` | ~250 | 注册表 |
| `test_iterative_train_params.py` | 350 | 参数解析 |
| `test_mini_batch_training.py` | ~300 | mini batch |

## tests/e2e/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_multi_gen_flow.py` | 363 | 多代流程 |

## tests/gui/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `conftest.py` | ~50 | 配置 |
| `test_ai_battle.py` | ~360 | AI 对战 GUI |
| `test_display_updater.py` | ~350 | 显示更新 |
| `test_game_board.py` | ~180 | 游戏棋盘 |
| `test_gomoku_gui.py` | ~300 | 主界面 |
| `test_heatmap.py` | ~310 | 热力图 |
| `test_main.py` | 353 | 入口 |
| `test_mode_selector.py` | ~160 | 模式选择 |
| `test_replay_browser.py` | ~170 | 回放浏览 |
| `test_replay_viewer.py` | ~340 | 回放查看 |

## tests/cli/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_main.py` | ~320 | CLI 主入口 |
| `test_train_main_cli.py` | ~100 | 训练 CLI |
| `test_train_main_temp_cleanup.py` | ~200 | 临时清理 |

## tests/app/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_battle.py` | ~200 | 对战 |
| `test_battle_model_load_error.py` | ~110 | 加载错误 |
| `test_gui_helper.py` | ~200 | GUI 辅助 |
| `test_session.py` | ~140 | 会话 |
| `test_session_model_load_error.py` | ~100 | 会话错误 |

## tests/tools/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_app.py` | ~200 | 工具应用 |
| `test_data_parser.py` | 344 | 数据解析 |
| `test_executor.py` | ~290 | 执行器 |

## tests/unit/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| `test_heatmap_color.py` | ~260 | 热力图颜色 |
| `test_lr_range_test.py` | ~230 | LR range |

## tests/integration/

| 文件 | 行数 | 测试类型 |
|------|------|----------|
| （已无测试文件，端到端测试已合并至其他模块） |
