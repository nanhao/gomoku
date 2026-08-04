/** 全局运行时配置实例定义
 *
 * mcts_config.h 中声明为 extern，此处为唯一定义点。
 * 默认值由 MctsConfig 结构体提供，bindings.cpp 在模块初始化时覆盖。
 */

#include "mcts_config.h"

MctsConfig g_mcts_config;
