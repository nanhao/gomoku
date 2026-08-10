# intention_id=1
为 ai/create_samples.py 脚本做一个 html/css 的纯 ui。我大致描述一下 ui 的布局，你负责做原型设计：
- 需包含一个区域，用来填写参数执行脚本。自动预先填充默认值。
- 需包含一个区域，用来展示对局列表。例如，如果本次执行10次对弈，列表则包含10条元素，每条元素列出对局的基本信息
- 需包含一个区域，用来展示三行热力图。第一行是每一步的模型原始输出分布；第二行是经过引擎的温度处理之后的分布；第三行是生成的样本的分布。三行要对齐，从而可以直观的看出每一步的三个热力图有什么区别。如果某一步没有温度处理或者没有样本生成，则用占位图填充，从而保持对齐。这个区域由于会占据很大的宽度，因此需要横向滚动条。每一列标注步数

# intention_id=2
增强 ai/create_samples.py，加上一个过滤机制。即：当对战双方有且仅有一方开启温度采样，且采样方获胜时，才生成样本。否则，不产生样本。这么做的原因是，采样方获胜的对局才具有学习意义。

# intention_id=3
增强 ai/create_samples.py，加上一个参数 --sample_games，接收一个大于0的整数。该参数与 --games 参数互斥，不能同时指定。当指定 --sample_games 时，脚本持续进行自我对弈，直到收集到指定的样本局数时才停下。举例来说，--sample_games=100，说明要持续自我对弈，直到收集到100局样本才停下。

# intention_id=4
目前的 loss 都是 L1，全部要改成 kl-divergence。
另外，SampleInspector 不需要了，删掉。

# intention_id=5
考虑去掉五连追溯奖励机制和现有的 cycle 模式

# intention_id=6
新增一个脚本，这个脚本做以下逻辑：
1, i=1
2, cp ai\models\cnn_v3\gen4\cnn_v3.pth ai\models\cnn_v3\gen5\cnn_v3.pth
3，python ai/create_samples.py --model cnn_v3.4 --mode winner_path --sample_games 500 --output ai\merge_samples\v4\path{i}.json --temp-side random
4, python ai/train_main.py --model cnn_v3.5 --train-from-dir ai/merge_samples/v4/path{i}.json --epochs 1 --batch-size 4096 --shuffle
5, python ai/battle.py --ai1 cnn_v3.4 --ai2 cnn_v3.5 --games 100
6, 判断 5 的胜负方，如果 cnn_v3.5 的胜率超过 50%，i++; mv ai\models\cnn_v3\gen5\cnn_v3.pth ai\models\cnn_v3\gen4\cnn_v3.pth; 然后回到2进行循环。如果 cnn_v3.5 的胜率不超过 50%，停下并报告完整统计信息

# intention_id=7
train_main.py 脚本新增一个参数，可以指定初始学习率

# intention_id=8
battle.py 脚本也要加上温度采样参数，从而支持温度采样

# intention_id=9
ai/create_samples.py 的过滤机制修改。
现有过滤机制：
当对战双方有且仅有一方开启温度采样，且采样方获胜时，才生成样本。否则，不产生样本。
修改后的过滤机制：
当对战双方有且仅有一方开启温度采样，且采样方获胜时，才挑选其中的变异步生成样本。否则，不产生样本。
变异步定义：引擎实际落子位与模型预测的最佳落子位不一致。

另外，删除 winning_move 模式，不再使用这个模式了。
winning_path 模式，不再禁用数据增强。改为默认开启，可以用参数关闭。

# intention_id=10
全面 review 测试代码。
检查项：
- 测试代码是否有 bug
- 测试代码是否有冗余、重复测试
- 是否有无效测试
- 哪些测试的性能可以提升
- 测试代码的结构是否合理
- 测试代码是否有超过500行的大文件
给出修改意见。review 报告放在 design/10/ 目录下

# intention_id=11
新增 cnn_v4 模型，在 cnn_v3 的基础上增加一个32通道隐藏层，其他保持和 cnn_v3 一致

# intention_id=12
修改 ai/iterative_train.py 的逻辑。
原逻辑：
1, i=1
2, cp ai\models\cnn_v3\gen4\cnn_v3.pth ai\models\cnn_v3\gen5\cnn_v3.pth
3，python ai/create_samples.py --model cnn_v3.4 --mode winner_path --sample_games 500 --output ai\merge_samples\v4\path{i}.json --temp-side random
4, python ai/train_main.py --model cnn_v3.5 --train-from-dir ai/merge_samples/v4/path{i}.json --epochs 1 --batch-size 4096 --shuffle
5, python ai/battle.py --ai1 cnn_v3.4 --ai2 cnn_v3.5 --games 100
6, 判断 5 的胜负方，如果 cnn_v3.5 的胜率超过 50%，i++; mv ai\models\cnn_v3\gen5\cnn_v3.pth ai\models\cnn_v3\gen4\cnn_v3.pth; 然后回到2进行循环。如果 cnn_v3.5 的胜率不超过 50%，停下并报告完整统计信息
新逻辑：
1, i=1
2, cp ai\models\cnn_v4\gen99\cnn_v4.pth ai\models\cnn_v4\gen100\cnn_v4.pth
3，python ai/create_samples.py --model cnn_v4.99 --sample_games 500 --output ai\merge_samples2\v100\path.json
4, python ai/train_main.py --model cnn_v4.100 --train-from-dir ai/merge_samples2/v100 --epochs 300 --batch-size 4096 --shuffle
5, python ai/battle.py --ai1 cnn_v4.99 --ai2 cnn_v4.100 --games 100 --temp-moves=2 --temperature 2，打印出 cnn_v4.100 的胜率和 i 的值
6, 不管胜率如何，都 i++，mv ai\models\cnn_v4\gen100\cnn_v4.pth ai\models\cnn_v4\gen99\cnn_v4.pth; 然后回到2进行循环

# intention_id=13
optimze ai/iterative_train.py, extract the following hard-coded arguments into parameters so that it's more flexible.
- --ai1 eg: cnn_v4.99
- --ai2 eg: cnn_v4.100
- --sample_dir eg: ai\merge_samples2

the new logic should be:
1, i=1
2, cp {pth_of_ai1} {pth_of_ai2}
3，python ai/create_samples.py --model {ai1} --sample_games 500 --output {sample_dir}\v{gen_of_ai2}\path.json --temp-moves 100
4, python ai/train_main.py --model {ai2} --train-from-dir {sample_dir}\v{gen_of_ai2} --epochs 300 --batch-size 4096 --shuffle
5, python ai/battle.py --ai1 {ai1} --ai2 {ai2} --games 100 --temp-moves=2 --temperature 2 --temp-side both，print info
6, i++，mv {pth_of_ai2} {pth_of_ai1}; then go back to 2

you should compute {pth_of_ai1} {pth_of_ai2} {gen_of_ai2} correctly.

# intention_id=14
新增 cnn_v5 模型，在 cnn_v4 的基础上增加一个32通道隐藏层，其他保持和 cnn_v4 一致

# intention_id=15
新增 cnn_v6，网络结构是在 cnn_v5 的基础上加上残差块和 bn。对比如下， v5：
1，卷积层：3->32, 3x3
2, relu
3, 卷积层：32->32, 3x3
4, relu
5, 卷积层：32->32, 3x3
6, relu
7, 卷积层：32->32, 3x3
8, relu
9, 卷积层：32->32, 3x3
10, relu
11, 卷积层：32->1, 3x3
12， flatten -> 225
13, LogSoftmax

v6：
1，卷积层：3->32, 3x3
2, bn
3, relu
4, 卷积层：32->32, 3x3
5, bn
6, relu
7, 卷积层：32->32, 3x3
8, bn
9, 残差链接
10, relu
11, 卷积层：32->32, 3x3
12, bn
13, relu
14, 卷积层：32->32, 3x3
15, bn
16, 残差链接
17, relu
18, 卷积层：32->1, 3x3
19， flatten -> 225
20, LogSoftmax

也就是说，把原来的第二、三层卷积层改为一个残差块；第四、五层卷积层改成第二个残差块。同时加上bn

# intention_id=16
修改 cnn_v6，增加一个价值头输出。加在 flatten 后面，使用 tanh，输出一个值域为 [-1,1] 的标量，表示当前局面的胜率

# intention_id=17
把 cnn_v6 升级为 cnn_v7，恢复 intention_id=15里面的 cnn_v6

# intention_id=18
完整 review 整个测试集。现在运行一次全量测试太久了，我需要把单次运行的时间压缩到一分钟左右

# intention_id=19
把所有源码统一挪到 src 目录下

# intention_id=20
给 src 目录下的每一个子目录（包括 src 目录自己）都创建一个 class_diagram 目录。然后在里面用 pydeps 生成当前目录的包依赖图，svg 格式。同时，若图里的节点大于等于3个，也补充 plantuml 交互图，说明各个节点如何配合实现核心流程。注意交互图里的元素要跟依赖图保持一致。
这个任务比较耗时，你可以唤起多个 subagent 并行执行。

# intention_id=21
app 包和 gui 包存在循环依赖。分析原因并给出解决方案。输出方案在 design/21

# intention_id=22
optimze ai/iterative_train.py to the new logic:
1, i=1
2, cp {pth_of_ai1} {pth_of_ai2}
3，python ai/create_samples.py --model {ai1} --sample_games {sample_games} --output {sample_dir}\v{gen_of_ai2}\path.json --temp-moves 100
4, python ai/train_main.py --model {ai2} --train-from-dir {sample_dir}\v{gen_of_ai2} --epochs {epochs} --batch-size 4096 --shuffle
5, python ai/battle.py --ai1 {ai1} --ai2 {ai2} --games 100 --temp-moves=2 --temperature 2 --temp-side both，print info
6, i++; if the winning rate of ai2 is more than 55%, then: mv {pth_of_ai2} {pth_of_ai1}; then go back to 2; else continue to 2

{sample_games} default to 500, and can be overrided by passing arguments.
{epochs} default to 300, and can be overrided by passing arguments.

you should print info correctly.

# intention_id=23
目前模型只有在训练的时候会使用 gpu，自我对弈的时候不会。我希望在 reate_samples.py 自我对弈的时候也能使用 gpu。

# intention_id=24
cnn_v7 已经实现了策略头和价值头，这一步要实现按照 alphago zero 论文里面的蒙特卡洛树搜索方法。其跟传统 MCTS 有两点不同：
1，选择阶段，纳入了模型输出的策略头作为先验概率
2，模拟阶段，直接将模型的价值头作为模拟的结果
cnn_v7 要采用全新的游戏引擎、训练数据和训练方法。
1，模型的输入有变化，同样是三通道，其中一个通道代表我方落子位置，另一个通道代表对方落子位置，剩下的通道代表我方棋子颜色。
2，训练样本改为当前棋盘状态+MCTS返回的概率分布+本局实际结果（获胜为1，负为-1，平局0）
3，训练时，模型的策略头要拟合 MCTS 的概率分布，价值头要拟合实际结果，可以参照 alphago zero 的论文中的损失函数，但考虑一下要不要防止过拟合的正则项
4，cnn_v7 的自我对弈生成样本、训练都用新的脚本，不复用原来的。
alphago zero 的论文已经收录在本项目根目录下了。

# intention_id=25
src/ai/scripts/mcts_train.py 要像 train_main.py 一样画出双 y 轴曲线图（loss/ val-loss/ grad）。

# intention_id=26
当前 mcts 包里面的代码是孤岛，和旧体系是完全割裂的。需要做整合。
把 BaseModel.predict() 的返回值做一层封装，叫做 ModelPredict。
ModelPredict 有 type 字段，支持多种类型。目前支持两种：a，单纯只预测落子概率分布（对应 random、fc、 cnn v1-6）；b，概率分布+局面评估 （对应 cnn_v7），先做这一层修改，改完跑全量测试，确保通过

# intention_id=27
refactor the return type of AiEngine.get_move() from tuple to a class called MoveDecision. Put the tuple in this class. Don't change all the functions. Use TDD to do this refactor.

# intention_id=28
now that you have ModeDecision, you can put probs info in the return of AiEngine.get_move() so that the probs argument is no longger needed. use TDD to do this refactor.

# intention_id=29
refactor MCTSMoveDecider.decide():
1, change the function name to get_move
2, change the return type to MoveDecision
3, put the position and the mcts_visit_counts into the move_decision. you should add a _mcts_visit_counts field into the MoveDecision
use TDD to do this refactor

# intention_id=30
remove the reset() function in mcts package.

# intention_id=31
移除当前 mcts 实现当中的 batch 和 virtual loss 机制

# intention_id=32
当前 mcts 不为终局节点创建子节点，导致在终局节点调用 make_move 的时候会失败，这次要解决这个问题。

# intention_id=33
do the folowing refactor:
- move the injection point of temperature/top_k/rng parameters of AiEngine.get_move() into constructor
- remove the log_key parameter of MCTSMoveDecider.get_move(). instead, inject the log_key_prefix into MCTSMoveDecider from the constructor. MCTSMoveDecider.get_move() should maintain the entire log_key internally. the passing prefix should be "run{run_id}_game{game_idx}", and the get_move() should append "_search{search_id}" to the prefix. The search_id is incrementally from 0, that is to say, start from 0. we don't use uuid in step_id any more.
- add a parameter called strategy to MCTSMoveDecider.get_move(). the meaning and logic of this parameter is identical to that of AiEngine.get_move()
- remove the state parameter of MCTSMoveDecider.get_move(). Instead, add board and current_player to it. This make the interface of MCTSMoveDecider.get_move() identical to AiEngine.get_move()

# intention_id=34
since now the terminated leaf tree of the mcts is created. we don't need the calling-side workaround. And what's more, the reset() calling is meaningless because mcts tree is not shared between games right? I think we can simpified the calling-side to only one calling to make_move without checking and reset().
Having this change, we can go further. We can make AiEngine and mcts/decision_maker.py into one hierarchy sharing the get_move function. The interface of get_move() function is identical now because we refactor them in intention-33.
For the make_move() calling. We can add a abstract function called after_move() to the engine interface. The SinglePolicyAiEngine can adopt am empty implementation, while the PolicyValueAiEngine can retain the current implementation.

# intention_id=35
1, refactor: after_move(action) -> after_move(x, y)
2, new feature: add a create_engine function in the engine/builder.py. It should return a appropriate AiEngine instance. The choice logic is based on the ai model. If the model's output contians both policy and value, it should use the PolicyValueEngine, otherwise SinglePolicyEngine
3, new feature: change AiGameManager and battle.py to adopt create_engine() and after_move()
4, All the AiEngine instance should not be shared across games

# intention_id=36
optimze the value head of cnn_v7:
Add a hidden FC layer with ReLU between flatten and the output:
Flatten → FC(225→128) → ReLU → FC(128→1) → Tanh

# intention_id=37
I think the fc layers of the value head takes two many parameters. I prefer to add remove the fc solution and make the value head branches from before the cnnv_out. So it uses another cnnv_out, than it goes into a relu, then fc(225->1). What do you think?

# intention_id=38
optimize src/ai/scripts/mcts_self_play.py. Currently the script save the sample.json to disk after all games are done. We need to change this behavior. 
1, When the user press Ctrl+c during the game, the script needs to append the memoroy data to the sample.json file so that there is no data lost
2, Automatically append the memory data to the sample.json file for each 50 games so that reduce the risk of losing data

# intention_id=39
introduce batch mcts and virtual loss in mcts. But don't use multi thread.

# intention_id=40
测试了屏蔽 value 网络、屏蔽 policy 网络、调大 mcts 评估次数和 batch，感觉还是网络太小支持不了。因为 agz 的方法是从零开始训练，会有很多噪音，模型太小无法从噪音里面去识别出有效信息，训练就停滞。所以决定还是要扩展模型规模和调整训练节奏。
所以新加一个模型 cnn_v8。网络结构和 cnn_v7 的唯一区别就是，在 flatten 后面加一个 fc-linear 层，这样就多了好几万参数了。
由于训练太耗时了，决定把 mcts 做一个退化处理。batch=225，一步搜索 225 次，同时落子改为依照平均奖励值q，而不是 N。

# intention_id=41
目前运行 mcts_self_play 脚本很耗时。需要一个提速的方案。

# intention_id=42
新增 cnn_v9，在 cnn_v8 的基础上，把所有卷积层的通道数增加到 48

# intention_id=43
经过研究，发现 mcts 机制不适合我的五子棋。因为策略头学不会。一开始随机对弈，产生的策略头样本质量很差，前中期全部是噪音，而最后一步却是接近 one-hot。
所以我想改变策略了。
新增 cnn_v10。 cnn_v10 的网络架构是在 cnn_v9 的基础上去掉策略头，只保留价值头。
需要修改现在的游戏流程以适配 cnn_v10，落子策略如下：
- 1，棋盘预处理：取出所有空位，然后排除孤立空位，孤立空位的评估值设置为-1。孤立空位的定义是：上下左右以及四条斜线方向上相邻2格之内没有其他棋子
- 2，将预处理之后剩下的空位，挨个进行模拟落子，如果落子后是终局，则将终局的评估值（1/-1/0）设置为该位置的评估值，否则，将新局面留着给下一步用
- 3，由2得到一组新的局面 g0. 将 g0 输入给 cnn_v10，得到一组评估 v0，取反后得到 v0'
- 4，生成概率分布：将 v0' 填入对应位置的评估值，然后对整个棋盘做一次概率归一化。做概率归一化时，已有落子的位置不参与
- 5，基于4得到的概率分布进行落子决策，贪婪和温度采样的配置不变

本次暂不处理 cnn_v10 的训练。

# intention_id=44
1，新建一个脚本，用于 cnn_v10 自我对弈产生样本，可以参照 src/ai/scripts/mcts_self_play.py 的参数
2，新建一个脚本，用于 cnn_v10 的训练，可以参照  src/ai/scripts/mcts_train.py
3，修改 battle.py，使其能够兼容 cnn_v10

顺便改了回放功能，使其支持 cnn_v10.
结果一训练，还是策略崩溃，根本连随机都完全打不过。分析了原因发现 value 被放大

# intention_id=45
新增 cnn_v11，回到双头结构。cnn_v11 是在 cnn_v9 的基础上，做以下变动：
- 把残差塔的高度增加到7（目前是2）
- 把卷积层的通道数增加到 256
- 策略头改为 cnnv_out(2通道, size=1) -> bn -> relu -> flatten -> fc-linear(输出为15*15) -> logsoftmax
- 价值头改为 value_cnnv_out(1通道,size=1) -> bn -> relu -> flatten -> fc-linear(输出为256) -> relu -> fc-linear(256->1) -> tanh

# intention_id=46
我希望在 cpp 目录下，用 C++ 实现一个 mcts 的版本。有几个技术要点：
- 跟 python 使用相同的权重和网络结构
- 使用多线程，每个 mcts 线程负责一次选择->评估->回传的周期。并且有一个 gpu 线程，负责异步的搜集一批待评估的局面，一次性发给 gpu 进行评估，然后返回给各 mcts 线程进行后续处理。mcts 线程在提交局面给 gpu 线程后，就等待直到其返回
你帮我调研一下，然后来一起讨论一下怎么做。最后目标是产出一份调研报告 intention/cpp-report.md。可以参考 D:\projects\AlphaZeroGomoku 这个项目的实现。

# intention_id=47
本次要实现一个脚本，可以实现并行开启 n 局 cnn_v11 的自我对弈。可以指定对局双方，例如 ai1=cnn_v11.1 ai2=cnn_v11.2.
技术上，要求 n 局游戏必须是在不同的线程中并行进行，使用 c++ 版的 mcts。所有的游戏共享一个全局的线程池。并且由 gpu 进行局面评估。
脚本需要实时打印进度，并在最终完成后打印统计数据和双方胜率。
脚本可以指定温度采样的参数。

# intention_id=48
调整现在的自我对弈 gpu print 信息，加上滑动窗口，avg_batch_size 我希望是最近10个 batch 的平均 size。wait_ratio   我希望是最近5s内的等待率

# intention_id=49
现在要来做一个 cnn_v11 对弈生成样本缓冲区的脚本。技术要点如下：
- 对弈生成样本脚本保持 src/ai/scripts/mcts_battle_cpp.py 的异步架构不变，参数也保持一致
- 同样全程使用 C++，python 只做简单封装，生成样本也由 c++ 完成
- 对弈时，双方轮流交替执方
- 每局对弈结束后，产生样本。样本的数据结构和语义与项目中已有的双头网络样本保持一致
- sample_buffer 是样本缓冲区，它是一个队列，先进先出，sample_buffer_max_size 为8万条样本（局面）
- 每完成10局对弈，就把样本落盘到 sample_buffer。缓冲区 size >= sample_buffer_max_size 时，把最旧的10局对局移出，然后放入最新的10局对局。这里以10局对局左右进出缓冲区的单位，不管每一局里面有多少步。只在统计总size 的时候要看多少步
- 样本启用 D4 群 8 变换增强（旋转+镜像）
- 输出文件组织方式由你来设计
- 实时 print 对弈进展统计信息 

# intention_id=50
现在来做 cnn_v11 的python训练脚本。技术要点如下：
- 从样本缓冲区随机抽出 512 个样本做一个 mini-batch，按照 alphago zero 的损失函数计算 loss，其中正则项用 adamw 的 weight_decay=1e-4 代替
- 一次 mini-batch 更新一个权重，然后再重新随机抽取进行下一个 mini-batch，一共进行 50 次更新后，暂停训练，触发一次 checkpoint 评估
- 评估交给评估器执行。评估器用这个 checkpoint 和历史最佳权重best进行对弈（复用mcts_battle_cpp.py的逻辑）50局（50个游戏线程），前2步采样，之后贪婪。双方轮流交替执方。50局结束后，如果 checkpoint 的胜率 >=55%，则晋升为 best
- 如果晋升失败，则在该 checkpoint 基础上继续训练。连续三次失败后，停止训练

# intention_id=51
目前c++自对弈脚本在双方不同模型的情况下，效率极低，需要重构提升效率。

# intention_id=52
我希望 mcts_sample_buffer.py 和 mcts_train_cnn_v11.py 的 model 参数是可选的。当没有指定 --model 参数时，脚本自动识别 cnn_v11 的最高代数，从最高代数开始。

# intention_id=53
在项目根目录下实现一个 shell 脚本，脚本内容为：
1，执行 python src/ai/scripts/mcts_sample_buffer.py --games 30 --workers 10
2, 如果 1 没有报错，执行 python src/ai/scripts/mcts_train_cnn_v11.py --updates-per-checkpoint 15
3，如果 2 没有报错，回到 1