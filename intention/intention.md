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