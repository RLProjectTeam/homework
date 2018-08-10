# homework

## 一、report阅读
### 1.	paper中模拟的环境：

后两个环境都在gym中有

a)	maze

b)	mountain car

c)	acrobat（机器臂）

### 2.	模型【重点的思路部分】

**（1）	feature extraction network（单层前馈神经网络+relu）**

输入:
      （Alice在这个episode的当前状态，Alice在这个episode的起始状态），称为current episodic tuple；
		
       或者（Bob 在这个episode的当前状态，Bob在这个episode的目标状态）

输出:  feature-dim维的向量


对于augmented self play，这个地方的输入就是(Alice的tuple同上，Alice memory vector)

**（2）	memory model**

a)	把last episodic tuple作为memory vector

b)	把最近k个episode的episodic tuple的均值作为memory vector

c)	用lstm来学习出memory vector


### 3.	实验

a) 文中指出mountain car效果不好，本文不考虑用这个试验（found no success as the agent hardly learns anything）

b) 本文中设置的试验episode数

|               | Mazebase        | car       | acrobot  |
| -------------|:--------------: |:-------------:| -----:|
| number of episodes     |300k | 30k|  30k|
| 画图画最近k次的平均reward| 5000      | 1000      |   1000 |
|Batch size | 256     |    1 | 1|


c) Adam optimizer(with learning rate of 0.001) and policy-gradient algorithm with baseline to train the agents


## 二、 reading code(in file: SelfPlay)
### （1）utils (file)

“注”中的略表示也许可以不用详看此代码，不是我们之后改思路的重点

|       py文件        | 目的       | 注 |
| -------------|:--------------: |:---------:|
| util | 设置随机数的seed| 略|
|optim_registry | 根据config，确定optimizer用什么 | 略|
|log | 生成log文件| 略（这个没细看）|
|**constant**| 常量变量的含义| 如果遇到全部大写字母的变量，可能能在这个代码中找到定义|
|config.py|读取config，以及有些key是空值的话，赋予合适的值|base_path默认memory-augmented-self-play-master，模型保存路径[MODEL][SAVE_DIR]默认为memory-augmented-self-play-master/model，TB Params不知道是啥，如果log的key是空则将日志存在"log_42.txt"中|
|argument_parser| 辅助性代码，帮助读入config的| 略|

### （2）scripts

|       py文件        | 目的       | 注 |
| -------------|:--------------: |:---------:|
| filter_json_lines |Preprocess the log file by filtering non-json line 预处理日志文件| 略（没细看）|


### （3）policy
|       py文件        | 目的       | 注 |
| -------------|:--------------: |:---------:|
| base_policy | 指定policy中网络框架，参数  | 一些主要的参数整理见（3.1）|
| polic_config| 设置基本参数 | 略|
|registry| 由config确定了哪个游戏，从而确定用文件夹中哪个policy| 略|


（3.1）**base_policy.py**
+ 策略更新频率：update_frequency = int(policy_config[BATCH_SIZE]) = 32，可见policy_config.py
+ shared_features_size = policy_config[SHARED_FEATURES_SIZE]， 在policy文件夹中的acrobot_policyp.py里赋值是128
+ memory模块：记忆的维度episode_memory_size= config[EPISODE_MEMORY_SIZE]， input_dim=self.shared_features_size，output_dim=self.shared_features_size
+ **特征提取网络（a feature extraction network，单层的前馈神经网络）：如果self_play设为True,input_size变为原先的2倍，即对于Alice的特征提取网络输入的是(current_state, start_state)，Bob的特征提取网络输入的是(current_state, target_state)**
+ actor、critic每一层网络的初始化：用均匀分布来初始化
+ reward：目前的reward只考虑的是agent的observation本身带来的奖励，即observation[0].reward
+ **改进的alice中critic、actor的输入是feature+memory：In case of Alice, these features are concatenated with the features coming from the memory and the concatenated features are fed into the actor and the critic networks.** 对应的代码部分是
```shared_features = torch.cat((F.relu(self.shared_features(data)),                          self.summarize_memory().unsqueeze(0).detach()), dim=1)```
+ update函数待看


### （4）plotter file
根据log文件画出alice、bob在每个指标（key）下的曲线图，保存的路径由config[plot][base_path]指定

## （5）model file
|       py文件        | 目的       | 注 |
| -------------|:--------------: |:---------:|
| base_model | 保存模型、导入模型  | 略|





### (5)selfplay.py
代码框架见self play论文

而augmented memory中用到的alice历史信息的定义，如下：

```alice_history = np.concatenate((selfplay.alice_observations.start.state.reshape(1, -1),                                      selfplay.alice_observations.end.state.reshape(1, -1)), axis=1)```

