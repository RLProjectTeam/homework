# homework

## 一、report阅读
### 1.	paper中模拟的环境：

后两个环境都在gym中有

a)	maze（略）

b)	mountain car

c)	acrobat（机器臂）

### 2.	模型【重点的思路部分】

（1）	feature extraction network（单层前馈神经网络+relu）

输入:
      （Alice在这个episode的当前状态，Alice在这个episode的起始状态），称为current episodic tuple；
		
       或者（Bob 在这个episode的当前状态，Bob在这个episode的目标状态）

输出:  feature-dim维的向量


对于augmented self play，这个地方的输入就是(Alice的tuple同上，Alice memory vector)

（2）	memory model

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
|<font color=red>constant</font>| 常量变量的含义| 如果遇到全部大写字母的变量，可能能在这个代码中找到定义|
|config.py|读取config，以及有些key是空值的话，赋予合适的值|base_path默认memory-augmented-self-play-master，模型保存路径[MODEL][SAVE_DIR]默认为memory-augmented-self-play-master/model，<font color=red>TB Params不知道是啥</font>，如果log的key是空则将日志存在"log_{}.txt".format(str(config[GENERAL][SEED])中|





### (1)selfplay.py
代码框架见self play论文

而augmented memory中用到的alice历史信息的定义，如下：

```alice_history = np.concatenate((selfplay.alice_observations.start.state.reshape(1, -1),                                      selfplay.alice_observations.end.state.reshape(1, -1)), axis=1)```

