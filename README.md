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


## 二 reading code
### (1)selfplay.py
代码框架见self play论文

而augmented memory中用到的alice历史信息的定义，如下：
```alice_history = np.concatenate((selfplay.alice_observations.start.state.reshape(1, -1),                                      selfplay.alice_observations.end.state.reshape(1, -1)), axis=1)```

