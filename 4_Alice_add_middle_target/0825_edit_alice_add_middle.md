记录修改过程

## 修改0-1：enviroment/mazebase_wrapper.py

featurizers.grid_one_hot(self.game, obs)
obs = np.array(obs).flatten()  ### maze_size, maze_size, 78  eg[10,10,78] ###
featurizers.vocabify(self.game, info)
info = np.array(info).flatten()  #  [10,10] ########## before (obs)?????   ###########
                
#game_observation[OBSERVATION] = np.concatenate((obs, info), 0)  # ver 1: not good
game_observation[OBSERVATION] =obs   # ver2 


以及在这个文件里action空间减小
self.actions = ['up', 'down', 'right', 'left', 'toggle_switch']   ########## edit ## self.game.all_possible_actions()


# 修改0-2：policy/mazebase_policy.py
default_input_size = 10 * 10 * 78   #####(78+1)


# 修改1：app/selfplay.py中run_selfplay_episode函数【让alice在走的过程中扔钥匙，如果bob在前往goal的途中捡到了钥匙，会有奖励】

注意新增改动：alice的end_state（作为bob的target_state应该要做修改，即应该把环境中本来的switch删去，然后把这个新增的middle_state作为['Switch', 'state0']）

部分改动部分见下方（原代码中用##########标注出了改动部分############）
  ############## add part : set middle state in alice ##################
    def determine_this_state_as_middle(observation):
        import random
        set_middle = False
        p  = random.random()
        x =  random.random()
        if x>=p:
            set_middle = True
        middle_state = observation[0]
        return set_middle, middle_state
    ######################################################################
    
