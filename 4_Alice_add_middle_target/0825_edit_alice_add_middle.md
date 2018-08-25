记录修改过程

## 修改0-1：enviroment/mazebase_wrapper.py

featurizers.grid_one_hot(self.game, obs)
obs = np.array(obs).flatten()  ### maze_size, maze_size, 78  eg[10,10,78] ###
featurizers.vocabify(self.game, info)
info = np.array(info).flatten()  #  [10,10] ########## before (obs)?????   ###########
                
#game_observation[OBSERVATION] = np.concatenate((obs, info), 0)  # ver 1: not good

game_observation[OBSERVATION] =obs   # ver2 

# 修改0-1：policy/mazebase_policy.py
default_input_size = 10 * 10 * 78   #####(78+1)


# 修改1：app/selfplay.py中run_selfplay_episode函数
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
    
