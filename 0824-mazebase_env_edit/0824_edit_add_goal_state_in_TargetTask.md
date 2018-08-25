# 修改1：enviroment/mazebase_wrapper.py

featurizers.grid_one_hot(self.game, obs)
obs = np.array(obs).flatten()  ### maze_size, maze_size, 78  eg[10,10,78] ###
featurizers.vocabify(self.game, info)
info = np.array(info).flatten()  #  [10,10] ########## before (obs)?????   ###########
                
#game_observation[OBSERVATION] = np.concatenate((obs, info), 0)  # ver 1: not good

game_observation[OBSERVATION] =obs   # ver2 

# 修改2：policy/mazebase_policy.py
default_input_size = 10 * 10 * 78   #####(78+1)

# 修改3：environment/mazebase_wrapper.py
增加函数get_state_at_goal(self)：''' get the state at goal, then put it as s* in target task, instead of using [0,0....0]'''

【增加修改：去掉door
具体修改如下：开门后的observation中'door'不会消失，不过
 ['Switch', 'state0']  ----> ['Switch', 'state1']  #open door
】
  
# 修改4： environment/selfplay_target.py
在target task中，bob输入的是(s_t, 0)，这里改为输入(s_t, goal_state)  

 def reset(self):
        ####################################################
        ''' edit: make use of selfplay trainning. Here not input [0,0,...0] as the second element, but state_at_goal(just like s* in selfplay)'''
        
        self.observation = self.environment.reset()
        
        self.bob_observations.goal = self.environment.get_state_at_goal()  #############
        
        print(self.bob_observations.goal,'bob goal')
        
        self.bob_observations.goal.state = self.environment.get_state_at_goal().state ############
        
        self.is_over = False
        
        self.agent_id = 1
        
        return self.observe()
 
 
 # 补充修改：动作空间减小(mazebase_wrapper.py
 self.actions = ['up', 'down', 'right', 'left', 'toggle_switch']   ########## edit ###############
 
