# -*- coding: utf-8 -*-
'''由[app]selfplay中的selfplay.environment.game.observe()['observation'][0] 
即当前环境字符串lst信息，得到agent在哪里；
并且如果alice决定在当前位置设置一个switch，需要把环境本身的switch个删去，并把新switch加入到其中'''

from copy import deepcopy
import numpy as np
from mazebase.games import featurizers
from environment.observation import Observation


def find_agent_loc(SelfplayEpisode_selfplay_env):
    ''' input is the selfplay.environment in  function run_selfplay_episode(selfplay....)
    
    find the current location of agent
    
    selfplay.environment is class MazebaseWrapper, 
    use 【   selfplay.environment.game.observe()['observation'][0]   】
    can output the environment details
    '''
    
    obs_str_lst =  SelfplayEpisode_selfplay_env.game.observe()['observation'][0]  
    
    for i in range(len(obs_str_lst)):
        for j in range(len(obs_str_lst[i])):
            e_lst = obs_str_lst[i][j]
            if 'Agent' in e_lst:
                return i,j


def lst_sub(lst1, lst2):
    ''' pop the element of lst1 which is in lst2'''
    out_lst = []
    for e in lst1:
        if e not in lst2:
            out_lst.append(e)
    return out_lst


def replace_switch(SelfplayEpisode_selfplay_env, new_switch_loc=None, is_over = False):  ## debug: add is_over
    ''' 
    SelfplayEpisode_selfplay_env表示SelfplayEpisode_selfplay.environment
    new_switch_loc=(i,j) the location of middle_state 
    output Observation Class
    '''
    game_observation  = SelfplayEpisode_selfplay_env.game.observe()
    obs_str_lst = game_observation['observation'][0] 
    new_obs_str_lst = deepcopy(obs_str_lst)
    
    # remove old switch and add new switch
    for i in range(len(obs_str_lst)):
        for j in range(len(obs_str_lst[i])):
            e_lst = obs_str_lst[i][j]
            
            if 'Switch' in e_lst:   # remove old switch
                e_lst_new = lst_sub(e_lst, ['Switch','state0'])
                new_obs_str_lst[i][j] = e_lst_new
            
            if new_switch_loc == None:
                pass
            else: # add new switch
                if i==new_switch_loc[0] and j==new_switch_loc[1]:  
                    new_obs_str_lst[i][j] += ['Switch','state0']
    
    # map new_obs_str_lst to one-hot vector
    featurizers.grid_one_hot(SelfplayEpisode_selfplay_env.game,  new_obs_str_lst)
    obs_vec = np.array(new_obs_str_lst).flatten()
    print(obs_vec,'obs vec ================')
    return Observation(id = game_observation['id'],
                       reward = game_observation['reward'],
                       state= obs_vec,
                       is_episode_over = is_over)
    
    
if __name__ == '__main__':
    obs_str_lst = [[['Corner'], [], ['Block'], [], ['Agent', 'Toggling', 'SingleTileMovable'], [], ['Block'], ['Block'], ['Switch', 'state0'], ['Corner']], [[], ['Water'], ['Block'], [], [], [], ['Water'], [], [], ['Block']], [['Block'], ['Block'], ['Block'], ['Door', 'open', 'state1'], ['Block'], ['Block'], ['Block'], ['Block'], ['Block'], ['Block']], [[], [], [], [], ['Block'], ['Water'], ['Block'], [], [], []], [[], [], [], [], [], [], [], [], [], []], [[], ['Block'], [], ['Block'], [], [], ['Water'], [], [], []], [['Block'], [], [], [], [], [], [], [], [], []], [[], [], [], ['Water'], [], ['Block'], [], [], [], []], [[], [], [], [], [], [], [], [], [], []], [['Water', 'Corner'], [], ['Block'], ['Goal', 'goal_id0'], [], [], [], [], [], ['Corner']]]