# -*- coding: utf-8 -*-
"""
将agent到达goal时的observation得到
observation_str ---> new_observation_str
"""

from copy import deepcopy


def lst_sub(lst1, lst2):
    ''' lst1 - lst2'''
    lst_out = deepcopy(lst1)
    for e in lst2:
        lst_out.remove(e)
    return lst_out
    

def replace_lst_element(lst,e_old, e_new):
    '''把lst中e_old元素替换为e_new'''
    new_lst = []
    for e in lst:
        if e == e_old:
            e = e_new
        new_lst.append(e)
    return new_lst
        

def place_agent_at_goal(obs_str_lst):
    
    '''把agent放到goal，以及switch状态要改变，得到新的observation字符串
    # ['Switch', 'state0']  ----> ['Switch', 'state1']  #open door '''
    
    new_obs_str_lst = deepcopy(obs_str_lst)
    agent_ele = ['SingleTileMovable', 'Toggling', 'Agent']
    Goal_ele = ['Goal', 'goal_id0']

    
    set_intersect = lambda set1,set2: set1&set2 
    lst_add = lambda lst1,lst2 : lst1+lst2
    
    
    for i in range(len(obs_str_lst)):
        col = obs_str_lst[i]  # one col in mazebase
        for j in range(len(col)):
            check_agent_at_grid = set_intersect(set(col[j]), set(agent_ele ) ) # check agent at this grid. If not, return null set
            check_this_grid_is_goal = set_intersect( set(col[j]), set(Goal_ele))
            check_this_grid_is_switch = set_intersect(set(col[j]), {'Switch'} )
            
            if check_agent_at_grid:
                #(i,j): col(left to right), row(bottom to top)
                new_obs_str_lst[i][j] = lst_sub( obs_str_lst[i][j],  agent_ele)
                
            if check_this_grid_is_goal:
                new_obs_str_lst[i][j] = lst_add( obs_str_lst[i][j], agent_ele)
            
            
            if check_this_grid_is_switch:
                new_obs_str_lst[i][j] = replace_lst_element( obs_str_lst[i][j], 
                                         'state0' ,'state1')
            
    return new_obs_str_lst


#####
if __name__=='__main__':
    obs_str_lst =[[['Corner'], [], ['Block'], ['Goal', 'goal_id0'], ['Block'], [], [], [], [], ['Corner']], [[], ['Block'], [], [], ['Block'], ['Switch', 'state0'], [], [], [], []], [[], [], [], ['Water'], ['Block'], [], ['Water'], [], [], []], [[], ['Water'], [], [], ['Block'], [], [], [], ['Water'], []], [[], [], [], [], ['Block'], [], [], [], [], []], [['Block'], [], ['Block'], [], ['Block'], [], [], ['Toggling', 'Agent', 'SingleTileMovable'], [], ['Block']], [[], [], [], ['Block'], ['Block'], [], ['Block'], [], [], []], [[], [], [], [], ['Block'], [], [], [], [], []], [[], [], [], [], ['Block'], [], [], [], [], []], [['Corner'], [], [], [], ['Door', 'open', 'state1'], [], [], [], [], ['Corner']]]
    place_agent_at_goal(obs_str_lst)