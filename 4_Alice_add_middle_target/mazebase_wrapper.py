import random
from copy import deepcopy

import mazebase
# These weird import statements are taken from https://github.com/facebook/MazeBase/blob/23454fe092ecf35a8aab4da4972f231c6458209b/py/example.py#L12
import mazebase.games as mazebase_games
import numpy as np
from mazebase.games import curriculum
from mazebase.games import featurizers

from environment.env import Environment
from environment.observation import Observation
from utils.constant import *


class MazebaseWrapper(Environment):
    """
    Wrapper class over maze base environment
    """

    def __init__(self):
        super(MazebaseWrapper, self).__init__()
        self.name = MAZEBASE
        try:
            # Reference: https://github.com/facebook/MazeBase/blob/3e505455cae6e4ec442541363ef701f084aa1a3b/py/mazebase/games/mazegame.py#L454
            small_size = (10, 10, 10, 10)
            lk = curriculum.CurriculumWrappedGame(
                mazebase_games.LightKey,
                curriculums={
                    'map_size': mazebase_games.curriculum.MapSizeCurriculum(
                        small_size,
                        small_size,
                        (10, 10, 10, 10)
                    )
                }
            )

            game = mazebase_games.MazeGame(
                games=[lk],
                featurizer=mazebase_games.featurizers.GridFeaturizer()
            )


        except mazebase.utils.mazeutils.MazeException as e:
            print(e)
        self.game = game
        self.actions = self.game.all_possible_actions()

    def observe(self):
        game_observation = self.game.observe()
        # Logic borrowed from:
        # https://github.com/facebook/MazeBase/blob/23454fe092ecf35a8aab4da4972f231c6458209b/py/example.py#L192
        obs, info = game_observation[OBSERVATION] 
        featurizers.grid_one_hot(self.game, obs)
        obs = np.array(obs).flatten()  ### maze_size, maze_size, 78  eg[10,10,78] ###
        featurizers.vocabify(self.game, info)
        info = np.array(info).flatten()  #  [10,10] ########## before (obs)?????   ###########
        
        
#        game_observation[OBSERVATION] = np.concatenate((obs, info), 0)   #### edit ver1: not good #####
        game_observation[OBSERVATION] = obs   ############## edit ver2 #################

        is_episode_over = self.game.is_over()
        return Observation(id=game_observation[ID],
                           reward=game_observation[REWARD],
                           state=game_observation[OBSERVATION],
                           is_episode_over=is_episode_over)
    
    def get_state_at_goal(self):
        ''' get the state at goal, then put it as s* in target task, instead of using [0,0....0]'''
        def lst_sub(lst1, lst2):
            ''' lst1 - lst2'''
            lst_out = deepcopy(lst1)
            for e in lst2:
                lst_out.remove(e)
            return lst_out
        
        def place_agent_at_goal(obs_str_lst):
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
                    if check_agent_at_grid:
                        #(i,j): col(left to right), row(bottom to top)
                        new_obs_str_lst[i][j] = lst_sub( obs_str_lst[i][j],  agent_ele)
                        
                    if check_this_grid_is_goal:
                        new_obs_str_lst[i][j] = lst_add( obs_str_lst[i][j], agent_ele)
            return new_obs_str_lst
    
        
        
        
        game_observation = self.game.observe()
        obs_now, info = game_observation[OBSERVATION]
        obs_at_goal = place_agent_at_goal(obs_now)
        
        featurizers.grid_one_hot(self.game, obs_at_goal)
        obs_at_goal = np.array(obs_at_goal).flatten()  ### maze_size, maze_size, 78  eg[10,10,78] ###
        
        featurizers.vocabify(self.game, info)
        info = np.array(info).flatten()  #  [10,10] ########## before (obs)?????   ###########
        
        
#        obs_at_goal_vector = np.concatenate((obs_at_goal, info), 0)  ########## edit ver 1 ###########
        obs_at_goal_vector = obs_at_goal   ######### edit ver2################
        return Observation(id=game_observation[ID],
                           reward = 0,
                           state=obs_at_goal_vector,
                           is_episode_over=True)
        
 
    
    def reset(self):
        try:
            self.game.reset()
        except Exception as e:
            print(e)
        return self.observe()

    def display(self):
        return self.game.display()

    def is_over(self):
        return self.game.is_over()

    def act(self, action):
        self.game.act(action=action)
        return self.observe()

    def all_possible_actions(self):
        return self.actions

    def set_seed(self, seed):
        # Not needed here as we already set the numpy seed
        pass

    def create_copy(self):
        return deepcopy(self.game.game)

    def load_copy(self, env_copy):
        self.game.game = env_copy

    def are_states_equal(self, state_1, state_2):
        return np.array_equal(state_1, state_2)


if __name__ == "__main__":
    env = MazebaseWrapper()
    env.display()
    print('state at goal',env.get_state_at_goal())
    actions = env.all_possible_actions()
    print(actions)
    for i in range(10):
        print("==============")
        _action = random.choice(actions)
        print(_action)
        env.act(_action)
        env.display()
