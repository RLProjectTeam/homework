import random

from environment.selfplay import SelfPlay
from environment.mazebase_wrapper import MazebaseWrapper
from environment.observation import ObservationTuple, Observation
from utils.constant import *
from copy import deepcopy
import numpy as np

class SelfPlayTarget(SelfPlay):
    """
    Wrapper class over self play environment
    """

    def __init__(self, environment, task=TARGET):
        super(SelfPlayTarget, self).__init__(environment=environment, task=task)
        self.name = SELFPLAY + "_" + TARGET + "_" + self.environment.name
        self.alice_start_environment = None
        self.agent_id = 1
        self.agents = (BOB)
        self.observation = Observation()
        self.alice_observations = None
        self.bob_observations = ObservationTuple()
        _all_possible_actions = self.environment.all_possible_actions()
        self.stop_action = None
        self.actions = _all_possible_actions
        self.is_over = None
        self.task = task

    def observe(self):
        return self.observation

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

    def alice_observe(self):
        return None

    def bob_observe(self):
        observation = self.observe()
        return (observation,  self.bob_observations.goal)  # self.bob_observations.target)


    def alice_start(self):
        return None

    def alice_stop(self):
        return None

    def bob_start(self):
        self.agent_id = 1
        self.is_over = False

    def bob_stop(self):
        return None

    def agent_stop(self):
        return None

    def display(self):
        return self.environment.display()

    def is_over(self):
        return self.is_over

    def act(self, action):
        self.observation = self.environment.act(action=action)
        return self.bob_observe()

if __name__ == "__main__":
    play = SelfPlay(environment=MazebaseWrapper(), task=COPY)
    play.display()
    print(play.observe())
 #   print(sum(play.environment.get_state_at_goal().state))
    actions = play.all_possible_actions()
    print(actions)
    for i in range(10):
        print("==============")
        _action = random.choice(actions)
        print(_action)
        play.act(_action)
        print((play.observe()).reward)
