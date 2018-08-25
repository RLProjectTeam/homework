import random
from copy import deepcopy

from environment.env import Environment
from environment.mazebase_wrapper import MazebaseWrapper
from environment.observation import ObservationTuple, Observation
from utils.constant import *
from utils.add_switch_in_alice_subtask import replace_switch


class SelfPlay(Environment):
    """
    Wrapper class over self play environment

    SelfPlay supports two modes:
    * COPY aka REPEAT where the Bob should repeat what Alice did
    In this setting, Bob start from same position as Alice and has to reach the same end position as Alice.
    * UNDO where Bob should undo what Alice did
    In this setting, Bob start from the end position of Alice and has to reach the start end position of Alice.
    """

    def __init__(self, environment, task=None):
        super(SelfPlay, self).__init__()
        self.environment = environment
        self.name = SELFPLAY + "_" + self.environment.name
        self.alice_start_environment = None
        self.alice_end_environment = None
        # The environment (along with the state) in which alice starts
        self.agent_id = 0
        self.agents = (ALICE, BOB)
        self.observation = Observation()
        self.alice_observations = ObservationTuple()
        self.bob_observations = ObservationTuple()
        _all_possible_actions = self.environment.all_possible_actions()
        self.stop_action = len(_all_possible_actions)
        self.actions = _all_possible_actions
        self.is_over = None
        self.task = task

    def _process_observation(self):
        self.observation.reward = 0.0
        self.observation.is_episode_over = self.is_over

    def observe(self):
        self._process_observation()
        return self.observation

    def reset(self):
        self.observation = self.environment.reset()
        self.alice_observations = ObservationTuple()
        self.bob_observations = ObservationTuple()
        self.is_over = False
        self.agent_id = 0
        return self.observe()

    # =============================================================================
    ''' 如果alice设置了新的switch位置，让bob能看加switch后新环境；
         alice在未设置新switch前看到的环境是移除了old_switch的环境， 
         alice设置了middle_state位置后的，看见的环境是 原环境中的old_switch替换为new_switch
         注意，这里只考虑task=copy，即bob的target是alice的end(保持了环境的一致性)
    '''
    
    def alice_observe(self,alice_middle_loc=None):
        if alice_middle_loc == None:
            observation = replace_switch(self.environment, alice_middle_loc) # 直接观察环境（移除old switch)
        else:
            observation = replace_switch(self.environment, alice_middle_loc)
        return (observation, self.alice_observations.start)
    
    def bob_observe(self, alice_middle_loc=None):
        if(self.environment.are_states_equal(self.observation.state, self.bob_observations.target.state)):
            self.is_over = True
        
        if alice_middle_loc == None:
            observation = replace_switch(self.environment, alice_middle_loc) # observation = self.observe()
        else:
            observation = replace_switch(self.environment, alice_middle_loc)
        return (observation, self.bob_observations.target)    
   
    
    def alice_start(self):
        # Memory=None is provided to make the interface same as selfplay_memory
        self.agent_id = 0
        self.is_over = False
        ########## remove old switch ###################
        self.alice_observations.start =   deepcopy( replace_switch(self.environment, None)) 
        if (self.task == COPY):
            self.alice_start_environment = self.environment.create_copy()

    def alice_stop(self):
        self.agent_id = -1
        self.is_over = True
        self.alice_observations.end = deepcopy( self.alice_observe()[0] )  ###### deepcopy(self.observe())
        if(self.task == UNDO):
            self.alice_end_environment = self.environment.create_copy()


    def bob_start(self):
        self.agent_id = 1
        self.is_over = False
        if (self.task == COPY):
            self.environment.load_copy(self.alice_start_environment)
            self.bob_observations.target = deepcopy(self.alice_observations.end)
            self.bob_observations.start = deepcopy(self.bob_observe()[0])
            
            if (not self.environment.are_states_equal(self.bob_observations.start.state,
                                                      self.alice_observations.start.state)):
                print("Error in initialising Bob's environment")
        elif(self.task == UNDO):
            self.environment.load_copy(self.alice_end_environment)
            self.bob_observations.target = deepcopy(self.alice_observations.start)
            self.bob_observations.start = deepcopy(self.bob_observe()[0])
            #self.observation = self.environment.observe()
            #self.bob_observations.start = deepcopy(self.observe())
            if(not self.environment.are_states_equal(self.bob_observations.start.state,
                                                     self.alice_observations.end.state)):
                print("Error in initialising Bob's environment")


    def bob_stop(self):
        self.agent_id = -1
        self.is_over = True
        self.bob_observations.end_observation =  deepcopy(self.bob_observe()[0])  #deepcopy(self.observe())
     # =============================================================================



    def agent_stop(self):
        if (self.agent_id == 0):
            self.alice_stop()
        elif (self.agent_id == 1):
            self.bob_stop()

    def get_current_agent(self):
        return self.agents[self.agent_id]

    def switch_player(self):
        self.agent_id = (self.agent_id + 1) % 2

    def display(self):
        return self.environment.display()

    def is_over(self):
        return self.is_over

    def act(self, action):
        prev_agent_id = self.agent_id
        if (action == self.stop_action):
            self.agent_stop()
        elif (action != self.stop_action):
            self.observation = self.environment.act(action=action)
        if (prev_agent_id == 0):
            return self.alice_observe()
        elif (prev_agent_id == 1):
            return self.bob_observe()
        else:
            return self.observe()

    def all_possible_actions(self, agent=ALICE):
        if (agent == ALICE):
            return self.actions + [self.stop_action]
        elif (agent == BOB):
            return self.actions

        return self.actions

    def get_task(self):
        return self.task

    def set_task(self, task=COPY):
        play.task = task

    def set_seed(self, seed):
        self.environment.set_seed(seed)


if __name__ == "__main__":
    play = SelfPlay(environment=MazebaseWrapper())
    # env.display()
    actions = play.all_possible_actions()
    print(actions)
    for i in range(100):
        print("==============")
        _action = random.choice(actions)
        print(_action)
        play.act(_action)
        print((play.observe()).reward)
