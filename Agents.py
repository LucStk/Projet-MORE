import numpy as np
from numpy.lib.shape_base import _take_along_axis_dispatcher
from utils import *


class Random_agent:
    def __init__(self,env):
        self.env = env
    
    def act(self):
        return np.random.choice(self.env.actions_possibles())

    def learn(self):
        pass

    def time_to_learn(self):
        pass

    def store(self, ob, reward,new_ob):
        pass

class Standar_Ql_agent:
    """
    Standar Ql agent sans replay buffer.
    """
    def __init__(self,env, epsilon, alpha, discount):
        self.discount = discount
        self.alpha    = alpha
        self.env      = env
        self.epsilon  = epsilon
        self.Q_values = np.random.randn(*env.transitions.shape)
    
    def act(self):
        """
        Choix epsilon greedy
        """
        id_current_state     = self.env.id_state(self.env.curent_state)
        id_accessible_states = np.where(self.env.transitions[id_current_state])[0]
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions_possibles())
        
        Q = self.Q_values[id_current_state][id_accessible_states]
        a = np.argmax(Q)
        return self.env.states[id_accessible_states[a]]


    def learn(self):
        (ob, reward,new_ob) = self.lastTransition
        id_ob  = self.env.id_state(ob)
        new_ob = self.env.id_state(ob)
        
        new_action = self.env.id_state(self.act())

        self.Q_values[id_ob][new_ob] = (1-self.alpha)*self.Q_values[id_ob][new_ob] + \
                                self.alpha*(np.sum(reward) + \
                                self.discount*self.Q_values[new_ob][new_action])

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        self.lastTransition = (ob, reward, new_ob)
