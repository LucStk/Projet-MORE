import numpy as np
from numpy.lib.shape_base import _take_along_axis_dispatcher
from utils import *
from random import sample


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
    def __init__(self,env, epsilon, alpha, discount, scal = [1,1], 
                 batch_size = 10):
        """
        env      : le modèle de markov
        epsilon  : probabilité d'une décision aléatoire
        alpha    : paramètre de soft update, pourcentage de Q_cible à prendre
        discount : le discount reward
        scal     : coefficient du produit scalaire pour le calcul du reward
        """
        self.batch_size = batch_size
        self.discount = discount
        self.alpha    = alpha
        self.env      = env
        self.epsilon  = epsilon
        self.scal     = scal
        self.Q_values = np.random.randn(*env.transitions.shape)
        self.memory   =  [] #set()
    
    def act(self):
        return self.act_state(self.env.current_state)

    def act_state(self, state):
        """
        Choix epsilon greedy
        """
        id_current_state     = self.env.id_state(state)
        id_accessible_states = np.where(self.env.transitions[id_current_state])[0]
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions_possibles())
        
        Q = self.Q_values[id_current_state][id_accessible_states]
        a = np.argmax(Q)
        return self.env.states[id_accessible_states[a]]


    def learn(self):
        """
        On apprend sur la transaction courrante ainsi que sur 10 autres
        séléctionnés aléatoirement
        """
        batch = [self.lastTransition] 
        try:
            batch + sample(self.memory, self.batch_size)
        except ValueError:
            batch + list(self.memory)

        for (ob, reward, new_ob) in batch:

            id_ob  = self.env.id_state(ob)
            id_new_ob  = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))

            self.Q_values[id_ob][new_ob] = (1-self.alpha)*self.Q_values[id_ob][new_ob] + \
                                    self.alpha*(np.sum(reward*np.array(self.scal)) + \
                                    self.discount*self.Q_values[new_ob][new_action])

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        self.lastTransition = (ob, tuple(reward[0]), new_ob)
        self.memory.append(self.lastTransition)
        #self.memory = self.memory.union({self.lastTransition})
        
