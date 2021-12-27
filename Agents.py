import numpy as np
from numpy.lib.shape_base import _take_along_axis_dispatcher
from torch.autograd import backward
from utils import *
from random import sample
from sklego.linear_model import LowessRegression
from sklearn.exceptions import NotFittedError
import torch

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
        self.memory   =  list()
    
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
            id_new_ob = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))


            self.Q_values[id_ob][id_new_ob] = (1-self.alpha)*self.Q_values[id_ob][id_new_ob] + \
                                    self.alpha*(np.sum(reward*np.array(self.scal)) + \
                                    self.discount*self.Q_values[id_new_ob][new_action])


    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        self.lastTransition = (ob, tuple(reward[0]), new_ob)
        self.memory.append(self.lastTransition)

class Switch_Ql_agent:
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
        self.Q_values1 = np.random.randn(*env.transitions.shape)
        self.Q_values2 = np.random.randn(*env.transitions.shape)
        self.memory   =  list()
        self.memory_reward = np.zeros(2)

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
        
        if self.memory_reward[0] < self.memory_reward[1]:
            Q = self.Q_values1[id_current_state][id_accessible_states]
        else:
            Q = self.Q_values2[id_current_state][id_accessible_states]

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
            id_new_ob = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))


            self.Q_values1[id_ob][id_new_ob] = (1-self.alpha)*self.Q_values1[id_ob][id_new_ob] + \
                                    self.alpha*(reward[0] + \
                                    self.discount*self.Q_values1[id_new_ob][new_action])

            self.Q_values2[id_ob][id_new_ob] = (1-self.alpha)*self.Q_values2[id_ob][id_new_ob] + \
                        self.alpha*(reward[1] + \
                        self.discount*self.Q_values2[id_new_ob][new_action])

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        self.lastTransition = (ob, tuple(reward[0]), new_ob)
        self.memory.append(self.lastTransition)
        self.memory_reward += reward[0]


class MORE_DL_Ql_agent:
    """
    Standar Ql agent sans replay buffer.
    """
    def __init__(self,env, epsilon, alpha, discount, scal = [1,1], 
                 batch_size = 10, discount_reward = 0.99, sigma = 0.4):
        """
        env      : le modèle de markov
        epsilon  : probabilité d'une décision aléatoire
        alpha    : paramètre de soft update, pourcentage de Q_cible à prendre
        discount : le discount reward
        scal     : coefficient du produit scalaire pour le calcul du reward
        """
        self.batch_size = batch_size
        self.discount = discount
        self.discount_reward = discount_reward
        self.alpha    = alpha
        self.env      = env
        self.epsilon  = epsilon
        self.scal     = scal
        self.Q_values = {}
        self.w = np.random.random(2)
        self.w /= np.sum(self.w)
        #self.w = self.init_w()
        self.memory   =  list()
        self.memory_reward = np.zeros(2)
        self.sigma = sigma
        
        params = []
        for state in self.env.states:
            self.Q_values[self.env.id_state(state)]={}
            for id_action in self.env.id_states_possibles_from(state):
                self.Q_values[self.env.id_state(state)][id_action] = torch.nn.Sequential(torch.nn.Linear(2, 30), torch.nn.ReLU(),
                                                                                         torch.nn.Linear(30,2))
                params += list(self.Q_values[self.env.id_state(state)][id_action].parameters())

        self.opti = torch.optim.Adam(params, lr = 0.01)
        self.loss = torch.nn.HuberLoss()

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

        Q = []
        for action in id_accessible_states:
            Q.append(-self.w @ np.exp(-self.Q_values[id_current_state][action](torch.Tensor(self.w)).detach().numpy()))
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

        for (ob, reward, new_ob, old_w, new_w) in batch:

            id_ob  = self.env.id_state(ob)
            id_new_ob = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))

            self.opti.zero_grad()
            with torch.no_grad():
                Q_lin = reward + self.discount*self.Q_values[id_new_ob][new_action](torch.Tensor(new_w)).detach().numpy()

            self.loss(self.Q_values[id_ob][id_new_ob](torch.Tensor(old_w)), torch.Tensor(Q_lin)).backward()
            self.opti.step()

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        old_w = self.w.copy()
        self.w[0] = self.w[0]**self.discount_reward * np.exp(-reward[0][0])
        self.w[1] = self.w[1]**self.discount_reward * np.exp(-reward[0][1])
        self.w /= np.sum(self.w)

        self.lastTransition = (ob, tuple(reward[0]), new_ob, old_w, self.w.copy())
        self.memory.append(self.lastTransition)




class MORE_DL_agent:
    """
    Standar Ql agent sans replay buffer.
    """
    def __init__(self,env, epsilon, alpha, discount, scal = [1,1], 
                 batch_size = 10, discount_reward = 0.99, sigma = 0.4):
        """
        env      : le modèle de markov
        epsilon  : probabilité d'une décision aléatoire
        alpha    : paramètre de soft update, pourcentage de Q_cible à prendre
        discount : le discount reward
        scal     : coefficient du produit scalaire pour le calcul du reward
        """
        self.batch_size = batch_size
        self.discount = discount
        self.discount_reward = discount_reward
        self.alpha    = alpha
        self.env      = env
        self.epsilon  = epsilon
        self.scal     = scal
        self.Q_values = {}
        self.w = np.random.random(2)
        self.w /= np.sum(self.w)
        #self.w = self.init_w()
        self.memory   =  list()
        self.memory_reward = np.zeros(2)
        self.sigma = sigma
        
        params = []


        self.Q_values = torch.nn.Sequential(torch.nn.Linear(6+2, 30), torch.nn.Sigmoid(),
                                            torch.nn.Linear(30,2))

        self.opti = torch.optim.Adam(self.Q_values.parameters(), lr = 0.001)
        self.loss = torch.nn.MSELoss()

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

        Q = []
        for action in id_accessible_states:
            input = torch.cat((self.env.one_hot(id_current_state,action),torch.Tensor(self.w)))
            Q.append(-self.w @ np.exp(-self.Q_values(input).detach().numpy()))


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

        for (ob, reward, new_ob, old_w, new_w) in batch:

            id_ob  = self.env.id_state(ob)
            id_new_ob = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))

            self.opti.zero_grad()
            with torch.no_grad():
                input = torch.cat((self.env.one_hot(id_new_ob,new_action),torch.Tensor(self.w)))
                Q_lin = reward + self.discount*self.Q_values(input).detach().numpy()

            input = torch.cat((self.env.one_hot(id_ob,id_new_ob),torch.Tensor(self.w)))
            self.loss(self.Q_values(input), torch.Tensor(Q_lin)).backward()
            self.opti.step()

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        old_w = self.w.copy()
        self.w[0] = self.w[0]**self.discount_reward * np.exp(-reward[0][0])
        self.w[1] = self.w[1]**self.discount_reward * np.exp(-reward[0][1])
        self.w /= np.sum(self.w)

        self.lastTransition = (ob, tuple(reward[0]), new_ob, old_w, self.w.copy())
        self.memory.append(self.lastTransition)

class MORE_Ql_agent:
    """
    Standar Ql agent sans replay buffer.
    """
    def __init__(self,env, epsilon, alpha, discount, scal = [1,1], 
                 batch_size = 10, discount_reward = 0.99, sigma = 0.4):
        """
        env      : le modèle de markov
        epsilon  : probabilité d'une décision aléatoire
        alpha    : paramètre de soft update, pourcentage de Q_cible à prendre
        discount : le discount reward
        scal     : coefficient du produit scalaire pour le calcul du reward
        """
        self.batch_size = batch_size
        self.discount = discount
        self.discount_reward = discount_reward
        self.alpha    = alpha
        self.env      = env
        self.epsilon  = epsilon
        self.scal     = scal
        self.Q_values = {}
        self.w = np.random.random(2)
        self.w /= np.sum(self.w)
        #self.w = self.init_w()
        self.memory   =  list()
        self.memory_reward = np.zeros(2)
        self.sigma = sigma

        for state in self.env.states:
            self.Q_values[self.env.id_state(state)]={}
            for id_action in self.env.id_states_possibles_from(state):
                self.Q_values[self.env.id_state(state)][id_action] = LowessRegression(sigma=0.1)
                self.Q_values[self.env.id_state(state)][id_action].fit(self.w.reshape(-1, 1), np.random.random((1,2)))

    def init_w(self):
        w = np.zeros(self.env.transitions.shape)
        for state in self.env.states:
            for id_action in self.env.id_states_possibles_from(state):
                w[self.env.id_state(state)][id_action] = np.random.random()

        w /= np.sum(w)

        return w

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

        Q = []

        for action in id_accessible_states:
            Q.append(-self.w @ np.exp(-self.Q_values[id_current_state][action].predict(self.w.reshape(-1, 1))))


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

        for (ob, reward, new_ob, old_w, new_w) in batch:

            id_ob  = self.env.id_state(ob)
            id_new_ob = self.env.id_state(new_ob)
            new_action = self.env.id_state(self.act_state(new_ob))

            

            self.Q_lin = (1-self.alpha)*self.Q_values[id_ob][id_new_ob].predict(old_w.reshape(-1, 1)) + \
                                    self.alpha*(reward + \
                                    self.discount*self.Q_values[id_new_ob][new_action].predict(new_w.reshape(-1, 1)))

            self.Q_values[id_ob][id_new_ob].fit(old_w, Q_lin)

    def time_to_learn(self):
        return True

    def store(self, ob, reward,new_ob):
        old_w = self.w.copy()
        new_w = self.w.copy()
        self.w[0] = self.w[0]**self.discount_reward * np.exp(-reward[0][0])
        self.w[1] = self.w[1]**self.discount_reward * np.exp(-reward[0][1])
        self.w /= np.sum(self.w)

        self.lastTransition = (ob, tuple(reward[0]), new_ob, old_w, self.w.copy())
        self.memory.append(self.lastTransition)
