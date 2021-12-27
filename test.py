from utils import *
from Agents import *
import torch
import numpy as np
import matplotlib.pyplot as plt


env = Markov_chain('env0')

nb_itérations = 20000
reward = np.zeros((1,env.rewards.shape[1]))
S = []
env.reset()

"""
Standar QL paramètre du papier:
epsilon  : 0.5
alpha    : 0.7
discount : 0.9
"""

agent = Standar_Ql_agent(env, epsilon = 0.5, alpha = 0.2, 
                         discount=0.90, scal = [1, 1])

for i in range(1,nb_itérations):
    s = env.current_state
    new_s, r = env.action(agent.act())
    agent.store(s, r, new_s)
    if agent.time_to_learn():
        agent.learn()

    if i % 2000 == 0:
        agent.epsilon /= 2
    S += [s]
    reward = np.concatenate((reward,r), axis = 0)
print(S)
print(reward)

###Plot

cumsum = np.cumsum(reward, axis = 0)
plt.plot(cumsum[:,0], label="reward 1")
plt.plot(cumsum[:,1], label="reward 2")
plt.legend()
plt.show()
print()