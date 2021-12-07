from utils import *
from Agents import *
import torch
import numpy as np
import matplotlib.pyplot as plt


env = Markov_chain('env0')

nb_itérations = 100
reward = np.zeros((1,env.rewards.shape[1]))
S = []
env.reset()

agent = Standar_Ql_agent(env, epsilon = 0.1, alpha = 0.2, discount=0.90)

for i in range(nb_itérations):
    s = env.curent_state
    new_s, r = env.action(agent.act())
    agent.store(s, r, new_s)
    if agent.time_to_learn():
        agent.learn()

    
    S += [s]
    reward = np.concatenate((reward,r), axis = 0)
print(S)
print(reward)

###Plot

plt.plot(reward)
plt.show()
plt.plot(np.cumsum(reward, axis = 0))
plt.show()
print("")