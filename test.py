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

agent = MORE_DL_agent(env, epsilon = 0.5, alpha = 0.2, 
                         discount=0.90, scal = [1, 1])

for i in range(nb_itérations):
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

c = np.cumsum(reward, axis = 0)
plt.plot(c[:,0], label="reward 1")
plt.plot(c[:,1], label="reward 2")
plt.legend()

r1 = [c[o,0] if o - 1000 < 0 else c[o,0] - c[o-1000, 0] for o,i in enumerate(c)]
r2 = [c[o,1] if o - 1000 < 0 else c[o,1] - c[o-1000, 1] for o,i in enumerate(c)]

average1000_r1 = [sum(r1[:i])/1000 \
                  if i < 1000 else sum(r1[i-1000:i])/1000 \
                         for i, r in enumerate(r1) ]

average1000_r2 = [sum(r2[:i])/1000 \
                  if i < 1000 else sum(r2[i-1000:i])/1000 \
                         for i, r in enumerate(r2) ]

plt.plot(average1000_r1, label="reward 1")
plt.plot(average1000_r2, label="reward 2")
plt.legend()
plt.show()
plt.show()
print()