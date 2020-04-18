import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
from dqn import DQN

orgDQN, multistep = False, False

if len(sys.argv) == 1:
    print("There is no argument, please input")

for i in range(1,len(sys.argv)):
    if sys.argv[i] == "orgDQN":
        orgDQN = True
    elif sys.argv[i] == "multistep":
        multistep = True


env = gym.make('CartPole-v1')

if orgDQN:
    print("start Original")
    env.reset()
    dqn = DQN(env, multistep=False)
    orgDQN_record = dqn.learn(1500)
    del dqn

if multistep:
    print("start Multi-step")
    env.reset()
    dqn = DQN(env, multistep=True)
    multistep_record = dqn.learn(1500)
    del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

x = np.arange((1500))

if orgDQN:
    plt.plot(np.arange((len(orgDQN_record))), orgDQN_record, label='Orginal DQN')
if multistep:
    plt.plot(np.arange((len(multistep_record))), multistep_record, label='Multistep DQN')

plt.legend()
fig =plt.gcf()
plt.savefig("result.png")
plt.show()