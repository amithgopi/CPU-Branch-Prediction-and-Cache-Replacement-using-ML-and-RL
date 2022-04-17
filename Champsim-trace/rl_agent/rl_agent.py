# import gym
import random
import torch
import numpy as np
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as plt

import csv
import os
import time
import struct


from torch import nn
from collections import deque, namedtuple


_r_fd = int(os.getenv("PY_READ_FD"))
_w_fd = int(os.getenv("PY_WRITE_FD"))


_r_pipe = os.fdopen(_r_fd, 'rb', 0)
_w_pipe = os.fdopen(_w_fd, 'wb', 0)


def _read_n(f, n):
    buf = ''
    while n > 0:
        data = f.read(n)
        if data == '':
            raise RuntimeError('unexpected EOF')
        buf += data.decode("utf-8") 
        n -= len(data)
    return buf


def _api_get(apiName, apiArg):
    # Python sends format
    # [apiNameSize][apiName][apiArgSize][apiArg]
    # on the pipe
    msg_size = struct.pack('<I', len(apiName))
    _w_pipe.write(msg_size)
    _w_pipe.write(apiName.encode('utf-8'))

    apiArg = str(apiArg)  # Just in case
    msg_size = struct.pack('<I', len(apiArg))
    _w_pipe.write(msg_size)
    _w_pipe.write(apiArg.encode('utf-8'))


# APIs to C++
def send_to_pipe(arg):
    return _api_get("action", arg)


def read_from_pipe():
    # Response comes as [resultSize][resultString]
    buf = _read_n(_r_pipe, 4)
    msg_size = struct.unpack('<I', buf.encode('utf-8'))[0]
    data = _read_n(_r_pipe, msg_size)
    if data == "__BAD API__":
        raise Exception(data)
    return data



class NeuralNetwork(nn.Module):
    def __init__(self, states_count, action_count):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(states_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



REPLAY_BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory():
    def __init__(self):
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def addToBuffer(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)



    def getSamplesFromBuffer(self):
        experiences = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def length(self):
        return len(self.memory)

    



ALPHA = 5e-4 
GAMMA = 0.99
EPSILON = 1.0

class Agent():
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count
        self.epsilon = EPSILON
        
        self.learning_network = NeuralNetwork(state_count, action_count).to(device)
        self.target_network = NeuralNetwork(state_count, action_count).to(device)
        self.optimizer = optim.Adam(self.learning_network.parameters(), lr=ALPHA)
        
        self.memory = Memory()
        
    def updateEpsilon(self):
        self.epsilon = max(0.01, self.epsilon*0.995)
        
    
    def step(self, state, action, reward, next_state, done):
        # Add to memory
        self.memory.addToBuffer(state, action, reward, next_state, done)
        
        # We want the agent to learn on every step from memory
        if(self.memory.length() > BATCH_SIZE):
            experiences = self.memory.getSamplesFromBuffer()
            self.learn(experiences)
        
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.learning_network.eval()
        with torch.no_grad():
            action_values = self.learning_network(state)
        self.learning_network.train()
        self.updateEpsilon()
        
        # Select action based on Epsilon-greedy approach
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_count))
        
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        q_expected = self.learning_network(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = func.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Target network
        for target_param, local_param in zip(self.target_network.parameters(), self.learning_network.parameters()):
            target_param.data.copy_(0.001*local_param.data + (1.0-0.001)*target_param.data)
        
    

def getStateFromCsv(csv_res):
    env_status = csv_res.split(',')
    state = [int(env_status[1]) , int(env_status[2]) , int(env_status[3]) , int(env_status[4])]
    # print(state)
    return np.array(state)


def getNextStateAndRewardFromCsv(csv_res, takenAction):
    env_status = csv_res.split(',')
    state = [int(env_status[1]) , int(env_status[2]) , int(env_status[3]) , int(env_status[4])]
    reward = 1 if (int(env_status[3]) == takenAction) else -1
    # print(csv_res, reward)
    # print(state)
    return np.array(state),reward


scores = []
action = 0
scores_window = deque(maxlen=100)
agent = Agent(state_count=4, action_count=2)

episode = 0
while(1):
    # state = env.reset()
    score = 0
    res = read_from_pipe()
    # print(res)
    if "env" in res:
        state = getStateFromCsv(res)
        # get current state here state

    # agent performs action on current state
    action = agent.act(state)
    send_to_pipe(action)
    # print(action)
    
    # environment returns the results of those actions
    res = read_from_pipe()
    if "reward" in res:
        next_state,reward = getNextStateAndRewardFromCsv(res, action)
        # calculate reward
        # create these variables, next_state, reward from res 
    
    # agent learns from the response of environment
    agent.step(state, action, reward, next_state, True)
    
    # increment score
    score += reward
    
    # Update the current state
    # state = next_state - not needed


    scores.append(score)
    scores_window.append(score)
    episode = episode + 1
    if episode%100 == 0:
        print('Instruction {}, Average Score: {:.2f}\n'.format(episode, np.mean(scores_window)), end="")
        scores_window.clear



torch.save(agent.learning_network.state_dict(), 'checkpoint.pth')

# def window_avg_points(og_points, window=1):
#     i = 0
#     new_points = []
#     while i < len(og_points):
#         new_points.append(np.mean(og_points[i: i+window]))
#         i += window
#     return new_points

# win_scores = window_avg_points(scores, 50)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(win_scores)), win_scores)
# plt.ylabel('Score')
# plt.xlabel('Episode # / 50')
# plt.show()
        