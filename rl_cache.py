import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io,time



import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob

import csv
from math import log2

class Optimal:

    def __init__(self, _trace_filename, numset_=2048, num_way=16, block_size=64):
        self.trace_filename = _trace_filename
        self.NUM_SET = numset_
        self.NUM_WAY = num_way
        self.BLOCK_SIZE = block_size
        self.OFFSET_BITS=int(log2(self.BLOCK_SIZE))
        self.distance_dictionary={}
        self.generate_distance_dictionary(self.trace_filename)
        self.cache=[]
        self.generate_initial_cache()

        

    def generate_distance_dictionary(self,_trace_filename ):
        #print("generating optimal distance")
        csvReader = csv.DictReader(open(_trace_filename))
        index=0
        for row in csvReader:
            if((index+1)%1000000==0):
                print(index)
                break
            try :
                self.distance_dictionary[int(row['address'])].append(index)
            except KeyError:
                self.distance_dictionary[int(row['address'])]=[index]
            index+=1
        #print("total memory access:",index)

    
    def generate_initial_cache(self):
        #print("generating initial cache")
        for x in range(self.NUM_SET):
            self.cache.append([])
            for y in range(self.NUM_WAY):
                self.cache[x].append({"address":999999999,"distance":999999999})

    def bitmask(self, begin, end = 0) :
        return ((1 << (begin - end)) - 1) 

    def if_present_in_cache(self,address :int,set_:int):
        #print("if_present_in_cache")
        way=0
        if_present=False
        for y in range(self.NUM_WAY):
                #print(y, set_)
                #print("y:",y,"self.cache[set_][y]['address']:",self.cache[set_][y]['address'],"address:",address,"way:",way,"if_present:",if_present)
                if(self.cache[set_][y]["address"]==address):
                    way=y
                    if_present=True 
                    #print("self.cache[set_][y]['address']:",self.cache[set_][y]['address'],"address:",address,"way:",way,"if_present:",if_present)
                    return if_present,way
        return if_present,way

    def find_victim(self,address,set_):
        max_distance=0
        max_way=0
        for y in range(self.NUM_WAY):
            if max_distance<self.cache[set_][y]['distance']:
                max_distance=self.cache[set_][y]['distance']
                max_way=y
        return max_distance,max_way

    def get_next_distance_from_address(self,address,current_distance):
        check=False
        for y in self.distance_dictionary[address]:
            if(y>current_distance):
                check=True
                return y
        if(not check):
            return 999999999

    def accessCache(self,address,set_,way,index):
        next_distance=self.get_next_distance_from_address(address,index)
        self.cache[set_][way]={"address":address,"distance":next_distance}
        
    def run_simulation(self):
        #print("running simulation")
        index =0
        hit=0

        csvReader = csv.DictReader(open(self.trace_filename))
        for row in csvReader:
    
            if((index+1)%1000000==0):
                print(index)
            
            address=int(row['address'])
            set_= ((int(row['address']) >> self.OFFSET_BITS) & self.bitmask(int(log2(self.NUM_SET))))
    
            (if_present,way)=self.if_present_in_cache(address,set_)
            
            if(if_present):
                hit+=1
                next_distance=self.get_next_distance_from_address(address,index)
                self.cache[set_][way]={"address":address,"distance":next_distance}
            else:
                (max_distance,max_way)=self.find_victim(address,set_)
                next_distance=self.get_next_distance_from_address(address,index)
                self.cache[set_][max_way]={"address":address,"distance":next_distance}
            
            index+=1
        #print("hit: ",hit,"total memeory access: ",index)

import math
import sys

accessTypes = { 'LD':0, 'RFO': 1, 'PF': 2, 'WB':3, 'TL': 4 }

class Block():
    def __init__(self, numWays = 16):
        self.tag: int = 0
        self.valid: bool = False
        self.offset: int = 0
        # self.dirty: bool = False
        self.preuse: int = int(sys.maxsize)
        self.preuseCounter: int = 0
        self.ageSinceInsertion: int = 0
        self.ageSinceAccess: int = 0 
        self.accessType: int = 0
        self.accessCounts = [0, 0, 0, 0, 0]
        self.hits: int = 0
        self.recency: int = numWays - 1
    
    def getState(self):
        state = [self.offset, 1 if self.valid else 0, self.preuse, self.ageSinceInsertion, self.ageSinceAccess, self.accessType]
        state.extend(self.accessCounts)
        state.extend([self.hits, self.recency])
        return state
               

class Cache():
    def __init__(self, numSets_ = 2048, numWays_ = 16, blockSize_ = 65):
        self.numSets: int = numSets_
        self.numWays: int = numWays_
        self.blockSize: int = blockSize_
        self.BLOCKS = [Block(numWays = self.numWays) for _ in range(self.numSets*self.numWays)]
        self.setAccesses = [0 for _ in range(self.numSets)]
        self.setAccessesSinceMiss = [0 for _ in range(self.numSets)]
        self.preuseDistances = {}
        self.globalAccessCount = 0
        
        self.offsetBits: int = int(math.log2(self.blockSize))
        self.setBits: int = int(math.ceil(math.log2(self.numSets)))
        self.setBitMask: int = (1<<self.setBits)-1
        self.offsetBitMask: int = (1<<self.offsetBits)-1
        
    def splitAddress(self, address: int):
        setIdx: int = (address >> self.offsetBits) & self.setBitMask
        offset: int = address & self.offsetBitMask
        tag: int = address >> (self.offsetBits + self.setBits)
        return tag, setIdx, offset
    
    # TODO Need to normalize the state access count value
    def getCurrentState(self, address: int, accessType: int):
        tag, setIdx, offset = self.splitAddress(address)
        # Get preuse of cache
        preuse = sys.maxsize
        cacheLineAddress = address >> self.offsetBits # Address of cache line, remove offset from address
        if cacheLineAddress in self.preuseDistances:
            # User a global access counter to compute preuse distance as its differnce to the value in the preuseDistance dictionary
            preuse = globalAccessCount - self.preuseDistances[cacheLineAddress]
        
        blocks = self.BLOCKS[setIdx*self.numWays: setIdx*self.numWays + self.numWays]
        state = [offset, preuse ,accessType] # Access Info
        state.extend( [setIdx, self.setAccesses[setIdx], self.setAccessesSinceMiss[setIdx] ] ) # set info
        # cache line info
        for line in blocks:
            state.extend(line.getState())
        return state
    
    def updateRecency(self, setIdx, way):
        blocks = self.BLOCKS[setIdx*self.numWays: setIdx*self.numWays + self.numWays]
        # Store recency of block being updated
        currentBlockRecency = blocks[way].recency
        blocks[way].recency = 0
        # Update recency of all those lower that current
        for i in range(self.numWays):
            if blocks[i].recency < currentBlockRecency:
                blocks[i].recency += 1
        
    def accessCache(self, address: int, accessType: int, way: int):
        self.globalAccessCount += 1
        cacheLineAddress = address >> self.offsetBits
        # Update the preuseDistances dict to the current value of globalAccessCount on each access to a cache line address
        if cacheLineAddress in self.preuseDistances:
            self.preuseDistances[cacheLineAddress] = self.globalAccessCount
        # Split address to parts
        addressParts = self.splitAddress(address)
        tag, setIdx, offset = addressParts
        setBlockIndex = setIdx*self.numWays
        #update set params
        self.setAccesses[setIdx] += 1
        self.setAccessesSinceMiss[setIdx] += 1
        # Check for hits and update block params
        hit: bool = False
        way_i = 0
        for i in range(self.numWays):
            self.BLOCKS[setBlockIndex + i].ageSinceInsertion += 1 #reset on miss
            self.BLOCKS[setBlockIndex + i].ageSinceAccess += 1 #reset on hit
            self.BLOCKS[setBlockIndex + i].preuseCounter += 1
            
            if self.BLOCKS[setBlockIndex + i].tag == tag and self.BLOCKS[setBlockIndex + i].valid:
                hit = True
                way_i = i
        if hit:
            self.handleHit(setIdx, way_i, accessType, (tag, setIdx, offset) )
        else:
            self.handleMiss(setIdx, way, accessType, addressParts)
            self.setAccessesSinceMiss[setIdx] = 0
    
        self.updateRecency(setIdx, way) 
        return hit
            
    def handleHit(self, setIdx, way, accessType, addressParts):
        block: Block = self.BLOCKS[setIdx*way]
        tag, setIdx, offset = addressParts
        # Update block params
        block.offset = offset
        block.preuse = block.preuseCounter
        block.preuseCounter = 0
        block.ageSinceAccess = 0
        block.accessType = accessType
        block.accessCounts[accessType] += 1
        block.hits += 1

    def handleMiss(self, setIdx, way, accessType, addressParts):
        block: Block = self.BLOCKS[setIdx*way]
        tag, setIdx, offset = addressParts
        # Update block params
        block.valid = True
        block.tag = tag
        block.offset = offset
        block.preuse = int(sys.maxsize)
        block.preuseCounter = 0
        # block.ageSinceAccess = 0
        block.ageSinceInsertion = 0
        block.accessType = accessType
        block.accessCounts[accessType] += 1
             
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(time.time())

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state,eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        #print("agnet.act")
        # Epsilon-greedy action selection
        if random.random() > eps:
            #print("np.argmax(action_values.cpu().data.numpy():",np.argmax(action_values.cpu().data.numpy()))
            return np.argmax(action_values.cpu().data.numpy())
        else:
            action_var=random.randrange(0,self.action_size)
            #print("random.choice(np.arange(self.action_size))124:",action_var,self.action_size)      
            return action_var

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



cache_model=Cache()
reward_model=Optimal("cache_620.csv")

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    csvReader = csv.DictReader(open("cache_620.csv"))
    index=0
    
    for cache_fillup in range(500000):
        row=next(csvReader)
        address=int(row['address'])
        set_var= int(row['set'])
        type_var= int(row['type'])
        action=random.randrange(0,reward_model.NUM_WAY)
        cache_model.accessCache(address,type_var,action)

        reward_model.accessCache(address,set_var,action,index)
        if(index==2):
            print("**reward_model.cache[set_var]:",reward_model.cache[set_var])
            print(set_var, cache_model.splitAddress(address))
            #blocks = self.BLOCKS[setIdx*self.numWays: setIdx*self.numWays + self.numWays]
            for x in range(set_var*cache_model.numWays,set_var*cache_model.numWays + cache_model.numWays):
                print(cache_model.BLOCKS[x].valid, cache_model.BLOCKS[x].tag)
        index+=1

# initialize epsilon
    for i_episode in range(1, n_episodes+1):
        #state = env.reset()
        score = 0
        hit=0
        count=0
        print("episode: ",i_episode)
        for t in range(max_t):
            row=next(csvReader)
            address=int(row['address'])
            set_var= int(row['set'])
            type_var= int(row['type'])
            print("index:",index,t,"t:",t,"address:",address,"set_var:",set_var,"type_var:",type_var)
            (if_present,way)=reward_model.if_present_in_cache(address,set_var)
            #print("if_present:",if_present,"way:",way)
            #print("reward_model.cache[set_var]:",reward_model.cache[set_var])
            if(if_present):
                cache_model.accessCache(address,type_var,way)
                reward_model.accessCache(address,set_var,way,index)
                index+=1
                
                hit+=1
                #print(hit)
                count+=0
                continue
                
            
            
            state= np.asarray(cache_model.getCurrentState(address,type_var), dtype=np.float32)

            action = agent.act(state, eps)
            
            victim_distance,victim_way=reward_model.find_victim(address,set_var)
            #print("victim_distance:",victim_distance,"victim_way:",victim_way,"reward_model.get_next_distance_from_address(address,index):",reward_model.get_next_distance_from_address(address,index))
            if(victim_way==action):
                reward=1
                #print('success')
            elif (victim_distance<reward_model.get_next_distance_from_address(address,index)) :#we hav eto fin index
                reward=-1
            else :
                reward =0
            #print("action:",action,"reward:",reward)
            cache_model.accessCache(address,type_var,action)
            reward_model.accessCache(address,set_var,action,index)
            
            next_state=cache_model.getCurrentState(address,type_var)
            done=0
            agent.step(state, action, reward, next_state, done)
            print("reward_model.cache[set_var]:",reward_model.cache[set_var])
            #blocks = self.BLOCKS[setIdx*self.numWays: setIdx*self.numWays + self.numWays]
            for x in range(set_var*cache_model.numWays,set_var*cache_model.numWays + cache_model.numWays):
                print(cache_model.BLOCKS[x].valid)
            state = next_state
            score += reward
            index+=1
            count+=1
            #print("\n")

            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} accuracy: {:.5f}'.format(i_episode, np.mean(scores_window), (hit)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} accuracy: {:.5f}'.format(i_episode, np.mean(scores_window), hit))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores



agent = Agent(state_size=214, action_size=16, seed=0)
scores = dqn(2,1)

