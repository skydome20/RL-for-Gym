# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:57:38 2018
@author: skydome20
@reference: https://keon.io/deep-q-learning/
"""
import random
import gym
import numpy as np

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN_Agent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000) # remove old, update new
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.DNN_model()
        
    # DNN-model setting 
    def DNN_model(self):
        # init sequential model
        model = Sequential()
        # hidden-1: 24 nodes
        model.add(Dense(20, input_dim=self.state_size, 
                        activation='relu'))
        # hidden-2: 24 nodes
        model.add(Dense(20,  
                        activation='relu'))
        # output: 2 nodes(action)
        model.add(Dense(self.action_size, 
                        activation='linear'))
        # compile designed-model
        model.compile(loss='mse', 
                      optimizer = Adam(lr=self.learning_rate))
        
        return model  # = self.model
    
    # store experience
    def remember(self, state, action, reward, next_state, done):
        # in tuple type
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
    
    
    # DNN-model training by experience
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            
            if done:
                target[0][action] = reward
            else:
                # max(Q value of next_state)
                Q_head = self.model.predict(next_state)[0] # = [0.6, 0.4]
                
                # only update the q-value under the assigned action
                target[0][action] = reward + self.gamma * np.amax(Q_head) 
                
    
            # DNN-model training
            # x = state = [0.1, 0.2, 0.2, 0.4] = state_size
            # y = target = [1.49, 0.24] = action_size
            
            self.model.fit(x=state, y=target, epochs=1, verbose=0)
            
            # epsilon decay (for epsilon-greedy algorithm)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
    # epsilon-greedy algorithm : take action
    def act(self, state):
        
        # random action
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        # take action based on Q-value from DNN-model
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
            # act_values[0] = [0.6, 0.4]

    def save(self, name):
        self.model.save_weights(name) 

    def load(self, name):
        self.model.load_weights(name)



if __name__ == '__main__' :
    
    # init gym env and agent
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # DQN_Agent class
    agent = DQN_Agent(state_size, action_size)
    
    # parameters
    done = False
    batch_size = 30
    EPISODES = 10000
    
    

    # Start training
    for e in range(EPISODES):

            
        state = env.reset() # [-0.02, -0.01, 0.02, 0.006]
        #print('Before, ', state)
        state = np.reshape(state, [1, state_size]) # [[[-0.02, -0.01, 0.02, 0.006]]]
        #print('After, ', state)
        
        steps = 0
        
        while True:
            env.render() # show the dynamic graph (animation)
            
            # take action from DNN-model or random (epsilon-greedy)
            action = agent.act(state)
            
            # get info from action; and 'env.step()' has done it for you
            # return next_state, reward, done, info
            next_state, reward, done, _ = env.step(action)
            
            
            # self-defined reward
            reward = reward if not done else -10
            
            next_state = np.reshape(next_state, [1, state_size])
            
            
            # store experience
            agent.remember(state, action, reward, next_state, done)
            
            # move to next_state
            state = next_state
            steps += 1
            
            if done:
                print("episode: {}/{}, steps: {}, epsilon: {:.2}"
                      .format(e, EPISODES, steps, agent.epsilon))
                break
            
            # Train DNN-model by experience replay
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
                
        # save DNN model every 100 episode
        if e % 100 == 0:
            agent.save("./save/cartpole-dqn.h5")
      
            
            
            
            
            
            
    env.close()
        
        
        