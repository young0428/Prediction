from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import Sequential


class n_step_memory:
    def __init__(self, max_steps, gamma):
        self.max_steps = max_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=max_steps)
        self.discounted_reward = 0.0

    def add(self, experience):
        # experience는 (state, action, reward, next_state, done)의 튜플입니다.
        self.buffer.append(experience)
        self.discounted_reward = (self.discounted_reward + experience[2] * (self.gamma ** len(self.buffer)))

    def get(self):
        if len(self.buffer) < self.max_steps:
            return None
        
        # N-step 보상을 계산합니다.
        reward = self.discounted_reward
        state, action, _, _, _ = self.buffer.popleft()

        # N-step 이후의 상태와 done 플래그를 가져옵니다.
        _, _, _, next_state, done = self.buffer[-1]
        if done == True:
            return None
        

        # 다음 사용을 위해 discounted_reward를 업데이트합니다.
        self.discounted_reward = (self.discounted_reward - self.buffer[0][2]) / self.gamma

        return state, action, reward, next_state, done

    def reset(self):
        self.buffer.clear()
        self.discounted_reward = 0.0

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.01):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment = 0.001
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, error, sample):
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = sample
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough elements in the buffer to sample")

        self.beta = np.min([1., self.beta + self.beta_increment])
        total_priority = np.sum(self.priorities)
        probabilities = self.priorities / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)

        return indices, samples, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

class EEGManager:
    start_time = 0
    cur_time = 0
    cur_state = 0
    def __init__(self, game_duration):
        self.game_duration = game_duration
        self.cur_time = 0
        self.edf_file_name = None
        self.done = False
        pass
    
    def OpenEDF(self, edf_file_name):
        if not os.path.exists(edf_file_name):
            return
        
        
    
    def seperate_train_val(self):
        pass
    def initialize(self):
        pass
        

class DQNAgent:
    # INITIALIZING THE Q-PARAMETERS
    max_episodes = 200  # Set total number of episodes to train agent on.
    batch_size = 64

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.005            # Exponential decay rate for exploration prob

    scores = []
    def __init__(self, ):
        
    
        

# # Example Usage
# buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4, epsilon=0.01)
# # Add experiences to the buffer
# buffer.add(error=0.5, sample=experience)
# # Sample a batch of experiences
# indices, samples, weights = buffer.sample(batch_size=32)
# # Update priorities after learning
# buffer.update_priorities(indices, new_errors)










    
    

    
    
    
    