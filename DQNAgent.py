#%%
import logging
import os
logging.getLogger('cupy').setLevel(logging.ERROR)
os.environ['SSQ_PARALLEL'] = '1'
os.environ['SSQ_GPU'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cupy.cuda.compiler')
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

import random
import math
import time
from PreProcessing import SegmentsSSQCWT
from collections import deque
from readDataset import *
from CHB_processing.InfoProcessing import get_chb_interval_info
from DQNModel import Build_DQN_model

# %%



class n_step_memory:
    def __init__(self, max_steps, gamma):
        self.max_steps = max_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=max_steps)
        self.discounted_reward = 0.0
    def add(self, experience):
        # experience는 (state, action, reward, next_state, done)의 튜플
        self.buffer.append(experience)
        if len(self.buffer) >= self.max_steps:
            return True
        return False
    def get(self):
        if len(self.buffer) < self.max_steps:
            return None
        # N-step 보상을 계산.
        reward = 0
        for i in range(len(self.buffer)):
            reward += self.buffer[i][2] * (self.gamma ** i)
        state, action, _, _, _ = self.buffer[0]

        # N-step 이후의 상태와 done 플래그를 가져옴
        _, _, _, next_state, done = self.buffer[-1]

        return state, action, reward, next_state, done

    def reset(self):
        self.buffer.clear()
        #self.discounted_reward = 0.0

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.01):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment = 0.001
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.min_prior = 0.0
        self.max_prior = 1.0
        
        
    def add(self, sample):
        priority = self.max_prior
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
        probabilities = np.array(self.priorities) / total_priority
        indices = np.random.choice(list(range(len(self.buffer))), batch_size, p=probabilities)
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
    cur_cwt = 0
    def __init__(self, window_size, channels, data_type, game_duration = 30*60, step_size = 20):
        
        self.game_duration = game_duration
        self.cur_time = 0
        self.edf_file_name = None
        self.done = False
        self.channels = channels
        self.data_type = data_type
        self.window_size = window_size
        self.total_duration = 0
        self.cur_game_duration = 0
        self.step_size = step_size
        
        chb_info = get_chb_interval_info()
        self.total_time_flag = chb_info[0]
        self.patient_enable_interval_info = chb_info[1]
        self.patient_total_validation_duration = chb_info[2]
        self.enable_period_flag = chb_info[3]
        self.patient_names = list(self.total_time_flag.keys())
        self.designate_patient_name = None
        
        
        for name in self.patient_names:
            self.total_duration += self.patient_total_validation_duration[name]
        
        self.patient_prob = [ self.patient_total_validation_duration[name] / self.total_duration for name in self.patient_names]

    def select_patient(self):
        if self.designate_patient_name == None:
            self.selected_patient_name = np.random.choice(self.patient_names, 1, p = self.patient_prob)[0]
        else : self.selected_patient_name = self.designate_patient_name
        self.selected_patient_flag = self.total_time_flag[self.selected_patient_name]
            
    def set_patient(self, name):
        self.designate_patient_name = name
        
    def find_enable_adjacency_pos(self, intervals, cur_pos):
        for i in range(len(intervals[1])-1):
            if cur_pos > intervals[1][i][1] and cur_pos <= intervals[1][i+1][0] and intervals[1][i+1][1] - intervals[1][i+1][0] >= self.window_size:
                cur_pos = intervals[1][i+1][0] + self.window_size
                return cur_pos
        # fail to find adjacency position
        return -1
    def select_initial_start_position(self, interval):
        gap = self.window_size + self.game_duration
        try_cnt = 0
        while True:
            
            pos = random.randint(interval[0][0] + self.window_size, interval[0][1] - gap * 2)
            if self.enable_period_flag[self.selected_patient_name][pos] == 0:
                pos = self.find_enable_adjacency_pos(interval, pos)
            if pos > interval[0][1] - gap or pos == -1:
                try_cnt += 1
                if try_cnt > 10:
                    print("unable to select initial start position")
                    return -1
                continue    
            return pos
        
    def find_file_by_pos(self, pos):
        
        for interval in self.choiced_interval[1]:
            if pos >= interval[0] and pos <= interval[1] :
                return interval
        return -1
    
    def cal_ictal_distance(self, pos):
        cnt = 0
        distance = -1
        while pos + cnt < len(self.selected_patient_flag):
            if self.selected_patient_flag[pos+cnt] == 1:
                distance = cnt
                break
            cnt += 1
                
        return distance
    def init_game(self):
        loop_cnt = 0
        while True:
            self.select_patient()
            if self.patient_total_validation_duration[self.selected_patient_name] < self.game_duration:
                continue
            enable_interval = []
            enabled_interval_total_length = 0 
            for interval in self.patient_enable_interval_info[self.selected_patient_name]:
                interval_length = interval[0][1] - interval[0][0]
                if interval_length > (self.game_duration + self.window_size)*2:
                    enable_interval.append(interval)
                    enabled_interval_total_length += interval_length
                    
            # escape the loop when fail to select interval 5 times
            if len(enable_interval) == 0:
                loop_cnt += 1
                if loop_cnt >= 5:
                    print("unable to select interval!")
                    break
                continue
            
            interval_prob = [ (interval[0][1] - interval[0][0])/enabled_interval_total_length for interval in enable_interval]
            self.choiced_interval = enable_interval[np.random.choice(list(range(len(enable_interval))), 1, p = interval_prob)[0]]
            start_time = self.select_initial_start_position(self.choiced_interval)
            if start_time == -1 :
                loop_cnt += 1
                if loop_cnt >= 5:
                    print("unable to select interval!")
                    break
                continue
            
            self.cur_pos = start_time
            self.cur_cwt = self.pos2cwt(self.cur_pos)
            if self.cur_cwt is -1:
                print("fail to read data from edf file")
                continue
            self.alarm_state = 0
            self.cur_game_duration = 0
            self.done = False
            loop_cnt = 0
            break
            
        return self.cur_pos, (self.cur_cwt, self.alarm_state)
    def pos2cwt(self, pos):
        interval = self.find_file_by_pos(pos)

        file_name = interval[2]
        pos_in_file = pos - interval[0]
        self.segment = [[file_name, pos_in_file-self.window_size, self.window_size]]
        eeg_data = Segments2Data(self.segment, self.data_type, self.channels)
        if eeg_data is None:
            print("fail to read data from edf file")
            return -1
        # [1, channel_num, self.window_size * sampling_rate]       
        eeg_data.shape = (1, eeg_data.shape[2])
        state = SegmentsSSQCWT(eeg_data, sampling_rate=200, scale_resolution=128)
        state = np.expand_dims(np.squeeze(state), axis = -1)
        return state
    
    def cal_reward(self, cur_pos, alarm_state, action):
        reward = 0
        if action == 1:
            reward -= 1
        if (3 in self.selected_patient_flag[cur_pos - self.window_size : cur_pos]) or (3 in self.selected_patient_flag[cur_pos - self.window_size : cur_pos]):
            if action == 1 and alarm_state == 0:
                reward = 10
        return reward
    
    def step(self, action):
        done = False
        reward = self.cal_reward(self.cur_pos, self.alarm_state, action)
        if action == 1:
            self.alarm_state = 1
            
        self.cur_game_duration += self.step_size
        next_pos = self.cur_pos + self.step_size
        if 0 in self.enable_period_flag[self.selected_patient_name][next_pos-self.window_size:next_pos]:
            next_pos = self.find_enable_adjacency_pos(self.choiced_interval, next_pos)
        # terminated ?            
        if next_pos == -1 or self.cur_game_duration >= self.game_duration:
            done = True
            next_pos = self.cur_pos
            next_state = (self.cur_cwt, self.alarm_state)
        else:    
            next_cwt = self.pos2cwt(next_pos)
            if next_cwt is -1:
                print("fail to load edf file!")
                return -1,-1,-1,-1
            next_state = (next_cwt, self.alarm_state)
            
        self.cur_pos = next_pos
        self.cur_cwt = next_state[0]

        return next_pos, next_state, reward, done
        

class RainbowDQNAgent:
    # INITIALIZING THE Q-PARAMETERS
    max_episodes = 200  # Set total number of episodes to train agent on.
    
    action_num = 2
    Vmin = -10
    Vmax = 10
    num_atoms = 51
    
    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.005            # Exponential decay rate for exploration prob
    gamma = 0.99

    scores = []
    def __init__(self,  window_size, channels, data_type, batch_size , step_size = 20):
        
        self.window_size = window_size
        self.step_size = step_size
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
    # GPU 디바이스가 있는 경우, 첫 번째 GPU 디바이스 선택
            gpu_device = gpus[0]
            # GPU 디바이스를 문자열로 표시
            self.device = f"/device:{gpu_device.device_type}:0"
            
        self.batch_size = batch_size
        self.TAU = 0.1
        self.lr = 0.001
        self.soft_update = False
        self.z = np.linspace(self.Vmin, self.Vmax, self.num_atoms)
        
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        self.n_step = 5
        self.n_step_memory = n_step_memory(self.n_step, self.gamma)
        self.designate_patient = None
        
        cwt_inputs = tf.keras.layers.Input(shape=(128,int(200*self.window_size), 1))
        alarm_state_inputs = tf.keras.layers.Input(shape=(1))
        
        self.train_start = 1000
        
        self.optimizers = tf.keras.optimizers.Adam(learning_rate=self.lr, )
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        
        self.dqn = Build_DQN_model(cwt_inputs, alarm_state_inputs, num_atom=self.num_atoms)
        self.dqn_target = Build_DQN_model(cwt_inputs, alarm_state_inputs, num_atom=self.num_atoms)
        self.target_hard_update()

        self.game_manager = EEGManager(window_size, channels, data_type, step_size=step_size)
        
        
    def get_action(self, state):
        cwt_image, alarm = state
        cwt_image = np.expand_dims(cwt_image, axis=-1)
        state = (cwt_image, alarm)
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_num)
        else:
            action = self.get_optimal_aciton(state)
            
        return action
    
    def set_patient(self, patient_name):
        self.game_manager.set_patient(patient_name)
        
    def get_optimal_aciton(self, state):
        cwt_image, alarm_state = state
        cwt_image = np.array([np.expand_dims(np.squeeze(cwt_image), axis=-1)])
        alarm_state = np.array([[alarm_state]])
        z = self.dqn.predict_on_batch([cwt_image, alarm_state])
        q = tf.reduce_sum(tf.multiply(z, self.z),axis=-1)
        return np.argmax(q)
    
    def init_game(self):
        cur_pos, cur_state = self.game_manager.init_game()
        self.n_step_memory.reset()
        return cur_pos, cur_state

    def target_hard_update(self):
        if not self.soft_update:
            weights = self.dqn.get_weights()
            self.dqn_target.set_weights(weights)
            return
        if self.soft_update:
            q_model_theta = self.dqn.model.get_weights()
            dqn_target_theta = self.dqn_target.model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, dqn_target_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                dqn_target_theta[counter] = target_weight
                counter += 1
            self.dqn_target.set_weights(dqn_target_theta)
            
    def step(self, action):
        
        cur_pos = self.game_manager.cur_pos
        cur_state = (self.game_manager.cur_cwt, self.game_manager.alarm_state)
        next_pos, next_state, reward, done = self.game_manager.step(action)
        if done == -1:
            return None,None,None,None
        n_step_memory_full = self.n_step_memory.add((cur_state, action, reward, next_state, done))
        
        # if n_step_memory is filled
        if n_step_memory_full:
            sample = self.n_step_memory.get()
            if not sample is None:
                self.memory.add(sample)
        
        return next_pos, next_state, reward, done

    def update_model(self):
        indices, samples, is_weights = self.memory.sample(self.batch_size)
        
        elementalwisw_loss, loss = self.compute_dqn_loss(samples, is_weights)
        self.memory.update_priorities(indices, elementalwisw_loss)
        
        return loss
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)
    
    def train(self):
        pass

    def compute_dqn_loss(self, samples, is_weights):
        with tf.device(self.device):
            states, actions, rewards, next_states, dones = zip(*samples)
            dones = np.array(dones)
            delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)

            
            # Get current action probabilities
            cwt_inputs, alarm_states = zip(*states)
            cwt_inputs = np.array(cwt_inputs)
            alarm_states = np.array([[alarm_state] for alarm_state in alarm_states])
            
            next_cwt_inputs, next_alarm_states = zip(*next_states)
            next_cwt_inputs = np.array(next_cwt_inputs)
            next_alarm_states = np.array([[alarm_state] for alarm_state in next_alarm_states])
 
            

            z = self.dqn.predict_on_batch([next_cwt_inputs, next_alarm_states]) 
            q = tf.reduce_sum(tf.multiply(z, self.z),axis=-1)
            next_action = tf.argmax(q,axis=1)
            next_dist = np.array(self.dqn_target.predict_on_batch([next_cwt_inputs, next_alarm_states]))
            
            next_dist = next_dist[tf.range(self.batch_size), next_action]

            # Project distribution onto support
            proj_dist_batches = []
            for i in range(self.batch_size):
                t_z = rewards[i] + (1 - dones[i]) * self.gamma * self.z
                t_z = tf.clip_by_value(t_z, self.Vmin, self.Vmax)
                b = (t_z - self.Vmin) / delta_z
                l = np.floor(b)
                u = np.ceil(b)

                #offset = i * self.num_atoms
                #offset = tf.expand_dims(offset, axis=-1)
                
                l = tf.expand_dims(l, axis=1)
                u = tf.expand_dims(u, axis=1)
                b = tf.expand_dims(b, axis=1)

                proj_dist_batch = np.expand_dims(tf.zeros_like(next_dist[i]),axis=-1)
                #test = tf.tensor_scatter_nd_add(proj_dist_batch, np.tf.cast(l, tf.int32), tf.cast((next_dist[i] * (u - b)), dtype=tf.float32))
                proj_dist_batch += tf.tensor_scatter_nd_add(proj_dist_batch, tf.cast(l, tf.int32), np.expand_dims(next_dist[i],axis=-1) * (u - b))
                proj_dist_batch += tf.tensor_scatter_nd_add(proj_dist_batch, tf.cast(u, tf.int32), np.expand_dims(next_dist[i],axis=-1) * (b - l))
                proj_dist_batches.append(proj_dist_batch)
            proj_dist = tf.convert_to_tensor(np.squeeze(tf.stack(proj_dist_batches)), dtype=tf.float32)
            
            # Calculate element-wise loss
            with tf.GradientTape() as tape:
                dist = self.dqn([cwt_inputs, alarm_states])
                q = tf.reduce_sum(tf.multiply(dist, self.z),axis=-1)
                act = tf.argmax(q,axis=1)
                # # 인덱싱을 위한 인덱스 배열 생성
                actions_taken = tf.cast(tf.gather(act, tf.range(self.batch_size)), dtype=tf.int32)
                selected_values = tf.gather_nd(dist, tf.stack([tf.range(self.batch_size), actions_taken], axis=1))
                log_p = tf.cast(tf.math.log(selected_values),dtype=tf.float32)
                
                elementwise_loss = - tf.reduce_sum(proj_dist * log_p, axis=1)
                loss = tf.reduce_mean(elementwise_loss * is_weights)
                # # main_value = self.dqn([cwt_inputs, alarm_states])

                # loss = self.criterion(proj_dist, selected_values)

            # Calculate gradients and apply optimizer updates
            gradients = tape.gradient(loss, self.dqn.trainable_variables)
            self.optimizers.apply_gradients(zip(gradients, self.dqn.trainable_variables))

        return elementwise_loss, loss
#%%
if __name__ == '__main__':
    agent = RainbowDQNAgent(window_size=20, channels = ['FP1-F7'], data_type = 'chb_one_ch',batch_size=16, step_size=20)
    
    #%%
    validate_patient = select_validate_patient(data_type='chb_one_ch')
    validate_patient.remove('CHB017')
    #%%
    for patient_name in validate_patient:
        agent.set_patient(patient_name)
        step_cnt = 0
        for episode in range(100):
            cur_pos, cur_state = agent.init_game()
            done = False
            loss_sum = 0
            cnt = 0
            total_reward = 0
            while not done :
                action = agent.get_action(cur_state)
                next_pos, next_state, reward, done = agent.step(action)
                total_reward += reward
                if next_pos == None:
                    break
                step_cnt += 1
                if step_cnt > 200:
                    loss = agent.update_model()
                    loss_sum += loss
                    cnt+=1
                    if step_cnt % 10 == 0 :
                        agent.decay_epsilon()
                        agent.target_hard_update()
            if cnt > 0:
                avg_loss = (loss_sum/cnt)
            else:
                avg_loss = 0
            print(f'episode {episode} done, step_cnt : {step_cnt}, avg_loss : ' + '%.4f'%(avg_loss)+f'  reward : {total_reward}')
            loss_sum = 0
            cnt = 0
            total_reward = 0
                
                
                
                
                
        

# %%
