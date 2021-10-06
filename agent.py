import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from NN import DQNetwork
from Memory import Memory
from settings import DATA_DIRECTORY
from rewards import *

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

class Agent:
    def __init__(self, name, training= False, input_dim = False, gamma = False, burnin = False, epsilon = False, learning_rate = False,
                 action_dim = False, batch_size = False, eps_min = False, eps_dec = False, replace = False, memory_size = False):

        self.save_dir = DATA_DIRECTORY
        self.name = name
        self.is_training = training
        
        if self.is_training:
            self.input_dim = input_dim
            self.gamma = gamma
            self.burnin = burnin
            self.learning_rate = learning_rate
            self.action_dim = action_dim
            self.batch_size = batch_size
            self.eps_min = eps_min
            self.eps_dec = eps_dec
            self.replace = replace
            self.save_dir = DATA_DIRECTORY
            self.save_step = 0
            self.memory_size = memory_size
            self.action_space = list(range(self.action_dim))
            self.learn_step_counter = 0
            self.memory = Memory(self.memory_size, input_dim)

            self.q_online = DQNetwork(self.input_dim, self.action_dim,
                                    self.learning_rate, training=self.is_training)
            self.q_target = DQNetwork(self.input_dim, self.action_dim,
                                    self.learning_rate)

            self.epsilon = epsilon


        directory = f'{self.save_dir}/{self.name}'
        if os.path.exists(directory):
            self.load()
            if learning_rate:
                self.learning_rate = learning_rate
            if epsilon:
                self.epsilon = epsilon

        if not self.is_training:
            self.epsilon = 0



    @torch.no_grad()
    def choose_action(self, observation, output=False, force_exploit=False):
        if np.random.random() > self.epsilon or force_exploit:
            state = torch.tensor([observation], dtype=torch.float32, requires_grad=False)
            q_values = self.q_online(state).detach().numpy().flatten()
            action = np.argmax(q_values)

            if output:
                print('KI sieht {}\nund sagt {}\nq_values:{}\n'.format(observation, action, q_values))

            return action, True
        else:
            action = np.random.randint(self.action_dim)
            return action, False
            

    @torch.no_grad()
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    @torch.no_grad()
    def sample_memory(self, batch_size):
        old_states, actions, rewards, new_states, dones = self.memory.sample(batch_size)

        return old_states, actions, rewards, new_states, dones

    @torch.no_grad()
    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())
            self.q_target.zero_grad()
            for p in self.q_target.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    @torch.no_grad()
    def save(self):
        directory = f'{self.save_dir}/{self.name}'
        save_state = {
            'q_online': self.q_online.state_dict(),
            'q_target': self.q_target.state_dict(),
            'epsilon': self.epsilon,
            'save_step': self.save_step,
            'input_dim': self.input_dim,
            'gamma': self.gamma,
            'burnin': self.burnin,
            'learning_rate': self.learning_rate,
            'action_dim': self.action_dim,
        }
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}/{self.save_step}.step', 'wb') as f:
            pickle.dump(save_state, f)

        print(f'saved: {directory}/{self.save_step}.step')
        self.save_step += 1

    @torch.no_grad()
    def load(self, save_step=False, save_file=None):
        directory = f'{self.save_dir}/{self.name}'
        if type(save_step) == int:
            self.save_step = save_step
        else:
            self.save_step = 0
            for f in os.listdir(directory):
                value, ext = os.path.splitext(f)
                if ext == '.step':
                    if int(value) > self.save_step:
                        self.save_step = int(value)

        if save_file is None:
            with open(f'{directory}/{self.save_step}.step', 'rb') as f:
                save_state = pickle.load(f)
            print(f'loaded: {directory}/{self.save_step}.step')
        else:
            with open(save_file, 'rb') as f:
                save_state = pickle.load(f)

        self.save_step
        self.input_dim = save_state['input_dim']
        self.gamma = save_state['gamma']
        self.burnin = save_state['burnin']
        self.learning_rate = save_state['learning_rate']
        self.action_dim = save_state['action_dim']
        self.q_online = DQNetwork(self.input_dim, self.action_dim,
                                self.learning_rate, training=self.is_training)
        self.q_target = DQNetwork(self.input_dim, self.action_dim,
                                self.learning_rate)
        self.q_online.load_state_dict(save_state['q_online'])
        self.q_target.load_state_dict(save_state['q_target'])
        self.save_step = save_state['save_step']



    def learn(self):
        if len(self.memory) < max(self.batch_size, self.burnin):
            return None

        self.q_online.optimizer.zero_grad()
        self.replace_target_network()

        old_states, actions, rewards, new_states, dones = self.sample_memory(self.batch_size)

        indices = torch.arange(self.batch_size, dtype=torch.int64)

        q_pred = self.q_online(old_states)[indices, actions]
        q_actual = self.q_target(new_states).max(dim=1)[0]


        q_actual[dones] = 0.0
        q_actual = rewards + self.gamma*q_actual

        loss = self.q_online.loss(q_actual, q_pred)

        loss.backward()
        self.q_online.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        return loss

    def get_print_data(self):
        return (self.name, self.gamma, self.learning_rate)