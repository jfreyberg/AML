import numpy as np
import torch

class Memory():
    @torch.no_grad()
    def __init__(self, max_size, input_dim):
        self.size = max_size
        self.index = 0
        self.length = 0
        self.input_dim = input_dim

        self.old_state_memory = torch.zeros((self.size, self.input_dim), dtype=torch.float32, requires_grad=False)
        self.new_state_memory = torch.zeros((self.size, self.input_dim), dtype=torch.float32, requires_grad=False)

        self.action_memory = torch.zeros((self.size), dtype=torch.int64, requires_grad=False)
        self.reward_memory = torch.zeros(self.size, dtype=torch.float32, requires_grad=False)
        self.done_memory = torch.zeros(self.size, dtype=torch.bool, requires_grad=False)

    @torch.no_grad()
    def __len__(self):
        return self.length

    @torch.no_grad()
    def store(self, old_state, action, reward, new_state, done):

        self.index = self.index % self.size

        self.old_state_memory[self.index] = torch.tensor(old_state, dtype=torch.float32, requires_grad=False)
        self.action_memory[self.index] = torch.tensor(action, dtype=torch.int64, requires_grad=False)
        self.reward_memory[self.index] = torch.tensor(reward, dtype=torch.float32, requires_grad=False)
        self.done_memory[self.index] = torch.tensor(done, dtype=torch.bool, requires_grad=False)
        self.new_state_memory[self.index] = torch.tensor(new_state, dtype=torch.float32, requires_grad=False)

        self.index += 1
        if self.length != self.size:
            self.length = max(self.index, self.length)

    @torch.no_grad()
    def sample(self, batch_size):
        selection = np.random.choice(self.length, batch_size, replace=False)

        old_states = self.old_state_memory[selection]
        new_states = self.new_state_memory[selection]
        actions = self.action_memory[selection]
        rewards = self.reward_memory[selection]
        dones = self.done_memory[selection]

        return old_states, actions, rewards, new_states, dones
        
        
        
        
        
        
        
        
        
        
        
