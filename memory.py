import numpy as np
from collections import deque
import random


class Memory():
    def __init__(self,max_size):
        super().__init__()
        self.memory=deque(maxlen = max_size)
        
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
       return random.sample(self.memory, batch_size)