from collections import deque
import random

'''
The replay buffer of DDPG
Add operation need (state, c, action, reward, next_state, next_c, done)
where state and next_state are L x 3
other variables are 1
'''

class ReplayBuffer:
    '''Replay buffer saving the previous experiences

    Attributes
    ----------
    buffer_size: int
        An interger indicating the maximum number of the experiences we can store in the buffer
    num_experiences: int
        An integer indicating the number of experiences in the current buffer
    buffer: collections.deque
        The double-ended queue convenient for storage and sample of the experiences

    '''
    def __init__(self, buffer_size):
        '''Initialization

        Parameters
        ----------
        buffer_size: int
            An interger indicating the maximum number of the experiences we can store in the buffer
        '''
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque() # Double-ended queue

    def get_batch(self, batch_size):
        '''Sample a batch from the replay buffer
        '''
        return random.sample(self.buffer, batch_size)

    def size(self):
        '''Return the max size of the replay buffer
        '''
        return self.buffer_size

    def add(self, state, c, action, reward, next_state, next_c, done):
        '''Add new sample into buffer while keeping the number of samples under buffer_size

        state, c, action, reward, next_state, next_c, done: an experience
        '''
        experience = (state, c, action, reward, next_state, next_c, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        '''Return the number of experiences in the current replay buffer
        '''
        return self.num_experiences

    def erase(self):
        '''Reinitialize the replay buffer
        '''
        self.buffer = deque()
        self.num_experiences = 0