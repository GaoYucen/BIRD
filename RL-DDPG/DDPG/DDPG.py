# -----------------------------
# Deep Deterministic Policy Gradient
# Including ActorNetwork and CriticNetwork
# Date: 2021.07.26
#------------------------------

import tensorflow as tf
import numpy as np
from ActorNetwork_StackedLSTM import ActorNetwork
from CriticNetwork_LSTM_ActionEmbed import CriticNetwork
from replay_buffer import ReplayBuffer
import config
import pickle

class DDPG:
    '''Class used to implement the DDPG algorithm

    Attributes
    ----------
    # "__init__" setting
    name, state_dim, time_steps,c_dim, action_dim, action_range, sess

    # Numerical Hyperparameters of the network, can be changed in config.py
    replay_buffer_size, replay_start,size, batch_size

    # Three DDPG objects created
    actor_network, critic_network, replay_buffer

    '''
    def __init__(self, state_dim, time_steps, c_dim, action_dim, action_range):
        """DDPG Initialization

        Attributes
        ----------
        name: string
            name of the algorithm
        state_dim: int
            the dimension of the state, currently defined as 3 (price, util, remaining_time)
        time_steps: int
            an integer indicating the time scale of the input
        c_dim: int
            the dimension of the cell state provided by the prediction part
        action_dim: int
            the dimension of the action vector
        action_range: int, positive
            the range of the action space [-action_range, +action_range], predefined using the experts' experience
        sess: InteractiveSession
            Tensorflow Default Session used to execute the graph

        Raises
        ------
        AssertionError
            If the input action_range is negative

        """
        self.name = 'DDPG'
        self.state_dim = state_dim
        self.time_steps = time_steps
        self.c_dim = c_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.replay_buffer_size = config.DDPG_REPLAY_BUFFER_SIZE
        self.replay_start_size = config.DDPG_REPLAY_START_SIZE
        self.batch_size = config.DDPG_BATCH_SIZE
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.time_steps, self.state_dim, self.c_dim, self.action_dim, self.action_range)
        self.critic_network = CriticNetwork(self.sess, self.time_steps, self.state_dim, self.c_dim, self.action_dim)

        write = tf.summary.FileWriter('./total_name_scope', graph=self.sess.graph)
        write.close()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.sess.run(tf.initialize_all_variables())


    def train(self):
        """ There are three steps executed sequentially in this function.
        1. Sample and process the experiences from replay buffer
        2. Train both critic network and actor network
        3. Update target network
        """

        # Sample and Distribute data from Replay Buffer
        # t: (s,c,a,r) --> t+1: (s,c,done,a,q)
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        c_batch = np.asarray([data[1] for data in minibatch])
        action_batch = np.asarray([data[2] for data in minibatch])
        reward_batch = np.asarray([data[3] for data in minibatch])

        next_state_batch = np.asarray([data[4] for data in minibatch])
        next_c_batch = np.asarray([data[5] for data in minibatch])
        done_batch = np.asarray([data[6] for data in minibatch])
        
        action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

        next_action_batch = self.actor_network.target_actions(next_state_batch, next_c_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_c_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + config.DDPG_GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [config.DDPG_BATCH_SIZE, 1])

        # Train both critic network and actor network
        self.critic_network.train(y_batch, state_batch, c_batch, action_batch)

        action_batch_for_gradients = self.actor_network.actions(state_batch, c_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, c_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch, c_batch)

        # Update target network
        self.actor_network.update_target()
        self.critic_network.update_target()

    def action(self, state, c):
        '''Return action given by the actor network
        '''
        return self.actor_network.action(state, c)

    def perceive(self, state, c, action, reward, next_state, next_c, done):
        '''
        Add a new experience tuple into the replay buffer.
        Train the network if the number of experiences in the replay buffer reaches the start size.
        '''
        self.replay_buffer.add(state, c, action, reward, next_state, next_c, done)

        if self.replay_buffer.count() > config.DDPG_REPLAY_START_SIZE:
            self.train()

    def load_pretrained_weights(self, path):
        '''Load pretrained weights for actor network
        '''
        self.actor_network.load_weights(path)

    def load_weights(self, path):
        '''Load pretrained weights for actor network
        '''
        self.actor_network.load_weights(path)
        self.critic_network.load_weights(path)
        file = open(path + 'replay_buffer.pkl', 'rb')
        self.replay_buffer = pickle.load(file)

    def save_weights(self, path):
        self.actor_network.save_weights(path)
        self.critic_network.save_weights(path)
        file = open(path + 'replay_buffer.pkl', 'wb')
        pickle.dump(self.replay_buffer, file)

