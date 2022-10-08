# -----------------------------
# CriticNetwork with LSTM in DDPG
# Input state and action, output the q-value
# Date: 2021.08.04
#------------------------------

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell
import config

class CriticNetwork:
    '''A class used to construct the Critic Network of DDPG

    Attributes
    ----------
    # "__init__" setting
    sess, time_steps, state_dim, c_dim, action_range, action_dim

    # Numerical Hyperparameters of the network, can be changed in config.py
    lstm_hidden_units, learning_rate, tau, l2

    # Network objects defined in method "create_network"
    state_input, c_input, action_input, output, net

    # Target network objectes defined in method "create_target_network"
    target_state_input, target_c_input, target_action_input, target_output, target_update, lstm_cell, target_net

    # Training objects defined in method "create_training_method"
    r_input, cost, optimizer, action_gradients
    '''
    def __init__(self, sess, time_steps, state_dim, c_dim, action_dim=1):
        '''Intialization of actor network

        Parameters
        ----------
        sess: Tensorflow Interactive Session
            a tensorflow seesion used for the execution of the graph(or subgraph)
        time_steps: int
            an integer indicating the time scale of the input
        state_dim: int
            the dimension of the state, currently defined as 3 (price, util, remaining_time)
        c_dim: int
            the dimension of the cell state provided by the prediction part
        action_dim: int, optional
            the dimension of the action vector (default is 1 for regression)
        '''

        self.sess = sess
        self.time_steps = time_steps
        self.state_dim = state_dim
        self.c_dim = c_dim
        self.action_dim = action_dim
        self.lstm_hidden_units = config.CriticNetwork_LSTM_LSTM_HIDDEN_UINITS
        self.learning_rate = config.CriticNetwork_LSTM_LEARNING_RATE
        self.tau = config.CriticNetwork_LSTM_TAU
        self.l2 = config.CriticNetwork_LSTM_L2

        self.state_input, \
        self.c_input, \
        self.action_input, \
        self.output, \
        self.net = self.create_network()

        self.target_state_input, \
        self.target_c_input, \
        self.target_action_input, \
        self.target_output, \
        self.target_update, \
        self.lstm_cell, \
        self.target_net = self.create_target_network(self.net)

        self.create_training_method()
        self.sess.run(tf.initialize_all_variables())
        self.graph = sess.graph

    def create_training_method(self):
        self.y_input = tf.placeholder('float', [None, 1])
        weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])
        #self.cost = tf.reduce_mean(tf.square(self.y_input - self.output)) + weight_decay
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.output))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.action_gradients = tf.gradients(self.output, self.action_input)

    def create_network(self):
        '''Define the I&O placeholders and the network

        Returns
        -------
        state_input: Placeholder, shape=(?, time_steps, state_dim), dtype=float32
            Tensorflow placeholder used to input the recent states
        c_input: Placeholder, shape=(?, c_dim), dtype=float32
            Tensorflow placeholder used to input the predictor cell state
        action_input: Placeholder, shape=(?, action_dim), dtype=float32
            Tensorflow Placeholder used to input the action provided by the actor network
        output: tensor, shape=(?, 1), dtype=float32
            Tensor output of the actor network, indicating the pricing strategy
        net: list
            List of variables used in the actor network including the FC layer(s) and LSTM
        
        '''
        with tf.name_scope('critic_network'):
            state_input = tf.placeholder('float', [None, self.time_steps, self.state_dim], name = 'state')
            c_input = tf.placeholder('float', [None, self.c_dim], name = 'c')
            action_input = tf.placeholder('float', [None, self.action_dim], name = 'action')
            state_lstm_input = tf.unstack(state_input, self.time_steps, 1)

            W1 = tf.Variable(tf.random_uniform([self.lstm_hidden_units + self.action_dim + self.c_dim, 1], -3e-3, 3e-3),name = 'weight')
            b1 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name = 'bias')

            lstm_cell = rnn_cell.BasicLSTMCell(self.lstm_hidden_units, forget_bias=0.8, reuse=tf.AUTO_REUSE, name = 'LSTM')
            outputs, states = rnn.static_rnn(lstm_cell, state_lstm_input, dtype=tf.float32)
            output_action_c = tf.concat([outputs[-1], action_input, c_input], axis=1)
            output = tf.matmul(output_action_c, W1) + b1
            net = [W1, b1] + lstm_cell.weights

        return state_input, c_input, action_input, output, net

    def create_target_network(self, net):
        '''Create Target Network used in DDPG algorithm

        Parameter
        ---------
        net: list
            a list of parameters to initialize the target network

        Returns
        -------
        state_input: Placeholder, shape=(?, time_steps, state_dim), dtype=float32
            Tensorflow placeholder used to input the recent states for target network
        c_input: Placeholder, shape=(?, c_dim), dtype=float32
            Tensorflow placeholder used to input the predictor cell state for target network
        action_input: Placeholder, shape=(?, action_dim), dtype=float32
            Tensorflow placeholder used to input the action decided by the actor network
        q_value_output: tensor, shape=(?, 1), dtype=float32
            Output tensor of the q-value
        target_update: Operation
            An Operation that soft update the network parameters
        lstm_cell: rnn_cell
            LSTM cell, later used to set target parameters
        target_net: list
            List of variables used in the target actor network including the FC layer(s) and LSTM
        '''
        with tf.name_scope('target_critic_network'):
            state_input = tf.placeholder('float', [None, self.time_steps, self.state_dim], name='state')
            c_input = tf.placeholder('float', [None, self.c_dim], name='c')
            action_input = tf.placeholder('float', [None, self.action_dim], name='action')
            state_lstm_input = tf.unstack(state_input, self.time_steps, 1)

            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # 指数加权的平均法
                target_update = ema.apply(net)
                target_net = [ema.average(x) for x in net]

            lstm_cell = rnn_cell.BasicLSTMCell(self.lstm_hidden_units, forget_bias=0.8, reuse=tf.AUTO_REUSE, name='LSTM')
            outputs, states = rnn.static_rnn(lstm_cell, state_lstm_input, dtype = tf.float32)
            output_action_c = tf.concat([outputs[-1], action_input, c_input], axis = 1)
            q_value_output = tf.matmul(output_action_c, target_net[0]) + target_net[1]

            return state_input, c_input, action_input, q_value_output, target_update, lstm_cell, target_net

    def create_training_method(self):
        '''Define the gradients and the optimizer

        Attributes
        ----------
        r_inputs: Placeholder, shape=(?, 1), dtype=float32
            Placeholder used to get expected total reward
        cost: tensor, shape=(1), dtype=float32
            Square difference of the real and predicted future accumulated reward plus L2 regularization
        optimizer: AdamOptimizer
            Optimizer aiming at minimizing the pre-defined cost
        action_gradients: list of tensor
            Partial gradient of output value(predicted accumulated reward) with respect to the action
        '''
        self.r_input = tf.placeholder('float', [None, 1])
        weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.r_input - self.output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.action_gradients = tf.gradients(self.output, self.action_input)

    def update_target(self):
        '''Update the ema layer used for soft update 
        '''
        self.sess.run(self.target_update)

    def train(self, r_batch, state_batch, c_batch, action_batch):
        '''Train the critic network with DDPG algorithm

        Parameters
        ----------
        r_batch: numpy.ndarray
            Current reward plus futur predicted reward, by which the critic network is supervised      
        state_batch, c_batch, action_batch: numpy.ndarray
            Critic network input
        '''
        self.sess.run(self.optimizer, feed_dict = {
            self.r_input: r_batch,
            self.state_input: state_batch,
            self.c_input: c_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, c_batch, action_batch):
        '''Calculate the partial gradient of the q-value with respect to the action

        Parameters
        ----------
        state_batch, c_batch, action_batch: numpy.ndarray
            Critic network input
        '''
        return self.sess.run(self.action_gradients, feed_dict = {
            self.state_input: state_batch,
            self.c_input: c_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, c_batch, action_batch):
        '''Update the parameters in the target network and return the estimated q-value given by the target network.

        Parameters
        ----------
        state_batch, c_batch, action_batch: numpy.ndarray
            Target Critic network input

        Return
        ------
        self.target_output: tensor, shape=(?, 1), dtype=float32
            Output tensor of the target q-value

        '''
        net = self.sess.run(self.target_net, feed_dict={
            self.target_state_input: state_batch,
            self.target_c_input: c_batch,
            self.target_action_input: action_batch
        })
        self.lstm_cell.set_weights(net[2:])
        return self.sess.run(self.target_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_c_input: c_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, c_batch, action_batch):
        '''Update the parameters in the target network and return the estimated q-value given by the target network.

        Parameters
        ----------
        state_batch, c_batch, action_batch: numpy.ndarray
            Critic network input

        Return
        ------
        self.output: tensor, shape=(?, 1), dtype=float32
            Output tensor of the q-value
        '''
        return self.sess.run(self.output, feed_dict={
            self.state_input: state_batch,
            self.c_input: c_batch,
            self.action_input: action_batch
        })

    # TODO: load_network
    # TODO: save_network