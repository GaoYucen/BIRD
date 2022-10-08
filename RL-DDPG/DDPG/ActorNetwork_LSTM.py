# -----------------------------
# ActorNetwork with LSTM in DDPG
# Input state and output the action
# Date: 2021.07.26
#------------------------------

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell
import config   # Import the configuration of the network

'''
To set the weights of target network
use tf.nn.rnn_cell.BasicLSTMcell instead of tf.contrib.rnn.BasicLSTMcell
'''

class ActorNetwork:
    '''A class used to construct the Actor Network of DDPG

    Attributes
    ----------
    # "__init__" setting
    sess, time_steps, state_dim, c_dim, action_range, action_dim

    # Numerical Hyperparameters of the network, can be changed in config.py
    lstm_hidden_units, pre_learning_rate, learning_rate, tau

    # Network objects defined in method "create_network"
    state_input, c_input, output, net

    # Target network objectes defined in method "create_target_network"
    target_state_input, target_c_input, target_output, target_update, lstm_cell, target_net

    # Training Configuration objects defined in method "create_training_method"
    y_exp, q_gradient_input, parameters_gradients, pre_loss_MAPE, pre_loss_MSE, pre_optimizer, optimizer

    '''
    def __init__(self, sess, time_steps, state_dim, c_dim, action_range, action_dim=1):
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
        action_range: int, positive
            the range of the action space [-action_range, +action_range], predefined using the experts' experience
        action_dim: int, optional
            the dimension of the action vector (default is 1 for regression)

        Raises
        ------
        AssertionError
            If the input action_range is negative
        '''

        self.sess = sess
        self.time_steps = time_steps
        self.state_dim = state_dim
        self.c_dim = c_dim
        self.action_dim = action_dim
        self.action_range = action_range
        assert action_range > 0, "Action_range is negative."

        self.lstm_hidden_units = config.ActorNetwork_LSTM_LSTM_HIDDEN_UNITS
        self.pre_learning_rate = config.ActorNetwork_LSTM_PRE_LEARNING_RATE
        self.learning_rate = config.ActorNetwork_LSTM_LEARNING_RATE
        self.tau = config.ActorNetwork_LSTM_TAU

        self.state_input, \
        self.c_input, \
        self.output, \
        self.net = self.create_network()

        self.target_state_input, \
        self.target_c_input, \
        self.target_output, \
        self.target_update, \
        self.lstm_cell, \
        self.target_net = self.create_target_network(self.net)

        self.create_training_method()
        self.sess.run(tf.initialize_all_variables())

    def create_training_method(self):
        '''Define the training configurations

        Attributes
        ----------
        # DDPG Configuration
        q_gradient_input: Placeholder, shape=(?, action_dim), dtype=float32
            Partial gradient of the action in the critic network, used in the policy gradient of DDPG
        parameters_gradients: list of tensor
            Policy gradient of DDPG
        optimizer: AdamOptimizer
            DDPG Actor Network optimizer

        # Pretrain Configuration
        y_exp: Placeholder, shape=(?, 1), dtype=float32
            Expert opinion, supervised learning label in the pre-train process
        pre_loss_MAPE: float
            Mean Absolute Percentage Error for pre-train
        pre_loss_MSE: float
            Mean Square Error for pre-train
        pre_optimizer: AdamOptimizer
            Pretrain optimizer defined to minimize the L2 loss

        '''
        
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(ys=self.output, xs=self.net, grad_ys=-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.net))

        self.y_exp = tf.placeholder("float", [None, 1])
        self.pre_loss_MAPE  = tf.reduce_mean(tf.abs(self.y_exp - self.output) / tf.abs(self.y_exp))
        self.pre_loss_MSE = tf.reduce_mean(tf.square(self.y_exp - self.output))
        self.pre_optimizer = tf.train.AdamOptimizer(self.pre_learning_rate).minimize(self.pre_loss_MSE)    

    def create_network(self):
        '''Define the I&O placeholders and the network

        Returns
        -------
        state_input: Placeholder, shape=(?, time_steps, state_dim), dtype=float32
            Tensorflow placeholder used to input the recent states
        c_input: Placeholder, shape=(?, c_dim), dtype=float32
            Tensorflow placeholder used to input the predictor cell state
        output: tensor, shape=(?, 1), dtype=float32
            Tensor output of the actor network, indicating the pricing strategy
        net: list
            List of variables used in the actor network including the FC layer(s) and LSTM
        
        '''
        with tf.name_scope('actor_network'): 
        # For Visualization of the network under the package Tensorboard 
            state_input = tf.placeholder("float", [None, self.time_steps, self.state_dim], name = 'state')
            c_input = tf.placeholder("float", [None, self.c_dim], name = 'c')
            state_lstm_input = tf.unstack(state_input, self.time_steps, axis=1)

            W1 = tf.Variable(tf.random_uniform(shape=[self.lstm_hidden_units + self.c_dim, 1], minval=-3e-3, maxval=3e-3), name = 'weight')
            b1 = tf.Variable(tf.random_uniform(shape=[1], minval=-3e-3, maxval=3e-3), name = 'bias')

            lstm_cell = rnn_cell.BasicLSTMCell(num_units=self.lstm_hidden_units, forget_bias=0.8, reuse = tf.AUTO_REUSE, name = 'LSTM')
            outputs, states = rnn.static_rnn(lstm_cell, state_lstm_input, dtype=tf.float32)
            output_c = tf.concat([outputs[-1], c_input], axis = 1)
            output = tf.matmul(output_c, W1) + b1
            output = tf.tanh(output) * self.action_range
            net = [W1, b1] + lstm_cell.weights

        return state_input, c_input, output, net

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
        output: tensor, shape=(?, 1), dtype=float32
            Tensor output of the target actor network, indicating the pricing strategy
        target_update: Operation
            An Operation that soft update the network parameters
        lstm_cell: rnn_cell
            LSTM cell, later used to set target parameters
        target_net: list
            List of variables used in the target actor network including the FC layer(s) and LSTM

        '''
        with tf.name_scope('target_actor_network'):
            state_input = tf.placeholder("float", [None, self.time_steps, self.state_dim], name='state')
            c_input = tf.placeholder("float", [None, self.c_dim], name='c')
            state_lstm_input = tf.unstack(state_input, self.time_steps, 1)

            # DDPG soft update model parameters
            ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)  
            target_update = ema.apply(net) 
            target_net = [ema.average(x) for x in net]

            lstm_cell = rnn_cell.BasicLSTMCell(self.lstm_hidden_units, forget_bias=0.8, reuse = tf.AUTO_REUSE, name = 'LSTM')
            outputs, states = rnn.static_rnn(lstm_cell, state_lstm_input, dtype=tf.float32)
            output_c = tf.concat([outputs[-1], c_input], axis=1)
            output = tf.matmul(output_c, target_net[0]) + target_net[1]
            output = tf.tanh(output) * self.action_range

        return state_input, c_input, output, target_update, lstm_cell, target_net

    def update_target(self):
        '''Update the ema layer used for soft update 
        '''
        self.sess.run(self.target_update)

    def pretrain(self, y_batch, state_batch, c_batch):
        '''Pretrain the actor network with expert experience

        Parameters
        ----------
        y_batch: numpy.ndarray
            Expert experience batch serves as supervised learning labels
        state_batch: numpy.ndarray
            Sale states batch, actor network pretrain input
        c_batch: numpy.ndarray
            Predictor LSTM cell state batch, actor network pretrain input

        '''
        return self.sess.run([self.pre_optimizer, self.pre_loss_MAPE, self.pre_loss_MSE], feed_dict = {
            self.y_exp: y_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def train(self, q_gradient_batch, state_batch, c_batch):
        '''Train the actor network with DDPG algorithm

        Parameters
        ----------
        q_gradient_batch: numpy.ndarray
            Critic network partial gradient with respect to the action, used in the policy gradient of DDPG
        state_batch, c_batch: numpy.ndarray
            Actor network input

        '''
        self.sess.run(self.optimizer, feed_dict = {
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def actions(self, state_batch, c_batch):
        '''Output the pricing strategy given a batch of inputs

        Parameters
        ----------
        state_batch, c_batch: numpy.ndarray
            Actor network input

        '''
        return self.sess.run(self.output, feed_dict = {
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def action(self, state, c):
        '''Output the pricing strategy given one single input

        Parameters
        ----------
        state, c: numpy.ndarray
            Actor network input
        '''
        return self.sess.run(self.output, feed_dict = {
            self.state_input: [state],
            self.c_input: [c]
        })

    def target_actions(self, state_batch, c_batch):
        '''Output the pricing strategy of the target network

        Parameters
        ----------
        state_batch, c_batch: numpy.ndarray
            Target network input
        '''
        self.lstm_cell.set_weights(self.target_net[2:])
        return self.sess.run(self.target_output, feed_dict = {
            self.target_state_input: state_batch,
            self.target_c_input: c_batch
        })

    def predict(self, y_batch, state_batch, c_batch):
        '''Pretrain performance evaluation
        
        Parameters
        ----------
        y_batch: numpy.ndarray
            Supervised learning label, experts' experience
        state_batch, c_batch: numpy.ndarray
            Supervised learning input

        '''
        return self.sess.run([self.pre_loss_MAPE, self.pre_loss_MSE, self.output], feed_dict = {
            self.y_exp: y_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def save_weights(self, path):
        '''Save actor network weights

        Parameters
        ----------
        Path: string
            Path presents where to save the parameters
        
        '''
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, path + 'actor-network')
