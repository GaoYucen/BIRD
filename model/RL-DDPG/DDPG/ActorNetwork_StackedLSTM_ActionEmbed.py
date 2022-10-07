# -----------------------------
# ActorNetwork with LSTM in DDPG
# Input state and output the action
# Date: 2021.07.26
# ------------------------------

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell
import config

'''
To set the weights of target network
use tf.nn.rnn_cell.BasicLSTMcell instead of tf.contrib.rnn.BasicLSTMcell
'''


class ActorNetwork:
    def __init__(self, sess, time_steps, state_dim, c_dim, action_dim, action_range):
        '''
        Intialization of actor network
        state - 3 x L
        action_range - positive number
        '''
        self.sess = sess
        self.time_steps = time_steps
        self.c_dim = c_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_units = config.ActorNetwork_StackedLSTM_LSTM_HIDDEN_UINITS
        self.lstm_layer_num = config.ActorNetwork_StackedLSTM_LSTM_LAYER_NUM
        self.pre_learning_rate = config.ActorNetwork_StackedLSTM_PRE_LEARNING_RATE
        self.learning_rate = config.ActorNetwork_StackedLSTM_LEARNING_RATE
        self.tau = config.ActorNetwork_StackedLSTM_TAU
        self.action_range = action_range
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
        self.y_input = tf.placeholder("float", [None, 1])
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.output, self.net, -self.q_gradient_input)
        self.pre_loss_MAPE = tf.reduce_mean(tf.abs(self.y_input - self.output) / tf.abs(self.y_input))
        self.pre_loss_MSE = tf.reduce_mean(tf.square(self.y_input - self.output))
        self.pre_optimizer = tf.train.AdamOptimizer(self.pre_learning_rate).minimize(self.pre_loss_MSE)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.parameters_gradients, self.net))

    def create_network(self):
        with tf.name_scope('actor_network'):
            state_input = tf.placeholder("float", [None, self.time_steps, self.state_dim], name='state')
            c_input = tf.placeholder("float", [None, self.c_dim], name='c')
            state_lstm_input = tf.unstack(state_input, self.time_steps, 1)

            W1 = tf.Variable(tf.random_uniform([self.lstm_hidden_units + self.c_dim, 1], -3e-3, 3e-3), name='weight')
            b1 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name='bias')

            # Stacked LSTM
            lstm_cells = [rnn_cell.BasicLSTMCell(self.lstm_hidden_units, forget_bias=0.6, reuse=tf.AUTO_REUSE) for _ in
                          range(self.lstm_layer_num)]
            mul_cells = rnn_cell.MultiRNNCell(lstm_cells)
            print(mul_cells.state_size)
            outputs, states = rnn.static_rnn(mul_cells, state_lstm_input, dtype=tf.float32)
            output_c = tf.concat([outputs[-1], c_input], axis=1)
            output = tf.matmul(output_c, W1) + b1
            output = tf.tanh(output) * self.action_range
            net = [W1, b1]
            for i in range(self.lstm_layer_num):
                net += lstm_cells[i].weights

        return state_input, c_input, output, net

    def create_target_network(self, net):
        with tf.name_scope('target_actor_network'):
            state_input = tf.placeholder("float", [None, self.time_steps, self.state_dim], name='state')
            c_input = tf.placeholder("float", [None, self.c_dim], name='c')
            state_lstm_input = tf.unstack(state_input, self.time_steps, 1)

            ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            lstm_cells = [rnn_cell.BasicLSTMCell(self.lstm_hidden_units, forget_bias=0.6, reuse=tf.AUTO_REUSE) for _ in
                          range(self.lstm_layer_num)]
            mul_cells = rnn_cell.MultiRNNCell(lstm_cells)
            outputs, states = rnn.static_rnn(mul_cells, state_lstm_input, dtype=tf.float32)
            output_c = tf.concat([outputs[-1], c_input], axis=1)
            output = tf.matmul(output_c, target_net[0]) + target_net[1]
            output = tf.tanh(output) * self.action_range

        return state_input, c_input, output, target_update, mul_cells, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def pretrain(self, y_batch, state_batch, c_batch):
        return self.sess.run([self.pre_optimizer, self.pre_loss_MAPE, self.pre_loss_MSE], feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def train(self, q_gradient_batch, state_batch, c_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def actions(self, state_batch, c_batch):
        return self.sess.run(self.output, feed_dict={
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def action(self, state, c):
        return self.sess.run(self.output, feed_dict={
            self.state_input: [state],
            self.c_input: [c]
        })

    def target_actions(self, state_batch, c_batch):
        net = self.sess.run(self.target_net, feed_dict={
            self.target_state_input: state_batch,
            self.target_c_input: c_batch
        })
        self.lstm_cell.set_weights(net[2:])
        return self.sess.run(self.target_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_c_input: c_batch
        })

    def predict(self, y_batch, state_batch, c_batch):
        return self.sess.run([self.pre_loss_MAPE, self.pre_loss_MSE, self.output], feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.c_input: c_batch
        })

    def save_weights(self, path):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, path + 'actor-network')

    def load_weights(self, path):
        self.saver = tf.train.import_meta_graph(path + 'actor-network.meta')
        # saver.restore(self.sess, tf.train.latest_checkpoint('./weights/'))
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
