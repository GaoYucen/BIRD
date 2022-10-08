import tensorflow as tf
import numpy as np
import csv
import random
from ActorNetwork_StackedLSTM import ActorNetwork
import config

'''
P-price
Util-utilization
RT-remaining time
input: L x 3 ([P1, Util1, RT1], [P2, Util2, RT2], ..., [PL, UtilL, RTL]), c
output: action - [-C%, C%]
'''

def read_data(path):
    return np.load(path)

# split train and test data
def split_train(state_batch, c_batch, y_batch,test_ratio):
    shuffled_indices=np.random.permutation(len(state_batch))
    test_set_size=int(len(state_batch)*test_ratio)
    state_test = state_batch[shuffled_indices[:test_set_size]]
    state_train = state_batch[shuffled_indices[test_set_size:]]
    c_test = c_batch[shuffled_indices[:test_set_size]]
    c_train = c_batch[shuffled_indices[test_set_size:]]
    y_test = y_batch[shuffled_indices[:test_set_size]]
    y_train = y_batch[shuffled_indices[test_set_size:]]
    return state_test, state_train, c_test, c_train, y_test, y_train

# get batch from data
def get_batch(state, c, y):
    shuffled_indices=np.random.permutation(len(state))
    state_batch = state[shuffled_indices[:config.PRE_BATCH_SIZE]]
    c_batch = c[shuffled_indices[:config.PRE_BATCH_SIZE]]
    y_batch = y[shuffled_indices[:config.PRE_BATCH_SIZE]]

    return state_batch, c_batch, y_batch

if __name__ == "__main__":
    random.seed(10)
    state = read_data(config.X_DATA_PATH)
    price = np.zeros((state.shape[0], state.shape[1], 1))
    state = np.concatenate((state, price), axis=2)
    y = read_data(config.Y_DATA_PATH)
    c = np.zeros((y.shape[0], 128))

    train_csvfile = open('../data/train_MAPE_MSE.csv', 'w')
    test_csvfile = open('../data/test_MAPE_MSE.csv', 'w')
    train_writer = csv.writer(train_csvfile)
    test_writer = csv.writer(test_csvfile)
    train_writer.writerow(['epoch', 'MAPE', 'MSE'])
    test_writer.writerow(['epoch', 'MAPE', 'MSE'])
    #print(max(y), min(y))


    state_test, state_train, c_test, c_train, y_test, y_train = split_train(state, c, y, 0.2)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.InteractiveSession()
    actor_network = ActorNetwork(sess, 15, 3, 128, 1, 66)

    # TensorBoard Output
    # write = tf.summary.FileWriter('./name_scope', graph = sess.graph)
    # write.close()

    epoch_list = []
    MAPE_list = []
    MSE_list = []

    test_epoch_list = []
    test_MAPE_list = []
    test_MSE_list = []

    for epoch in range(config.PRE_EPOCH):
        state_batch, c_batch, y_batch = get_batch(state_train, c_train, y_train)
        _, train_MAPE, train_MSE = actor_network.pretrain(y_batch.reshape((y_batch.shape[0],1)), state_batch, c_batch) # train a batch
        epoch_list.append(epoch)
        MAPE_list.append(train_MAPE)
        MSE_list.append(train_MSE)
        train_writer.writerow([epoch, train_MAPE, train_MSE])

        if epoch % 1000 == 0:
            print('Epoch', epoch, 'Training MAPE = ', train_MAPE, 'MSE = ', train_MSE)
            loss = actor_network.predict(y_test.reshape((y_test.shape[0], 1)), state_test, c_test)
            print('Epoch ', epoch, 'MAPE = ', loss[0], 'MSE = ', loss[1], "output = ", loss[2])
            test_writer.writerow([epoch, loss[0], loss[1]])
            #test_epoch_list.append(epoch)
            test_MAPE_list.append(loss[0])
            test_MSE_list.append(loss[1])

    actor_network.save_weights(config.PRE_WEIGHT_PATH)

    train_csvfile.close()
    test_csvfile.close()

    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    # ax1.plot(epoch_list, MAPE_list, label = 'train MAPE')
    # ax1.plot(epoch_list, test_MAPE_list, label='test MAPE')
    # ax2.plot(epoch_list, MSE_list, label = 'train MSE')
    # ax2.plot(epoch_list, test_MSE_list, label='test MSE')
    # plt.legend(loc='upper right')
    # plt.show()