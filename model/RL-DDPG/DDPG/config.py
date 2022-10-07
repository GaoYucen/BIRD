
'''
Configurations
'''

'''
Hyperparaeter in real_train.py

    Constants
    ---------
    C_DIM: int
        an integer indicating the dimension of the hidden state of LSTM
    STATE_DIM: int
        an integer indicating the dimension of state in reinforcement learning
    TIME_STEPS: int
        an integer indicating the time steps in state
    ACTION_DIM: int
        an integer indicating the dimension of action in reinforcement learning
    ACTION_RANGE: float
        a float indicating the range of price adjustment
    INIT_PRICE: float
        a float indicating the initial price 
    TRAIN_EPOCH: int
        an integer indicating the training epochs
    TEST_EPOCH: int
        an integer indicating the testing epochs
    
    
    Path
    ---------
    RL_WEIGHT_PATH: string
        a string indicating the path of weights of reinforcement learning
    SIM_SALE_PATH: string
        a string indicating the path of data of simulation environment
    SIM_COUNT_PATH: string
        a string indicating the path of the distribution of sale of simulation environment
    REAL_DATA_PATH: string
        a string indicating the path of real data to train the reinforcement learning
    RL_RESULT_PATH: string
        a string indicating the path to save results of reinforcement learning


'''
C_DIM = 128
STATE_DIM = 3
TIME_STEPS = 5
ACTION_DIM = 1
ACTION_RANGE = 40
INIT_PRICE = 2500

TRAIN_EPOCH = 1000
TEST_EPOCH = 1000

RL_WEIGHT_PATH = 'weights_real_train/'
SIM_SALE_PATH = '../data/2hours_price_setting_env.csv'
SIM_COUNT_PATH = '../data/num_count.pkl'
REAL_DATA_PATH = '../data/YIK_QZH_training_data_source.xlsx'
RL_RESULT_PATH = 'model_result.csv'

'''
Hyperparaeter in pretrained_test.py
    
    Constants
    ---------
    PRE_EPOCH: int
        Number of pretrain epochs
    PRE_BATCH_SIZE: int
        Batch size in the pretrain process
        
    Path
    ---------
    X_DATA_PATH: string
        a string indicating the path of weights of pretraining
    Y_DATA_PATH: string
        a string indicating the path of weights of reinforcement learning
    PRE_WEIGHT_PATH: string
        a string indicating the path of weights of pretraining

'''
PRE_EPOCH = 1001
PRE_BATCH_SIZE = 64

X_DATA_PATH = '../data/x_pretrain.npy'
Y_DATA_PATH = '../data/y_pretrain.npy'
PRE_WEIGHT_PATH = 'weights/'

'''
Hyperparameter in ActorNetwork_LSTM.py

    Constants
    ---------
    ActorNetwork_LSTM_LSTM_HIDDEN_UNITS: int
        an integer indicating the dimension of the hidden state of LSTM
    ActorNetwork_LSTM_PRE_LEARNING_RATE: float
        a float indicating the learning rate of the Actor Network in the pretrain process
    ActorNetwork_LSTM_LEARNING_RATE: float
        a float indicating the learning rate of the Actor Network while training
    ActorNetwork_LSTM_TAU: float
        a float indicating the update rate of the target network (DDPG)
'''
ActorNetwork_LSTM_LSTM_HIDDEN_UNITS = 128
ActorNetwork_LSTM_PRE_LEARNING_RATE = 0.01
ActorNetwork_LSTM_LEARNING_RATE = 0.001
ActorNetwork_LSTM_TAU = 0.01

'''
Hyperparameter in ActorNetwork_StackedLSTM.py

    Constants
    ---------
    ActorNetwork_StackedLSTM_LSTM_HIDDEN_UINITS: int
        an integer indicating the dimension of the hidden state of stacked LSTM
    ActorNetwork_StackedLSTM_LSTM_LAYER_NUM: int
        an integer indicating the number of stacked layers of LSTM
    ActorNetwork_StackedLSTM_PRE_LEARNING_RATE: float
        a float indicating the learning rate of the Actor Network in the pretrain process
    ActorNetwork_StackedLSTM_LEARNING_RATE: float
        a float indicating the learning rate of the Actor Network while training
    ActorNetwork_StackedLSTM_TAU: float
        a float indicating the update rate of the target network (DDPG)
'''
ActorNetwork_StackedLSTM_LSTM_HIDDEN_UINITS = 128
ActorNetwork_StackedLSTM_LSTM_LAYER_NUM = 2
ActorNetwork_StackedLSTM_PRE_LEARNING_RATE = 0.01
ActorNetwork_StackedLSTM_LEARNING_RATE = 0.0005
ActorNetwork_StackedLSTM_TAU = 0.001

'''
Hyperparameter in CriticNetwork_LSTM.py

    Constants
    ---------
    CriticNetwork_LSTM_LSTM_HIDDEN_UINITS: int
        an integer indicating the dimension of the hidden state of LSTM in Critic Network
    CriticNetwork_LSTM_LEARNING_RATE: float
        a float indicating the learning rate of the Critic Network
    CriticNetwork_LSTM_TAU: float
        a float indicating the update rate of the target network (DDPG)
    CriticNetwork_LSTM_L2: float
        a float indicating the coefficient for L2 regularization
        
'''
CriticNetwork_LSTM_LSTM_HIDDEN_UINITS = 128
CriticNetwork_LSTM_LEARNING_RATE = 0.001
CriticNetwork_LSTM_TAU = 0.001
CriticNetwork_LSTM_L2 = 0.01

'''
Hyperparameter in CriticNetwork_StackedLSTM.py

    Constants
    ---------
    CriticNetwork_StackedLSTM_LSTM_HIDDEN_UINITS: int
        an integer indicating the dimension of the hidden state of LSTM in Critic Network
    CriticNetwork_StackedLSTM_LEARNING_RATE: float
        a float indicating the learning rate of the Critic Network
    CriticNetwork_StackedLSTM_LSTM_LAYER_NUM: int
        an integer indicating the number of stacked layers of LSTM
    CriticNetwork_StackedLSTM_TAU: float
        a float indicating the update rate of the target network (DDPG)
    CriticNetwork_StackedLSTM_L2: float
        a float indicating the coefficient for L2 regularization

'''
CriticNetwork_StackedLSTM_LSTM_HIDDEN_UINITS = 64
CriticNetwork_StackedLSTM_LEARNING_RATE = 0.001
CriticNetwork_StackedLSTM_LSTM_LAYER_NUM = 2
CriticNetwork_StackedLSTM_TAU = 0.001
CriticNetwork_StackedLSTM_L2 = 0.01

'''
Hyperparameter in DDPG.py

    Constants
    ---------
    DDPG_REPLAY_BUFFER_SIZE: int
    DDPG_REPLAY_START_SIZE: int
    DDPG_BATCH_SIZE: int
    DDPG_GAMMA: float
'''
DDPG_REPLAY_BUFFER_SIZE = 1000000
DDPG_REPLAY_START_SIZE = 10000
DDPG_BATCH_SIZE = 64
DDPG_GAMMA = 0.99


