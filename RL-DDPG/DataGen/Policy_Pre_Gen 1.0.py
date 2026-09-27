import numpy as np
import random
import math

'''
Only Considering the last four days
RT - Remaining Time: 0 - 4*5
P - Price Range: 10^2 - 10^5
Util - Shipping utilization: 0 - 1
a - action
'''

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

N = 10000
time_steps = 15
x_list = []
y_list = []


for _ in range(N):
    RT = random.randint(0, 20)
    P_idx = random.random() * 600 + 1500
    #P = pow(10, P_idx)
    P = P_idx
    Util = random.random()
    if (Util < 0.15):
        a = (-108 / 20) / P
    elif (Util < 0.4):
        a = (18 / 20) / P
    elif (Util < 0.6):
        a = (23 / 20) / P
    else:
        a = (33 / 20) / P
    # print("Remaining Time: {:d} \t Price: {:7.2f} \t Util: {:.2f} \t action {:.2f}%".format(RT, P, Util, a*100))
    x = np.zeros((time_steps, 3))
    x[-1, :] = [P, Util, RT]
    x_list.append(x)
    y_list.append(a)

x_numpy = np.array(x_list)
y_numpy = np.array(y_list)
print("The empircal range of the action value is between {:.2f}% and {:.2f}%.".format(np.min(y_numpy)*100,np.max(y_numpy)*100))
# print("A sample of the training data")
# print(x_numpy[0])
# print("Label: {:.2f}% \n".format(y_numpy[0]*100))
print("Shape of x: {}".format(x_numpy.shape))
print("Shape of y: {}".format(y_numpy.shape))

np.save("../data/x_pretrain.npy", x_numpy)
np.save("../data/y_pretrain.npy", y_numpy)