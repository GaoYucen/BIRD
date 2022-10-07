import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

train_csvfile = open('../data/train_MAPE_MSE.csv', 'r')
test_csvfile = open('../data/test_MAPE_MSE.csv', 'r')
train_reader = csv.reader(train_csvfile)
test_reader = csv.reader(test_csvfile)
next(train_reader)
next(test_reader)

epoch_list = []
train_MAPE_list = []
train_MSE_list = []
test_MAPE_list = []
test_MSE_list = []

count = 0
for row in train_reader:
    count += 1
    if (count % 2 != 0): continue
    epoch_list.append(int(row[0]))
    train_MAPE_list.append(float(row[1]))
    train_MSE_list.append(float(row[2]))

count = 0
for row in test_reader:
    count += 1
    if (count % 2 != 0): continue
    test_MAPE_list.append(float(row[1]))
    test_MSE_list.append(float(row[2]))

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
ax1.plot(epoch_list[550:], train_MAPE_list[550:], label = 'train MAPE')
ax1.plot(epoch_list[550:], test_MAPE_list[550:], label='test MAPE')
ax2.plot(epoch_list[550:], train_MSE_list[550:], label = 'train MSE')
ax2.plot(epoch_list[550:], test_MSE_list[550:], label='test MSE')
plt.legend(loc='upper right')
plt.show()