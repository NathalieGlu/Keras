# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:06:25 2017

@author: HATA
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import time

'''train_genue = np.genfromtxt('.\\train\\g\\text.csv', delimiter=',')
train_forged = np.genfromtxt('.\\train\\f\\text.csv', delimiter=',')
test_genue = np.genfromtxt('.\\test\\g\\text.csv', delimiter=',')
test_forged = np.genfromtxt('.\\test\\f\\text.csv', delimiter=',')'''

first = time.time()
genue = np.genfromtxt('D:\\python\\Keras\\sign\\g\\genue.csv', delimiter=',')
forged1 = np.genfromtxt('D:\\python\\Keras\\sign\\f\\forged.csv', delimiter=',')


def shuffle(x):
    np.random.shuffle(x)
    return x[:,:]

def shuffle_simple(x, y):
    stack = np.column_stack((x, y))
    np.random.shuffle(stack)
    return stack[:,:stack.shape[1]-1], stack[:, stack.shape[1]-1]

genue, num1 = shuffle_simple(genue, range(genue.shape[0]))
forged1, num2 = shuffle_simple(forged1, range(forged1.shape[0]))

n_genue = 10
n_forged = 10

f = open('log.txt', 'a')
for line in num1[:n_genue]:
    f.write(str(int(line)) + ' ')
f.write('\n')    
for line in num2[:n_forged]:
    f.write(str(int(line)) + ' ')
f.write('\n')
f.close()

train_genue = genue[:n_genue]
test_genue = genue[n_genue:]

train_forged = forged1[:n_forged]
test_forged = forged1[n_forged:]

def normalization(arr):
    for i in range(0, arr.shape[1]):
        #minimum = arr[:, i].min()
        maximum = arr[:, i].max()
        if maximum == 0:
            arr[:, i] = 0
        else:
            for j in range(0, arr.shape[0]):
                arr[j, i] = arr[j, i]/maximum
    return arr

def shuffle(x, y):
    stack = np.column_stack((x, y))
    np.random.shuffle(stack)
    return stack[:,:stack.shape[1]-1], stack[:, stack.shape[1]-1]

train_genue = normalization(train_genue)
train_forged = normalization(train_forged)
test_genue = normalization(test_genue)
test_forged = normalization(test_forged)

X = np.vstack((train_genue, train_forged))
Y = np.concatenate((np.ones(train_genue.shape[0]), np.zeros(train_forged.shape[0])))
#X, Y = shuffle(X,Y)
    
# create model
model = Sequential()
model.add(Dense(156, input_dim=164, activation="tanh", kernel_initializer="uniform"))
model.add(Dense(64, activation="tanh", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# tensorboarg graph
tensorboard = TensorBoard(log_dir=".\\output", histogram_freq=0, write_graph=True, write_images=True)
# tensorboard embending
'''embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))
tensorboard = TensorBoard(log_dir='output', histogram_freq=0, batch_size=10,
                           write_graph=True, write_grads=True, write_images=True,
                           embeddings_freq=50, embeddings_metadata=None,
                           embeddings_layer_names=embedding_layer_names)'''
# Fit the model
model.fit(X, Y, epochs=400, batch_size=10, verbose=2, callbacks=[tensorboard])
# calculate predictions
predictions = model.predict(X)
predictions1 = model.predict(test_genue)
predictions2 = model.predict(test_forged)
last = time.time()
print (last-first)
# round predictions as step func
rounded = [round(x[0]) for x in predictions]
rounded1 = [round(x[0]) for x in predictions1]
rounded2 = [round(x[0]) for x in predictions2]
predictions = Y - rounded
predictions1 = np.ones(test_genue.shape[0]) - rounded1
predictions2 = np.zeros(test_forged.shape[0]) - rounded2
print(predictions)
print(str(np.count_nonzero(predictions)) + "/" + str(len(predictions)))
print(predictions1)
print(str(np.count_nonzero(predictions1)) + "/" + str(len(predictions1)))
print(predictions2)
print(str(np.count_nonzero(predictions2)) + "/" + str(len(predictions2)))

f = open('log.txt', 'a')
f.write(str(last-first) + ' sec. ')
f.write(str(np.count_nonzero(predictions)/len(predictions)*100.0) + ' ')
f.write(str(np.count_nonzero(predictions1)/len(predictions1)*100.0) + ' ')
f.write(str(np.count_nonzero(predictions2)/len(predictions2)*100.0) + '\n\n')
f.close()

'''for layer in model.layers:
    weights = layer.get_weights()'''
'''
kearas model plot
from keras.utils import plot_model
plot_model(model, to_file='model.png')
model.summary()'''

'''
train_forged = np.vstack((forged1[87],forged1[91],forged1[94],forged1[98]))
test_forged = np.vstack((forged1[:87],forged1[88:91],forged1[91:94],forged1[96:98],forged1[99:]))
'''