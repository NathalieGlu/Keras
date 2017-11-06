# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:06:25 2017

@author: HATA
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

seed = 7
np.random.seed(seed)

train_genue = np.genfromtxt('.\\train\\g\\text.csv', delimiter=',')
train_forged = np.genfromtxt('.\\train\\f\\text.csv', delimiter=',')
test_genue = np.genfromtxt('.\\test\\g\\text.csv', delimiter=',')
test_forged = np.genfromtxt('.\\test\\f\\text.csv', delimiter=',')

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

train_genue = normalization(train_genue)
train_forged = normalization(train_forged)
test_genue = normalization(test_genue)
test_forged = normalization(test_forged)

X = np.vstack((train_genue, train_forged))
Y = np.concatenate((np.ones(train_genue.shape[0]), np.zeros(train_forged.shape[0])))
    
# create model
model = Sequential()
model.add(Dense(156, input_dim=164, activation="tanh", kernel_initializer="uniform"))
model.add(Dense(64, activation="tanh", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
tensorboard = TensorBoard(log_dir=".\\output", histogram_freq=0, write_graph=True, write_images=True)
# tensorboard embending
#tensorboard = TensorBoard(log_dir=".\\output", write_graph=True, write_images=True,embeddings_freq=10,embeddings_layer_names=embedding_layer_names, embeddings_metadata=None)
model.fit(X, Y, epochs=200, batch_size=10, verbose=2, callbacks=[tensorboard])
# calculate predictions
predictions = model.predict(X)
predictions1 = model.predict(test_forged)
predictions2 = model.predict(test_genue)
# round predictions as step func
rounded = [round(x[0]) for x in predictions]
rounded1 = [round(x[0]) for x in predictions1]
rounded2 = [round(x[0]) for x in predictions2]
predictions = Y - rounded
predictions1 = np.zeros(test_forged.shape[0]) - rounded1
predictions2 = np.ones(test_genue.shape[0]) - rounded2
print(predictions)
print(predictions1)
print(predictions2)

for layer in model.layers:
    weights = layer.get_weights()

#kearas model plot
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
#model.summary()
'''
for i in range(0, 164):
    minimum = train_genue[:, i].min()
    maximum = train_genue[:, i].max()
    if maximum == 0:
        train_genue[:, i] = 0
    elif minimum == maximum:
        train_genue[:, i] = 1
    else:
        for j in range(0, train_genue.shape[0]):
            train_genue[j, i] = (0.99*(train_genue[j, i]-minimum)-0.0001*(train_genue[j, i]-maximum))/(maximum-minimum)'''