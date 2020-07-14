import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from sklearn import metrics
import tensorflow as tf
import numpy.core.multiarray
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# X_train=pd.read_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/X_train.csv',)
# X_train = X_train.drop('Unnamed: 0', 1)
# X_test=pd.read_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/X_test.csv',)
# X_test = X_test.drop('Unnamed: 0', 1)
# Y_train=pd.read_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/Y_train.csv',)
# Y_train = Y_train.drop('Unnamed: 0', 1)
# Y_test=pd.read_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/Y_test.csv',)
# Y_test = Y_test.drop('Unnamed: 0', 1)

ncol = len(X_train.columns)

#Dnn Start

model = Sequential()

model.add(Dense(200, input_shape=(ncol,),activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(150, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(150, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(15, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(15, activation='relu'))

model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

def exp_decay(epoch):
    initial_lrate = 0.0001
    t = epoch
    k = 0.1
    lrate = initial_lrate * math.exp(-k*t)
    print(lrate)
    return lrate


lrate = LearningRateScheduler(exp_decay)

model.fit(X_train, Y_train, batch_size=64, epochs=100, callbacks=[lrate])


scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

for layer in model.layers:
    weights = layer.get_weights()

pred = model.predict(X_test)
rounded = [round(x[0]) for x in pred]
# print(rounded)
# print(Y_test)

df = pd.DataFrame(weights)
df.to_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/weights.csv', index=False)

model.fit(X_test, Y_test, batch_size=len(X_test),)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
