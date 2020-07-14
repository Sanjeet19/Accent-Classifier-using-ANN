import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

Z = pd.read_csv('D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/Features.csv',)
print(Z.shape)
# print(Z)
# exit(0)

X_train, X_test = train_test_split(Z, test_size=0.3)

Y1 = X_train.copy()
Y1 = Y1.iloc[:,-1]
Y2 = X_test.copy()
Y2 = Y2.iloc[:,-1]
ncol1 = len(Y1)
ncol2 = len(Y2)

Y_train = np.zeros((ncol1, 9))
Y_test = np.zeros((ncol2, 9))

for i in range(ncol1):
    n = Y1.iloc[i, ]
    Y_train[i, int(n)] = int(n)+1

for i in range(ncol2):
    n = Y2.iloc[i, ]
    Y_test[i, int(n)] = int(n)+1


X_train.drop(["3000", "Unnamed: 0"], axis=1, inplace=True)
X_test.drop(["3000", "Unnamed: 0"], axis=1, inplace=True)
X_train = tf.keras.utils.normalize(X_train.values, axis=1)
X_test = tf.keras.utils.normalize(X_test.values, axis=1)

pd.DataFrame(Y_train).to_csv("D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/Y_train.csv")
pd.DataFrame(Y_test).to_csv("D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/Y_test.csv")
pd.DataFrame(X_train).to_csv("D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/X_train.csv", sep=',')
pd.DataFrame(X_test).to_csv("D:/Programming/Python/Audio Processing/Accent project/A Code/CSV/X_test.csv", sep=',')
