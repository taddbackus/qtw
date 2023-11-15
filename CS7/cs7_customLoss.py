import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers

df = pd.read_csv('/Users/taddbackus/School/fall23/qtw/cs7/final_project(5).csv')

columns_to_encode = ['x24','x29','x30']
for i in columns_to_encode:
    df[i] = df[i].fillna('unknown')

missingData = []
for i in df:
    if df[i].isnull().sum() > 0:
        print(i,':',df[i].isnull().sum(),'missing')
        print(i,':',df[i].isnull().sum() / len(df),'%')
        missingData.append(i)

df['x37'] = df['x37'].str.replace('$','').astype(float)
df['x32'] = df['x32'].str.replace('%','').astype(float)

for i in missingData:
    df[i].fillna(df[i].median(),inplace=True)
for i in df:
    if df[i].isnull().sum() > 0:
        print(i, ':', df[i].isnull().sum(), 'missing')

df = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)

X = df.drop(columns='y')
y = df['y']
print(df.shape)
print(X.shape)
print(y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

value_counts = df['y'].value_counts()
print(value_counts)
lowestTotalCost = value_counts[0] * 40 + value_counts[1] * 100
print(lowestTotalCost)

def threshold_test(probabilities, th):
    return [1 if prob >= th else 0 for prob in probabilities]
def cost_score(confMatrix):
    return (confMatrix[0][1] * 40 + confMatrix[1][0] * 100) * 5
def find_cost(yProb):
    thresholds = np.linspace(0,1,101)
    lowestTh = 1
    lowestCost = lowestTotalCost
    for t in thresholds:
        conf_matrix = confusion_matrix(y_test, threshold_test(yProb,t))
        cost  = cost_score(conf_matrix)
        if cost < lowestCost:
            lowestCost = cost
            lowestTh = t
            lowestCM = conf_matrix
        print('Threshold:',t)
        print('Total Money Lost:',cost)
        print('==================')
    return lowestCost, lowestTh, lowestCM

X_train_NN, X_val_NN, y_train_NN, y_val_NN = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.1)


class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name='custome_weighted_loss', **kwargs):
        super(CustomWeightedLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        #y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        binary_crossentropy = tf.keras.losses.binary_crossentropy(y_true,y_pred)
        weighted_loss = tf.where(tf.math.equal(y_true,0), 40*binary_crossentropy, 100*binary_crossentropy)

        return weighted_loss

class CustomWeightedLoss2(tf.keras.losses.Loss):
    def __init__(self, name='custome_weighted_loss', **kwargs):
        super(CustomWeightedLoss2, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        #y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        binary_crossentropy = tf.keras.losses.binary_crossentropy(y_true,y_pred)

        mask_0_as_1 = tf.math.logical_and(tf.math.equal(y_true,0),tf.math.greater(y_pred,0.5))
        mask_1_as_0 = tf.math.logical_and(tf.math.equal(y_true,1),tf.math.less(y_pred,0.5))
        #weighted_loss = binary_crossentropy + tf.where(mask_0_as_1,40.0,0.0) + tf.where(mask_1_as_0,100.0,0.0)
        weighted_loss = tf.where(mask_0_as_1,40.0*binary_crossentropy,
                                 tf.where(mask_1_as_0,100*binary_crossentropy,
                                          binary_crossentropy))
        return weighted_loss
    
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=5)

nnModel = tf.keras.Sequential()
nnModel.add(tf.keras.Input(shape=(70,)))
nnModel.add(layers.Dense(128,activation='relu'))
nnModel.add(layers.Dense(128,activation='relu'))
nnModel.add(layers.Dropout(0.3))
nnModel.add(layers.Dense(64,activation='relu'))
nnModel.add(layers.Dense(64,activation='relu'))
nnModel.add(layers.Dropout(0.3))
nnModel.add(layers.Dense(1,activation='sigmoid'))

nnModel.compile(optimizer='adam',
                loss=CustomWeightedLoss2(),
                metrics=['accuracy'])

history = nnModel.fit(X_train_NN,
                      y_train_NN,
                      epochs=1000,
                      batch_size=32,
                      callbacks=[callback],
                      validation_data=[X_val_NN,y_val_NN])

loss, acc = nnModel.evaluate(X_test,y_test)
print(acc)

'''
plt.plot(np.linspace(1,10,10),history.history['val_loss'],label='validation')
plt.plot(np.linspace(1,10,10),history.history['loss'],label='train')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Neural Network loss over epochs')
plt.legend()
plt.show()
'''

nnProb = nnModel.predict(X_test)

print(find_cost(nnProb))