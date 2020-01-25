

import matplotlib.pyplot as plt


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

leng=3

data = [[i+j for j in range(leng)] for i in range(100)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(1,101)]
target = np.array(target, dtype=np.float32)

data = data.reshape(100, 1, leng)/200
target = target.reshape(100,1,leng)/200

# Build Model
model = Sequential()
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(data, target, nb_epoch=5000, batch_size=50,validation_data=(data,target))


predict = model.predict(data)

plt.scatter(range(20),predict,c='r')
plt.scatter(range(20),data,c='g')
plt.show()