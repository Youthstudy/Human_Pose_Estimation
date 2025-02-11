import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras

# # set GPU
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# print(len(gpus))
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# print(len(logical_gpus))

# # read data-sensor.csv
filePath = "E:\\2.job\\2.wendu\\dataset"
a = []
b = []
c = []
for i,j,k in os.walk(filePath):
    a.append(i)
    b.append(j)
    c.append(k)


needroot = []
for i in range(2,len(a)):
    for j in range(len(c[i])):
        needroot.append(a[i] + "\\" + c[i][j])

a = []
for i in range(len(needroot)):
    b = pd.read_csv(needroot[i])
    a.append(b[['VALUE']].values)

data = []
for i in range(len(a[0])):
    data.append([])
    for j in range(len(a)):
        data[i].append(float(a[j][i]))
        
pd_value = np.array(data)

# a = pd.read_csv('dataset/singalimu_walk5ms.csv')
# dataframe = a[[' QuatW',' QuatX',' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]
# pd_value = dataframe.values

# ========= split dataset ===================
train_size = int(len(pd_value) * 0.8)
trainlist = pd_value[:train_size]
testlist = pd_value[train_size:]

look_back = 6
features = 25
step_out = 1
epochs = 100
batch_size = 16

scaler = MinMaxScaler(feature_range=(0,1))
trainlist = scaler.fit_transform(trainlist)
testlist = scaler.fit_transform(testlist)

# ========= numpy train ===========
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back + 1])
    return np.array(dataX),np.array(dataY)
#训练数据太少 look_back并不能过大
trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)

print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
print(trainX[0].shape)


# # ========== set dataset ======================
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], features))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] , features))

# create and fit the LSTM network
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(look_back, features)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='relu'))
model.add(tf.keras.layers.Dense(features))
#model.compile(optimizer='adam', loss='mse')
model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

model.summary()

history = model.fit(trainX, trainY, batch_size= batch_size,validation_data=(testX, testY),epochs=epochs, verbose=1).history
model.save("model_data/lstm-model1.h5")

plt.figure()
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("model_data/model_loss.png")
#plt.ylim(ymin=0.70,ymax=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

for i in range(testPredict.shape[1]):
    plt.figure()
    plt.plot(testY[:,i],label = "real_data")
    plt.plot(testPredict[:,i],label = "predict_data")
    plt.legend(loc="lower right")
    # plt.title(f"{a[i]}")
    plt.savefig(f"model_data/{i}.png")
    

plt.show()


