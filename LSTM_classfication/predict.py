import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
import csv

look_back = 100
features = 7
step_out = 1
epochs = 100
split = 0.8


# ========= read dataset ===================
a = pd.read_csv('dataset/singalimu_walk5ms.csv')
dataframe = a[[' QuatW',' QuatX',' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]
pd_value = dataframe.values

# ========= read model ===================
model = keras.models.load_model("model_data/lstm-model.h5")

# ========= split dataset ===================
train_size = int(len(pd_value) * split)
trainlist = pd_value[:train_size]
testlist = pd_value[train_size:]

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

# set predict_data
predict_begin = 1
predict_num = 100
predict_result = np.zeros((testY.shape[0],features),dtype=float)

print(testX[0].shape)

for i in range(len(testX)):
    data = np.reshape(testX[i],(1,100,7))
    predict_data = model.predict(data,verbose = 0)
    predict_result[i] = predict_data

def write_csv(data_write,savedir,filename = '1.csv'):
    name = f"{savedir}/{filename}"
    with open(name,'w',newline="") as csvfile:
        a = csv.writer(csvfile)
        a.writerow([' QuatW', ' QuatX', ' QuatY',' QuatZ',
                    ' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)'])
        for i in range(len(data_write)):
            a.writerows(data_write[i])

a = [' QuatW',' QuatX',' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']
x = [i for i in range(len(predict_result))]
for i in range(predict_result.shape[1]):
    plt.figure()
    plt.plot(x,testY[:,i],label = "real_data")
    plt.plot(x,predict_result[:,i],label = "predict_data")
    plt.legend(loc="lower right")
    plt.title(f"{a[i]}")
    plt.savefig(f"model_data/{a[i]}.png")
    
write_csv(predict_result,"model_data")
# for i in range(look_back):
#     predict_result[i] = testX[-predict_begin:][0,i]



# # predict
# for i in range(predict_num):
#     begin_data = np.reshape(predict_result[i:i+look_back,], (predict_begin, look_back, features))
#     predict_data = model.predict(begin_data) 
#     predict_result[look_back+i] = predict_data
#     buff = predict_result[i+1:i+look_back]
#     predict_call_back = np.append(buff,predict_data,axis=0)

# # show plot
# plt.plot(predict_result[-predict_num:,5])
# plt.plot()
# plt.show()