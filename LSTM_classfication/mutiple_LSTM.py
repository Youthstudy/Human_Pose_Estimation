import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import math
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

split = 0.3
n_past = 100
n_pre = 50
dim_pre = 1

def createXY(dataset,n_past,n_pre,dim_pre):
  dataX = []
  dataY = []
  for i in range(n_past, len(dataset)-n_pre):
          dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
          dataY.append(dataset[i:i+n_pre,2])
  return np.array(dataX),np.array(dataY)


df = pd.read_csv("signalimu_walk2ms.csv")
df = df[[' QuatW',' QuatX',' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]
df_add = df[[' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]

def quat2rotation(dataset,add = []):
  data_all = []
  for i in range(len(dataset)):
    temp_r = R.from_quat([dataset[i][0],dataset[i][1],dataset[i][2],dataset[i][3]])
    temp_m = temp_r.as_matrix()
    a = temp_m.flatten().tolist()
    data_all.append(a)
    for j in range(len(add[i])):
        data_all[i].append(add[i][j])
  return data_all
  
df = quat2rotation(df.values,df_add.values)
df = np.array(df)

# split into train and test
test_split = int(len(df) * split)
df_for_training = df[:-test_split]
df_for_testing = df[-test_split:]

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.fit_transform(df_for_testing)

trainX,trainY=createXY(df_for_training_scaled,100,50)
testX,testY=createXY(df_for_testing_scaled,100,50)


