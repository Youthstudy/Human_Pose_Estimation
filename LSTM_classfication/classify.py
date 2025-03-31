import keras.utils
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *

from sklearn.model_selection import train_test_split
 
# 创建时间步长的数据集
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:(i+time_steps), :])
        y.append(data[i+time_steps, :])
    return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range = (0,1))
#划分数据集
def generate_classification_train_data(splitrate = 0.8,time_step = 1):
    # lris_df = datasets.load_iris()
    # X_data = lris_df.data
    # y_data = lris_df.target
    data = pd.read_csv(r'train\train_all.csv',header=None,encoding='utf-8')
     
    # y_data = pd.read_csv(r'train\label.csv',header=None,encoding='utf-8')
    data = np.array(data)
    data_guiyi = scaler.fit_transform(data)
    X,Y = create_dataset(data_guiyi,time_step)
    train_size = int(len(X) * splitrate)
    x_train, x_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]

    # X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,test_size=0.3)
 
    # x_train = np.array(X_train)
    # x_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test
 
# Seque构建方式（推荐）
class SequeClassifier():
    def __init__(self, units):
        self.units = units
        self.model = None
     
    #构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    def build_model(self, loss, optimizer, metrics,dropout):
        self.model = Sequential()
        self.model.add(LSTM(self.units,return_sequences=True))
        self.model.add(LSTM(self.units))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(4)) 
        
        self.model.compile(loss=loss, 
                           optimizer=optimizer, 
                           metrics=metrics)


if __name__ == "__main__":
    #1 获取训练数据集，并调整为三维输入格式
   
    dirs = "model"
    save_history = "history"
    loss = "mse"
    optimizer_list = ["adam"]
    time_step = [i for i in range(1,101)]
    dropout_list = [0.2]
    metrics = ['accuracy','categorical_accuracy']
    #2 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    units = 128 #lstm细胞个数

    sclstm = SequeClassifier(units) 
    for i in range(28,100):
        for optimizer in optimizer_list:
            for dropout in dropout_list:
                x_train, y_train, x_test, y_test = generate_classification_train_data(time_step=i)
                model_name = loss+'_'+optimizer+'_'+f"{dropout}"+f"_{i}"
                #评估方法类型
                sclstm.build_model(loss, optimizer, metrics,dropout)
                
                #3 训练模型
                epochs = 100
                batch_size = 64
                history = sclstm.model.fit(x_train, y_train, epochs=epochs,validation_data=(x_test, y_test), batch_size=batch_size)

                #4 模型评估
                score = sclstm.model.evaluate(x_test, y_test, batch_size=64)
                print("model score:", score)
                
                #5 模型持久化，把模型保存在本地
                
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                print("正在保存模型......")
                sclstm.model.save(dirs+"/" + model_name+".h5")
                print("模型已保存.save path-->"+dirs+"/"+model_name+".h5")

                # 6 保存历史数据
                print("正在保留历史数据......")

                hist_df = pd.DataFrame(history.history)
                #save to csv:
                hist_csv_file = save_history+"/"+model_name+".csv"
                with open(hist_csv_file, mode='w',newline = "") as f:
                    hist_df.to_csv(f)

                print("保存完成")
                
    #6 从指定模型保存的位置读取模型，做预测
    from keras.models import load_model
    read_model = load_model(dirs+"/" + model_name+".h5")
    out = read_model.predict(x_test)
    print("out:%s"%out)