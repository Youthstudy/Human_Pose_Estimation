import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
import os
from keras.models import load_model
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
 
#划分数据集
def generate_classification_train_data():
    # X_data = pd.read_csv(r'train\train.csv',header=None,encoding='utf-8')
    # y_data = pd.read_csv(r'train\label.csv',header=None,encoding='utf-8')
    lris_df = datasets.load_iris()
    X_data = lris_df.data
    y_data = lris_df.target
 
    X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,test_size=0.2)
 
    x_train = np.array(X_train)
    x_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test
 
#Seque构建方式（推荐）
class SequeClassifier():
    def __init__(self, units):
        self.units = units
        self.model = None
     
    #构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    def build_model(self, loss, optimizer, metrics):
        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=True))
        self.model.add(LSTM(self.units))
        self.model.add(Dense(3, activation='softmax')) #最后一层全连接层。对于N分类问题，最后一层全连接输出个数为N个，这里鸢尾花数据为3分类问题；
        
        self.model.compile(loss=loss, 
                           optimizer=optimizer, 
                           metrics=metrics)
 
 
if __name__ == "__main__":
    #1 获取训练数据集，并调整为三维输入格式
    x_train, y_train, x_test, y_test = generate_classification_train_data()
        
    x_train = x_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    
    #2 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    units = 128 #lstm细胞个数
    loss = "sparse_categorical_crossentropy"  #损失函数类型
    optimizer = "adam"  #优化器类型
    metrics = ['accuracy']  #评估方法类型
    sclstm = SequeClassifier(units) 
    sclstm.build_model(loss, optimizer, metrics)
 
    #3 训练模型
    epochs = 100
    batch_size = 64
    history = sclstm.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
 
    #4 模型评估
    score = sclstm.model.evaluate(x_test, y_test, batch_size=16)
    print("model score:", score)
    
    # 模型应用：预测
    #proba_prediction = sclstm.model.predict(x_test)
    
    #5 模型持久化，把模型保存在本地
    dirs = "model"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print("正在保存模型......")
    sclstm.model.save(dirs+"/classifier_model.h5")
    print("模型已保存.save path-->dirs%s"%"/classifier_model.h5")
    
    print("正在保留历史数据......")

    hist_df = pd.DataFrame(history.history)
    #save to csv:
    hist_csv_file = dirs+"/classifier_model"+".csv"
    with open(hist_csv_file, mode='w',newline = "") as f:
        hist_df.to_csv(f)

    print("保存完成")

    #6 从指定模型保存的位置读取模型，做预测
    from keras.models import load_model
    read_model = load_model(dirs+"/classifier_model.h5")
    out = read_model.predict(x_test)
    print("out:%s"%out)