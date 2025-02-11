import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
 
from sklearn import datasets
from sklearn.model_selection import train_test_split
 
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

#划分数据集
def generate_classification_train_data():
    lris_df = datasets.load_iris()
    X_data = lris_df.data
    y_data = lris_df.target
    X_data = pd.read_csv('./train/traindata_all.csv',header=None,encoding='utf-8')
    y_data = pd.read_csv('./train/trainlabel_all.csv',header=None,encoding='utf-8')

    X_data = X_data.values
    y_data = y_data.values

    X_data = np.delete(X_data,0,axis=0)
    y_data = np.delete(y_data,0,axis=0)

    X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,test_size=0.3)
 
    x_train = np.array(X_train)
    x_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')
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
        self.model.add(Dense(2, activation='softmax')) #最后一层全连接层。对于N分类问题，最后一层全连接输出个数为N个，这里鸢尾花数据为3分类问题；
        
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
    sclstm.model.fit(x_train, y_train, epochs=epochs,validation_data=(x_test, y_test), batch_size=batch_size)
 
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
    sclstm.model.save(dirs+"/classifier_model2.h5")
    print("模型已保存.save path-->dirs%s"%"/classifier_model2.h5")
    
    #6 从指定模型保存的位置读取模型，做预测
    # from keras.models import load_model
    # read_model = load_model(dirs+"/classifier_model.h5")
    # out = read_model.predict(x_test)
    # print("out:%s"%out)