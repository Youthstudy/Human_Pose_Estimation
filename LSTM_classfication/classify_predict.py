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



model_Path = r"E:\1.postgraduate\2.project\5.github\Human_Pose_Estimation\LSTM_classfication\model"
Folder_Path  = r"E:\1.postgraduate\2.project\5.github\Human_Pose_Estimation\LSTM_classfication\pre"
Save_Path = r"E:\1.postgraduate\2.project\5.github\Human_Pose_Estimation\LSTM_classfication\predict"

# 将该文件夹下的所有文件名存入一个列表
os.chdir(Folder_Path)
file_list = os.listdir()
print(file_list)

# 读取模型地址
os.chdir(model_Path)
model_list = os.listdir()
print(model_list)
for j in model_list:
    read_model = load_model(model_Path+"//"+j)
    #os.mkdir(Save_Path+'\\'+j )
    for i in file_list:
        root = Folder_Path + '/'+ i
        X_data = pd.read_csv(root,header=None,encoding='utf-8')
        X_data = X_data.values
        X_data = np.delete(X_data,0,axis=0)
        X_data = X_data.astype('float64')
        #6 从指定模型保存的位置读取模型，做预测
        out = read_model.predict(X_data)
        with open(Save_Path+'\\'+j + '\\' + i, "w", encoding="utf-8",newline='') as f:
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 3. 构建列表头
            name=['0','1']

            csv_writer.writerow(name)
            csv_writer.writerows(out)
            print("写入数据成功")
            # 5. 关闭文件
            f.close()

