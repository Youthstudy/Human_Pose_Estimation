'''
Data:2017-07-13
Auther;JXNU Kerwin
Description:使用Pandas拼接多个CSV文件到一个文件（即合并）
'''
import pandas as pd
import os

Folder_Path = r'E:\1.postgraduate\2.project\5.github\Human_Pose_Estimation\LSTM_classfication\dataset\lstmtrain'          #要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path =  r'E:\1.postgraduate\2.project\5.github\Human_Pose_Estimation\LSTM_classfication\train'       #拼接后要保存的文件路径
SaveFile_Name = r'train_all.csv'              #合并后要保存的文件名
 
# 修改当前工作目录

os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir()

# 读取第一个CSV文件并包含表头
df = pd.read_csv(Folder_Path +'\\'+ file_list[0],header = None)
print(df)