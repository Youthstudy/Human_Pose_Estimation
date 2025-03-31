import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def getPathByDialog():
    root = tk.Tk()
    root.withdraw()
    Filepath = filedialog.askopenfilename() #获得选择好的文件
    # print('Filepath:', Filepath)
    return Filepath

def getSavePathByDialog():
    root = tk.Tk()
    root.withdraw()
    Foldpath = filedialog.askdirectory() #获得选择好的文件
    print('Filepath:', Foldpath)
    return Foldpath

data = []
tmp1 = []
savepath = "E:/1.postgraduate/2.project/5.github/Human_Pose_Estimation/LSTM_classfication/dataset/train1/data_hebing"
for i in range(2):
    path = getPathByDialog()
    df = pd.read_csv(path,header=None)
    df = np.array(df.values)
    tmp1.append(df[:,1:5])
tmp = np.hstack([tmp1[0],tmp1[1]])

np.savetxt(savepath + '\\' + '3.csv',tmp,delimiter=",")

