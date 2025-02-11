import pandas as pd
import csv
import os
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation as R
import numpy as np

# success
def split_list(list):
    temp = []
    index_list = []
    s_list = []
    flag = 0
    for i in list[:,0]:
        try :
            temp.index(i)
        except ValueError:
            temp.append(i)
            index_list.append([])

        for a in range(len(temp)):
            if temp[a] == i:
                index_list[a].append(flag)
        flag += 1
    
    for i in range(len(index_list)):
        s_list.append([])
        for j in index_list[i]:
            s_list[i].append(list[j])
    return s_list

def get_new_filename(filename):
    basename = os.path.splitext(filename)[0]
    return basename

# exists: true !exists: false
def is_filename_exists(filename):
    return os.path.exists(filename)

def get_unique_filename(filename,counter):
    return f"{filename}_{counter}"

def chage_filename(filename,counter):
    flag = counter
    newfilename = filename
    while is_filename_exists(newfilename):
        flag += 1
        newfilename = get_unique_filename(get_new_filename(filename),flag)
    return f"{newfilename}.csv"

def write_csv(data_write,savedir):
    for i in range(len(data_write)):
        name = f"{savedir}/{data_write[i][0][0]}"
        with open(chage_filename(name,0),'w',newline="") as csvfile:
            a = csv.writer(csvfile)
            a.writerow(['SensorId', 
            #             ' AccX (g)',' AccY (g)',' AccZ (g)',
            #    ' GyroX (deg/s)',' GyroY (deg/s)',' GyroZ (deg/s)',
            #      ' MagX (uT)',' MagY (uT)',' MagZ (uT)',
            #        ' EulerX (deg)',' EulerY (deg)',' EulerZ (deg)',
               ' QuatW', ' QuatX', ' QuatY',' QuatZ',
               ' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)'])
            a.writerows(data_write[i])

def creat_file(f_path):
    split_fliename = f_path.split('/')
    split_fliename = os.path.splitext(split_fliename[-1])
    savefile = f"LSTM_classfication/dataset/{split_fliename[0]}_solved"
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    return savefile



def main():
    root = tk.Tk()
    root.withdraw()

    f_path = filedialog.askopenfilename()
    savedir = creat_file(f_path)

    data = pd.read_csv(f"{f_path}")
    data = pd.DataFrame(data)
    df = data[['SensorId', 
            #    ' AccX (g)',' AccY (g)',' AccZ (g)',
            #    ' GyroX (deg/s)',' GyroY (deg/s)',' GyroZ (deg/s)',
            #      ' MagX (uT)',' MagY (uT)',' MagZ (uT)',
            #        ' EulerX (deg)',' EulerY (deg)',' EulerZ (deg)',
               ' QuatW', ' QuatX', ' QuatY',' QuatZ',
               ' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]
    data_all = df.values

    writer = split_list(data_all)
    write_csv(writer,savedir)


main()
# print(df[['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ']])
