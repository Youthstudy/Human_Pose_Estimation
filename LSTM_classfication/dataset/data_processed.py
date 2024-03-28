import pandas as pd
import csv
import os
import tkinter as tk
from tkinter import filedialog



# success
def split_list(list):
    temp = []
    index_list = []
    s_list = []
    flag = 0
    for i in list[0]:
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
        a = 0
        for j in index_list[i]:
            s_list[i].append([])
            for k in range(len(list)):
                s_list[i][a].append(list[k][j])
            a += 1
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
    while is_filename_exists(filename):
        flag += 1
        filename = get_unique_filename(get_new_filename(filename),flag)
    return filename

#******************************************
def write_csv(data_write,root):
    for i in range(len(data_write)):
        name = f"{root}/{data_write[i][0][0]}.csv"
        with open(chage_filename(name,0),'w+',newline="") as csvfile:
            a = csv.writer(csvfile)
            a.writerow(['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)'])
            a.writerows(data_write[i])

def quat2martix(w,x,y,z):
    martix = []
    martix[0] = 1-2*y*y-2*z*z
    martix[1] = 2*x*y-2*w*z 
    martix[2] = 2*w*y+ 2*x*z
    martix[3] = 2*x*y+2*w*z
    martix[4] = 1-2*x*x-2*z*z 
    martix[5] = 2*y*z-2*w*x 
    martix[6] = 2*x*z-2*w*y 
    martix[7] = 2*w*x+2*y*z 
    martix[8] = 1-2*x*x-2*y*y
    return martix



def main():
    root = tk.Tk()
    root.withdraw()

    f_path = filedialog.askopenfilename()

    data = pd.read_csv(f"{f_path}")
    data = pd.DataFrame(data)
    df = data[['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ',' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]

    Sensor = df['SensorId'].to_list()
    time = df[' TimeStamp (s)'].to_list()
    FrameNumber = df[' FrameNumber'].to_list()
    QuatW = df[' QuatW'].to_list()
    QuatX = df[' QuatX'].to_list()
    QuatY = df[' QuatY'].to_list()
    QuatZ = df[' QuatZ'].to_list()
    LinAccX = df[' LinAccX (g)'].to_list()
    LinAccY = df[' LinAccY (g)'].to_list()
    LinAccZ = df[' LinAccZ (g)'].to_list()

    data_all = []
    data_all.append(Sensor)
    data_all.append(time)
    data_all.append(FrameNumber)
    data_all.append(QuatW)
    data_all.append(QuatX)
    data_all.append(QuatY)
    data_all.append(QuatZ)
    data_all.append(LinAccX)
    data_all.append(LinAccY)
    data_all.append(LinAccZ)


    writer = split_list(data_all)
    print(writer)
    write_csv(writer,f_path)


main()
# print(df[['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ']])
