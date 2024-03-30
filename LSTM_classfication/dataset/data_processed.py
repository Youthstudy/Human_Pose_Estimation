import pandas as pd
import csv
import os
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation as R

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
            a.writerow(['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ',
                        ' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)',
                        'rm0','rm1','rm2','rm3','rm4','rm5','rm6','rm7','rm8'])
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
    df = data[['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ',
               ' LinAccX (g)',' LinAccY (g)',' LinAccZ (g)']]

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
    
    r = R.from_quat([writer[1][0][3],writer[1][0][4],writer[1][0][5],writer[1][0][6]])
    print(writer[1][0][3],writer[1][0][4],writer[1][0][5],writer[1][0][6])
    print(r.as_matrix())
    for i in range(len(writer)):
        for j in range(len(writer[i])):
                temp = R.from_quat([writer[i][j][3],writer[0][i][4],writer[0][i][5],writer[0][i][6]])
                temp = temp.as_matrix()
                temp = temp.flatten()
                for k in range(len(temp)):
                    writer[i][j].append(temp[k])
    
    write_csv(writer,savedir)


main()
# print(df[['SensorId', ' TimeStamp (s)', ' FrameNumber',' QuatW', ' QuatX', ' QuatY',' QuatZ']])
