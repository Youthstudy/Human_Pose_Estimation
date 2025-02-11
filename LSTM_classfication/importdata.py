import os
import  pandas as pd
import numpy as np
filePath = "E:\\2.job\\2.wendu\\dataset"
a = []
b = []
c = []
for i,j,k in os.walk(filePath):
    # print(i,j,k)
    a.append(i)
    b.append(j)
    c.append(k)


needroot = []
for i in range(2,len(a)):
    for j in range(len(c[i])):
        needroot.append(a[i] + "\\" + c[i][j])

a = []
for i in range(len(needroot)):
    b = pd.read_csv(needroot[i])
    a.append(b[['VALUE']].values)

data = []
for i in range(len(a[0])):
    data.append([])
    for j in range(len(a)):
        data[i].append(float(a[j][i]))


print(np.array(data))