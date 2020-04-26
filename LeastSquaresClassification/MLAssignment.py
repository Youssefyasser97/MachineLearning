# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:54:50 2019

@author: w
"""
from PIL import Image
import numpy as np
from numpy.linalg import inv




Xdelta=np.zeros([2400,785])
x = [ i for i in range(1,2401)]
x1=np.ones([2400])
filelist =list(map(str,x))
t0=np.zeros([2400])

t1=np.zeros([2400])
t2=np.zeros([2400])
t3=np.zeros([2400])
t4=np.zeros([2400])
t5=np.zeros([2400])
t6=np.zeros([2400])
t7 =np.zeros([2400])
t8=np.zeros([2400])
t9=np.zeros([2400])
for j in range(0,2400):
    for imagefile in filelist:
        im=Image.open("F:\Semester 9\ML\Assignment 1_30476\Train/"+imagefile+".JPG")
        arrayimage =im.convert("L")
        imgarray = np.array(arrayimage)
        imgarray=imgarray.reshape(28*28)
        Xdelta[j][0:784]=imgarray
        Xdelta[j][784]=1


t0[0:240]=1
t0[240:2400]=-1

t1[0:240]=-1
t1[240:480]=1
t1[480:2400]=-1

t2[0:480]=-1
t2[480:720]=1
t2[720:2400]=-1

t3[0:720]=-1
t3[720:960]=1
t3[960:2400]=-1

t4[0:960]=-1
t4[960:1200]=1
t4[1200:2400]=-1



t5[0:1200]=-1
t5[1200:1440]=1
t5[1440:2400]=-1

t6[0:1440]=-1
t6[1440:1680]=1
t6[1680:2400]=-1

t7[0:1680]=-1
t7[1680:1920]=1
t7[1920:2400]=-1

t8[0:1920]=-1
t8[1920:2160]=1
t8[2160:2400]=-1

t9[0:2160]=-1
t9[2160:2400]=1

XdeltaT=np.transpose(Xdelta)

mult=np.zeros([2400,2400])
for i in range(len(XdeltaT)):
   for j in range(len(Xdelta[0])):
       for k in range(len(Xdelta)):
           mult[i][j] += XdeltaT[i][k] * Xdelta[k][j]

def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

multinv=inv(mult)
print(multinv.shape)

finalmult=np.zeros([2400,2400])
for i1 in range(len(multinv)):
   for j1 in range(len(XdeltaT[0])):
       for k1 in range(len(XdeltaT)):
           finalmult[i1][j1] += multinv[i1][k1] * XdeltaT[k1][j1]

W0=finalmult.dot(t0)
W1=finalmult.dot(t1)
W2=finalmult.dot(t2)
W3=finalmult.dot(t3)
W4=finalmult.dot(t4)
W5=finalmult.dot(t5)
W6=finalmult.dot(t6)
W7=finalmult.dot(t7)
W8=finalmult.dot(t8)
W9=finalmult.dot(t9)


print(W9.shape)

print(sum(W9))
