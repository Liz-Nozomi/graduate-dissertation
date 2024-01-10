import MDAnalysis
import MDAnalysis.analysis.align
import sys
import re
import itertools
import os
from MDAnalysis.tests.datafiles import GRO, XTC
import numpy as np

import json
import scipy.integrate as integrate


#read the xtc and gro files
XTC='sio2-oh-c4-finalpbc.xtc'
GRO='sio2-oh-c4-finalpbc.gro'
u = MDAnalysis.Universe(GRO, XTC)



##########################################################
#calculate the non zero mean



##########################################################
#get the positions of both MIM center and Oh in all frames
iso_com_times=[]  #选取帧数
tra_com_times=[]  
for ts in u.trajectory[0:-1:2]:   #for ts in u.trajectory[0:-1:2] -1表示最后一步结束, 2代表步长, 可以不要; 当加上这个2时 下面的Ds_iso[m][j]=msd/(dt*帧数(10)*2）中帧数要对应的×2  #轨迹属性，30000为帧数，帧数与时间间隔是dt的关系 与nstxout(帧的间隔）时间间隔步长×dt  nsteps/nstxout=30000
    
    iso=u.select_atoms('resname ISO') #选中分子
    iso_com=iso.center(iso.masses,compound='residues')  #选中质心 #the com of SOL in one frames
    iso_com_times.append(iso_com)   #    the com of SOL in all frames 把所有的质心存到空的列表中 代表一帧里面所有水分子的质心[[x1,y1,z1];[]...[]]
                                                                                            #存储多组数据

    tra=u.select_atoms('resname TRA')
    tra_com=tra.center(tra.masses,compound='residues')
    tra_com_times.append(tra_com)



################################################################################
#SOL diffusive as z axis 
###########################################################################
n_frames=len(iso_com_times)  #total frames 总帧数     
n_iso=len(iso_com)           #the number of SOL 
#n_r=1600   #0.1A 为单位  代表的盒子Z方向的尺寸，1600代表16nm
n_r=160                                                                     #divive the r into 0.2A

Ds_iso_frames=[]
for i in range(0,n_frames-30):       #减去30帧的目的是防止不存在终点轨迹而出现报错  减去的帧数要大于10ps内的帧数防止溢出                                        #get the msd and D every 10 ps
    Ds_iso=np.zeros([n_iso,n_r])
    for m in range (n_iso):
        r_distance=np.sqrt(np.square(iso_com_times[i][m][0])+np.square(iso_com_times[i][m][1]))       #所在位置，这里2代表z 
        r_distance_round=round(r_distance)   #每10ps内计算一次
        #print(r_distance_round)          
        for j in range(n_r):  #判断距离在哪一个位置
            if r_distance_round==j:
                iso_com_square29=np.sqrt(np.square(iso_com_times[i+29][m][0]+iso_com_times[i+29][m][1]))
                iso_com_square5=np.sqrt(np.square(iso_com_times[i+5][m][0]+iso_com_times[i+5][m][1]))
                msd=((iso_com_square29-iso_com_square5)**2)*0.01
                Ds_iso[m][j]=msd / (0.2*2*24*4)  #0.64是dt                                      #get the Ds in this time interval

    Ds_iso_sum=np.sum(Ds_iso,axis=0)

#####################################################
# calculate the mean of non-0 value  
    for j in range (len(Ds_iso_sum)):
        if Ds_iso_sum[j] != 0:
            #print (Ds_iso_sum[j])
            num = 0
            for m in range (n_iso):               
                if Ds_iso[m][j] !=0:
                    num=num+1
            print (num)
            Ds_iso_sum[j]=Ds_iso_sum[j] / num
            print(Ds_iso_sum[j])
    Ds_iso_frames.append(Ds_iso_sum)
#####################################################

Ds_iso_frames=np.array(Ds_iso_frames)
Ds_iso_frames_sum=np.sum(Ds_iso_frames,axis=0)

#加和结束以后还有可能存在0值，因此再加和消除一遍
#####################################################
for j in range (len(Ds_iso_frames_sum)):
    if Ds_iso_frames_sum[j] != 0:
        num = 0
        for f in range (len(Ds_iso_frames)):               
            if Ds_iso_frames[f][j] !=0:
                num=num+1
        print (num)
        Ds_iso_frames_sum[j]=Ds_iso_frames_sum[j] / num
        print(Ds_iso_frames_sum[j])
#####################################################


print (Ds_iso_frames_sum)



with open('24-sio2-oh-msd-iso-xy.txt','w') as f:
    for item in range(n_r):
        f.write(str(item))
        f.write("  ")
        f.write(str(Ds_iso_frames_sum[item]))
        f.write("\n")
###########################################################################




##########################################################################
#MET diffusive as z axis 
###########################################################################
n_frames=len(tra_com_times)     
n_tra=len(tra_com)
n_r=160
                                                                      #divive the r into 0.2A
Ds_tra_frames=[]
for i in range(0,n_frames-30):                                                   #get the msd and D every 10 ps
    Ds_tra=np.zeros([n_tra,n_r])
    for m in range (n_tra):
        r_distance=np.sqrt(np.square(tra_com_times[i][m][0])+np.square(tra_com_times[i][m][1]))        
        r_distance_round=round(r_distance) 
        #print(r_distance_round)          
        for j in range(n_r):
            if r_distance_round==j:
                tra_com_square29=np.sqrt(np.square(tra_com_times[i+29][m][0]+tra_com_times[i+29][m][1]))
                tra_com_square5=np.sqrt(np.square(tra_com_times[i+5][m][0]+tra_com_times[i+5][m][1]))
                msd=((tra_com_square29-tra_com_square5)**2)*0.01    #get the msd between 0.2*6=1.2ps and 0.2*30=6ps
                Ds_tra[m][j]=msd / (0.2*2*24*2)                                     #这里错误，应该为0.2 0.8= dt×nstxout，代表一针的时间长度  #get the Ds in this time interval

    Ds_tra_sum=np.sum(Ds_tra,axis=0)

#####################################################
# calculate the mean of non-0 value  
    for j in range (len(Ds_tra_sum)):
        if Ds_tra_sum[j] != 0:
            #print (Ds_tra_sum[j])
            num = 0
            for m in range (n_tra):               
                if Ds_tra[m][j] !=0:
                    num=num+1
            print (num)
            Ds_tra_sum[j]=Ds_tra_sum[j] / num
            print(Ds_tra_sum[j])
    Ds_tra_frames.append(Ds_tra_sum)
#####################################################

Ds_tra_frames=np.array(Ds_tra_frames)
Ds_tra_frames_sum=np.sum(Ds_tra_frames,axis=0)


#####################################################
for j in range (len(Ds_tra_frames_sum)):
    if Ds_tra_frames_sum[j] != 0:
        num = 0
        for f in range (len(Ds_tra_frames)):               
            if Ds_tra_frames[f][j] !=0:
                num=num+1
        print (num)
        Ds_tra_frames_sum[j]=Ds_tra_frames_sum[j] / num
        print(Ds_tra_frames_sum[j])
#####################################################


print (Ds_tra_frames_sum)



with open('24-sio2-oh-msd-tra-xy.txt','w') as f:
    for item in range(n_r):
        f.write(str(item))
        f.write("  ")
        f.write(str(Ds_tra_frames_sum[item]))
        f.write("\n")


