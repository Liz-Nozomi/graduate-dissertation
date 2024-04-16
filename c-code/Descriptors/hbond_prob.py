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
##################################################################################################################################

XTC='mcm2nm-mescnmi-prodrun-40ns-r1.xtc'
GRO='mcm2nm-mescnmi-prodrun-40ns-r1.gro'
u = MDAnalysis.Universe(GRO, XTC)

##################################################################################################################################

#define the accepted atom of -OH Or CH as Ha and Oa; the donated atoms of CN as Nd and Cd
def HBonds_SCN(Ha_atom,Oa_atom,Nd_atom,Cd_atom):   
    n_Ha = len(Ha_atom)
    n_Nd = len(Nd_atom)
    #denfine the number of axial and total HBs
    A_HB = 0                        
    T_HB = 0                    
    for j in range (n_Ha):       
        for i in range (n_Nd):           
            HN_vector= Ha_atom.positions[j] - Nd_atom.positions[i]  
            HN_distance= np.linalg.norm(HN_vector) 
            ON_vector=Oa_atom.positions[j] - Nd_atom.positions[i]
            ON_distance= np.linalg.norm(ON_vector)        
            if HN_distance <= 3.6:
                theta_HNO = np.arccos(np.dot(HN_vector,ON_vector)/(np.linalg.norm(HN_vector)*np.linalg.norm(ON_vector)))
                angle_HNO=np.rad2deg(theta_HNO)                            
                if angle_HNO <=30:  
                    T_HB=T_HB + 1                                      #record the total number 
                    CN_vector=Cd_atom.positions[i] - Nd_atom.positions[i]
                    theta_CNH = np.arccos(np.dot(HN_vector,CN_vector)/(np.linalg.norm(HN_vector)*np.linalg.norm(CN_vector)))
                    angle_CNH=np.rad2deg(theta_CNH)                          
                    if angle_CNH > 120:                                      
                        A_HB=A_HB + 1
    return A_HB,T_HB

#################################################################################################################################

A_HBs_CR_HN_SCN=[]      
T_HBs_CR_HN_SCN=[]

A_HBs_CW1_HN_SCN=[]
T_HBs_CW1_HN_SCN=[]

A_HBs_CW2_HN_SCN=[]      
T_HBs_CW2_HN_SCN=[]

A_HBs_OH_HN_SCN=[]      
T_HBs_OH_HN_SCN=[]



T_HBs_CR_HN_MIM=[]
T_HBs_CW1_HN_MIM=[]     
T_HBs_CW2_HN_MIM=[]     
T_HBs_OH_HN_MIM=[]

for ts in u.trajectory[1000:2010]:

    #MCM41 HO OH
    mcm_H=u.select_atoms('resname MCM and name Ho')
    mcm_O=u.select_atoms('resname MCM and name Oh')

    #Methylimidazolium CR HR(H5)
    mim_CR=u.select_atoms('resname MIM and name CR5')
    mim_HR=u.select_atoms('resname MIM and name HR9')

    #Methylimidazolium CW HW(H4) 
    mim_CW1=u.select_atoms('resname MIM and name CW2')  
    mim_HW1=u.select_atoms('resname MIM and name HW7')
    mim_CW2=u.select_atoms('resname MIM and name CW3')
    mim_HW2=u.select_atoms('resname MIM and name HW8')

    #Methylimidazolium N2 
    mim_N2=u.select_atoms('resname MIM and name NA4')

    #MethylSCN NC CN
    mes_NC=u.select_atoms('resname MES and name NC4')
    mes_CN=u.select_atoms('resname MES and name CN3')

###########################################################################

####The HBonds relative to the MeSCN with MCM-41 and MIM #################

    #Calculate the HBonds between CR-HR-NC
    A_HB_CR_HN,T_HB_CR_HN =HBonds_SCN(mim_HR,mim_CR,mes_NC,mes_CN)

    A_HBs_CR_HN_SCN.append(A_HB_CR_HN)
    T_HBs_CR_HN_SCN.append(T_HB_CR_HN)

    #Calculate the HBonds between CW1-HW1-NC
    A_HB_CW1_HN,T_HB_CW1_HN =HBonds_SCN(mim_HW1,mim_CW1,mes_NC,mes_CN)

    A_HBs_CW1_HN_SCN.append(A_HB_CW1_HN)
    T_HBs_CW1_HN_SCN.append(T_HB_CW1_HN)

    #Calculate the HBonds between CW2-HW2-NC
    A_HB_CW2_HN,T_HB_CW2_HN =HBonds_SCN(mim_HW2,mim_CW2,mes_NC,mes_CN)

    A_HBs_CW2_HN_SCN.append(A_HB_CW2_HN)
    T_HBs_CW2_HN_SCN.append(T_HB_CW2_HN)


    #Calculate the HBonds between OH of MCM41 and MeSCN: OH-HO-NC 
    A_HB_OH_HN,T_HB_OH_HN =HBonds_SCN(mcm_H,mcm_O,mes_NC,mes_CN)

    A_HBs_OH_HN_SCN.append(A_HB_OH_HN)
    T_HBs_OH_HN_SCN.append(T_HB_OH_HN)

'''
##############################################################################


####The HBonds relative to the MIM with MCM-41 and MIM #################

    #Calculate the HBonds between CR-HR-N
    T_HB_CR_HN =HBonds_MIM(mim_HR,mim_CR,mim_N2)

    T_HBs_CR_HN_MIM.append(T_HB_CR_HN)

    #Calculate the HBonds between CW1-HW1-N
    T_HB_CW1_HN =HBonds_MIM(mim_HW1,mim_CW1,mim_N2)

    T_HBs_CW1_HN_MIM.append(T_HB_CW1_HN)

    #Calculate the HBonds between CW2-HW2-N
    T_HB_CW2_HN =HBonds_MIM(mim_HW2,mim_CW2,mim_N2)

    T_HBs_CW2_HN_MIM.append(T_HB_CW2_HN)


    #Calculate the HBonds between OH of MCM41 and MeSCN: OH-HO-N 
    T_HB_OH_HN =HBonds_MIM(mcm_H,mcm_O,mim_N2)

    T_HBs_OH_HN_MIM.append(T_HB_OH_HN)

##############################################################################
'''


##############################################################################

#define the function to calculate the probility of the hbonds 
def hbond_probality(x):
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    counts_0 = []
    counts_1 = []
    counts_2 = []
    counts_3 = []
    counts_4 = []
    counts_5 = []
    counts_6 = []
    counts_7 = []
    counts_8 = []

    for i in range (len(x)):   
        if x[i] == 0:
            count_0 = count_0 + 1
        elif x[i] == 1:
            count_1 = count_1 + 1
        elif x[i] == 2:
            count_2 = count_2 + 1
        elif x[i] == 3:
            count_3 = count_3 + 1
        elif x[i] == 4:
            count_4 = count_4 + 1
        elif x[i] == 5:
            count_5 = count_5 + 1
        elif x[i] == 6:
            count_6 = count_6 + 1
        elif x[i] == 7:
            count_7 = count_7 + 1
        else:
            count_8 = count_8 + 1

    counts_0.append(count_0)
    counts_1.append(count_1)
    counts_2.append(count_2)
    counts_3.append(count_3)
    counts_4.append(count_4)
    counts_5.append(count_5)
    counts_6.append(count_6)
    counts_7.append(count_7)
    counts_8.append(count_8)
    
    hbond_sum = counts_0 + counts_1 + counts_2 + counts_3 + counts_4 + counts_5 + counts_6 + counts_7 + counts_8    
    return hbond_sum

###############################################################################################################

#calculate the total of the hbonds very frame

Frames =len(T_HBs_CR_HN_SCN)
T_HBs_SCN_Total=np.zeros(Frames)
A_HBs_SCN_Total=np.zeros(Frames)

for i in range(Frames):

    T_HBs_SCN_Total[i]=T_HBs_CR_HN_SCN[i] + T_HBs_CW1_HN_SCN[i] + T_HBs_CW2_HN_SCN[i] + T_HBs_OH_HN_SCN[i]   
    A_HBs_SCN_Total[i]=A_HBs_CR_HN_SCN[i] + A_HBs_CW1_HN_SCN[i] + A_HBs_CW2_HN_SCN[i] + A_HBs_OH_HN_SCN[i]


##############################################################################################################

#calculate the probility of the hbonds based on type of Hydrogen

#CR-HR--NC
T_HBs_CR_HN_SCN_Counts= hbond_probality(T_HBs_CR_HN_SCN)
A_HBs_CR_HN_SCN_Counts= hbond_probality(A_HBs_CR_HN_SCN)


#CW1-HW1--NC
T_HBs_CW1_HN_SCN_Counts= hbond_probality(T_HBs_CW1_HN_SCN)
A_HBs_CW1_HN_SCN_Counts= hbond_probality(A_HBs_CW1_HN_SCN)

#CW2-HW2--NC
T_HBs_CW2_HN_SCN_Counts= hbond_probality(T_HBs_CW2_HN_SCN)
A_HBs_CW2_HN_SCN_Counts= hbond_probality(A_HBs_CW2_HN_SCN)

#Oh-Ho--NC
T_HBs_OH_HN_SCN_Counts= hbond_probality(T_HBs_OH_HN_SCN)
A_HBs_OH_HN_SCN_Counts= hbond_probality(A_HBs_OH_HN_SCN)

#the total hbonds of mescn
T_HBs_SCN_Total_Counts= hbond_probality(T_HBs_SCN_Total)
A_HBs_SCN_Total_Counts= hbond_probality(A_HBs_SCN_Total)
 

print(T_HBs_CW1_HN_SCN_Counts)
print(A_HBs_CW1_HN_SCN_Counts)
################################################################################################################

n_hb_probility=len(T_HBs_SCN_Total_Counts)
with open('mcm2nm_hbonds-prob-SCN.txt','w') as f:       #remember to change the file names

    # save total hbonds
    f.write("T_HBs_SCN_Total_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(T_HBs_SCN_Total_Counts[item]))
        f.write("  ")
    f.write("\n") 
    
    f.write("T_HBs_CR_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(T_HBs_CR_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")
   
    f.write("T_HBs_CW1_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(T_HBs_CW1_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")

    f.write("T_HBs_CW2_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(T_HBs_CW2_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")
  
    f.write("T_HBs_OH_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(T_HBs_OH_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")  
    f.write("\n")
    f.write("\n")


    # save axial hbonds
    f.write("A_HBs_SCN_Total_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(A_HBs_SCN_Total_Counts[item]))
        f.write("  ")
    f.write("\n") 

    f.write("A_HBs_CR_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(A_HBs_CR_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n") 
    
    f.write("A_HBs_CW1_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(A_HBs_CW1_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")
 
    f.write("A_HBs_CW2_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(A_HBs_CW2_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n")
    
    f.write("A_HBs_OH_HN_SCN_Counts")
    for item in range(n_hb_probility):
        f.write("  ")        
        f.write(str(A_HBs_OH_HN_SCN_Counts[item]))
        f.write("  ")
    f.write("\n") 

