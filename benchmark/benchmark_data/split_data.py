#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

NUMBER_OF_FOLDERS = 10

def main():
    for fold in range(NUMBER_OF_FOLDERS):
        for split in ["test","train"]:    
            try:
                os.mkdir(str(fold))
            except:
                pass
            try:
                os.mkdir(str(fold)+"/"+split+"/")
            except:
                pass
            try:
                os.mkdir(str(fold)+"/"+split+"/Inertial Signals")
            except:
                pass

    for split in ["test","train"]:
        root="./UCI_HAR_Dataset/"+split+"/"
        Y=pd.read_csv(root+"y_"+split+".txt",header=None)
        Y.columns=["Y"]
        Y.sort_values("Y",inplace=True)
        L=len(Y)
        print("len:",L)
        Y["FOLD"]=(np.arange(L)%NUMBER_OF_FOLDERS)

        tag=""
        for signal in ["X_"+split+".txt",
                    "Inertial Signals/body_acc_x_"+split+".txt",
                    "Inertial Signals/body_acc_y_"+split+".txt",
                    "Inertial Signals/body_acc_z_"+split+".txt",
                    "Inertial Signals/total_acc_x_"+split+".txt",
                    "Inertial Signals/total_acc_y_"+split+".txt",
                    "Inertial Signals/total_acc_z_"+split+".txt",
                    "Inertial Signals/body_gyro_x_"+split+".txt",
                    "Inertial Signals/body_gyro_y_"+split+".txt",
                    "Inertial Signals/body_gyro_z_"+split+".txt"]:
            filename=""+signal
            createFolds(Y,filename,root,split)        

    for fold in range(NUMBER_OF_FOLDERS):
        for split in ["test","train"]:  
            direc=str(fold)+'/'+split
            old_file = os.path.join(direc, "X_"+split+"_"+str(fold)+'.csv')
            new_file = os.path.join(direc, 'ALL_'+split+'.csv')
            os.rename(old_file,new_file)
    

def createFolds(Y,filename,root,split):
    Xtable=pd.read_csv(root+"/"+filename,sep='\s+',header=None)
    XY=Y.join(Xtable)
    
    for fold in range(NUMBER_OF_FOLDERS):
        fileReplaced=filename.replace(".txt","")
        fileOut=str(fold)+"/"+split+"/"+fileReplaced+"_"+str(fold)+".csv"
        appo=XY[XY["FOLD"]==fold]
        appo.sort_index().to_csv(fileOut,index=True,sep=";")
        print(fileOut)

if __name__=='__main__':
    main()
