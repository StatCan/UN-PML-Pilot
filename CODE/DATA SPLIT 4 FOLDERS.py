#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


for fold in [0,1,2,3]:
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
        


# In[3]:


def createFolds(Y,filename,root,split):
    Xtable=pd.read_csv(root+"/"+filename,sep='\s+',header=None)
    XY=Y.join(Xtable)
    
    for fold in [0,1,2,3]:
        fileReplaced=filename.replace(".txt","")
        fileOut=str(fold)+"/"+split+"/"+fileReplaced+"_"+str(fold)+".csv"
        appo=XY[XY["FOLD"]==fold]
        appo.sort_index().to_csv(fileOut,index=True,sep=";")
        print(fileOut)


# In[4]:



for split in ["test","train"]:
    root="./UCI HAR Dataset/UCI HAR Dataset/"+split+"/"
    Y=pd.read_csv(root+"y_"+split+".txt",header=None)
    Y.columns=["Y"]
    Y.sort_values("Y",inplace=True)
    L=len(Y)
    print("len:",L)
    Y["FOLD"]=(np.arange(L)%4)

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


# In[5]:


for fold in [0,1,2,3]:
    for split in ["test","train"]:  
        direc=str(fold)+'/'+split
        print(direc)
        old_file = os.path.join(direc, "X_"+split+"_"+str(fold)+'.csv')
        new_file = os.path.join(direc, 'ALL_'+split+'.csv')
        os.rename(old_file,new_file)
        


# In[118]:





# In[128]:





# In[ ]:




