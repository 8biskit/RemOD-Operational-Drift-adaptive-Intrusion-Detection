#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:25:48 2022

@author: vikas
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import variable as vars
import utils


parser = argparse.ArgumentParser(description='get alarm score for normal and attack sequences',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset','-d', type=str, default='SWaT',
                    help='Select Dataset')

args = parser.parse_args()
dataset = args.dataset

lag=vars.lag
sr=vars.sr
h=vars.h
SMALL_SAMPLE_SIZE=vars.k
TIME_TO_CONFERM=vars.TIME_TO_CONFERM


normal = pd.read_excel('data/SWaT_Dataset_Normal_v0.xlsx').values[:,1:-1]
attack_ = pd.read_excel('data/SWaT_Dataset_Attack_v0.xlsx').values
attack = attack_[:,1:-1] #Exclude data-time and label column
cls_ = attack_[:,-1]
tr = vars.tr

MNIMUM_SCORE=vars.MNIMUM_SCORE
AllFlags=np.empty((normal.shape[1]),dtype=np)

    
for i in range(normal.shape[1]): #iterate over each sensors time series 
    print(str(i+1)+'(th) Sensor of '+dataset+' dataset')
    T_=normal[:,i] #Training on ith sensors 
    T=T_[int((len(T_)*(1-tr))):len(T_)] # TAKING FROM LAST OF NORMAL DATA WHICH IS ENOUGH FOR TRAINING INITIAL DATA IS NOT GOOD BECAUSE MACHINE IS STABLISING
    n_points=len(T)-lag+1
    thres1=MNIMUM_SCORE- utils.Gaussian2d([0.1,0],h)
    thres=utils.getThres(T, MNIMUM_SCORE- utils.Gaussian2d([0.3,0],h)) #0.4675299695567543 do by get threshold to get higher value for changing data, make generarel based on h ie segma/2 and segma, 
    X=np.zeros((n_points, 2), dtype=float) #X: sample space
    for j in range(lag-1,len(T)):
        seq=T[j-lag+1:j+1]
        X[j-lag+1][0]=np.mean(seq)
        X[j-lag+1][1]=np.std(seq)

    scaler=MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X) # normalization
    sampled=utils.sampling(X,sr) # to reduce the size


    T_att=attack[:,i] #Testing on ith sensors 
    T_att_=np.append(T,T_att) # append the training to plot them also, but excluded in avaluation 
    cls_new=np.append( np.full(len(T)-lag+1, 'Normal'), cls_) # class for T_att_ array

    anomaly_score = utils.getAnomalyScore(T_att_, sampled, lag, scaler, cls_new, h, thres1) 
    
    