#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:05:50 2022

@author: vikas
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from scipy import signal as sig
import variable as vars
import queue

q=queue.Queue()

def getThres(T, thres0):
    n_sec=4
    T=T.astype(float)
    T=sig.medfilt(T,3601)
    scaler_thres=MinMaxScaler()
    scaler_thres.fit(T.reshape(-1, 1))
    T=scaler_thres.transform(T.reshape(-1, 1))
    len_T=len(T)
    len_section=int(len_T/n_sec)
    means=np.zeros(n_sec, dtype=float)
    for i in range(n_sec):
        means[i]=np.mean(T[i*len_section:(i+1)*len_section])

    shiftby=np.std(means)*2
    return thres0 + shiftby

def getLenghtOfAttack(cls_new,i):
    len=0
    while cls_new[i-1]!='Normal':
        i=i-1
        len=len+1
    return len


def improveSample(sampled,t, h, thres1, Score, lag, cls_new):
    if t-q.queue[0][0]<vars.TIME_TO_CONFERM:
        return sampled

    q_x=q.get()
    i=q_x[0]
    lenAtt=q_x[3]
    if (('Attack') in cls_new[i-min(int(lenAtt), lag*3)-lag:i]) and (Score[i-min(int(lenAtt), lag*2)-lag:i]>thres1).any():
        return sampled
    
    x=np.array([q_x[1], q_x[2]])
    kdf_=kdf(sampled, h, x)
    if kdf_>thres1:
        sampled=np.vstack((sampled,x))
    return sampled



def getAnomalyScore(T, sampled, lag, scaler, cls_new, h, thres1): 
    score=np.zeros(len(T)-lag+1, dtype=float)
    x=np.zeros(2,dtype=float)
    q.queue.clear()
    lenAtt=0
    for i in range(lag-1,len(T)):
        seq=T[i-lag+1:i+1]
        x[0]=np.mean(seq)
        x[1]=np.std(seq)
        x=scaler.transform(x.reshape(1,-1))[0]
        score[i-lag+1]=kdf(sampled, h, x)

        if cls_new[i-lag+1]=='Normal' and cls_new[i-lag+1]=='Attack':
            lenAtt=getLenghtOfAttack(cls_new,i)
        if score[i-lag+1]>thres1:
            q.put([i-lag+1,x[0],x[1], lenAtt])
        if not q.empty():
            sampled=improveSample(sampled,i-lag+1, h, thres1, score, lag, cls_new)
    print('feature space size at end: '+str(len(sampled)))
    return score


def sampling(X,sr):
    len_X=len(X)
    index = np.random.choice(len_X, int(len_X*sr))
    sampled=X[index]
    print('feature space size at begining: '+str(len(sampled)))
    return sampled


def nearestPoints(X, x, N):
    X_=(X-x)**2
    X_=X_.T
    series=(X_[0]+X_[1])**(1/2)
    indx=series.argsort()
    return X[indx]


def kdf(X_, h, x):
    s=0
    K=vars.k
    e=math.e
    pi=math.pi
    X=nearestPoints(X_,x,K)
    for i in range(K):
        a=X[i][0]-x[0]
        b=X[i][1]-x[1]
        s=s+(e**(-(a**2+b**2)/(2*h)))/(2*pi*h)
    s = vars.MNIMUM_SCORE-s/K
    return s


def Gaussian2d(x, h):
    e=math.e
    pi=math.pi
    a=x[0]
    b=x[1]
    return (e**(-(a**2+b**2)/(2*h)))/(2*pi*h)
