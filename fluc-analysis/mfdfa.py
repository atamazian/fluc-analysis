#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 by Araik Tamazian

import numpy as np

'''
    This function is used for fluctuation function estimation of the time series based on J.W. Kantelhardt's paper (2002)
        "Multifractal of detrended fluctuation analysis of non stationary time series"
    Input parameters:   1, y: Time series
                        2, scale:  List or array variable define the sample sizes of the non-overlapping segments
                                for instance: scale = np.logspace(np.log10(10**1),np.log10(10**5),30)
                        3, q: order statistical moment
                        4, m: Polynomial trend (m=1: linear, m=2: quadratic, m=3: cubic)
                        By default, q=2, m=1
    Output:             1, scale: similar to input scale. It is a list (or array) variable
                        2, F: The fluctuation function. It is a list variable
'''

def mfdfa(y, scale, q=2, m=1):
    n = y.shape[0]
    nscale = scale.shape[0]
    y = np.cumsum(y-np.mean(y))                           
    
    def seg_calc(i):
        ns = int(np.floor(n/scale[i]))         #number of segments: Ns = int(N/s)
		
        ind = np.arange(ns*scale[i])
        index = np.split(ind, ns)
        yv = np.split(y[ind], ns)
        RMSt = np.empty(ns)
        
        def seg_fit(v):
            c = np.polyfit(index[v],yv[v],m)
            fit = np.polyval(c,index[v])
            return np.sqrt(np.mean(np.power(yv[v]-fit, 2)))            

        vfunc = np.vectorize(seg_fit)
        RMSt = vfunc(range(ns))     
        qRMS = np.power(RMSt, q)
        return np.power(np.mean(qRMS), 1.0/q)

    vfunc2 = np.vectorize(seg_calc)
    F = vfunc2(range(nscale))
    return (scale,F)
 
'''
    This function is used for estimation of Hust exponent
    Input parameters:   1, scale:  List or array variable define the sample sizes of the non-overlapping segments
                                for instance: scale = np.logspace(np.log10(10**1),np.log10(10**5),30)
                        2, F: Flunctuation function values array
    Output:             1, Hurst exponent
'''
 
 def getHurst(scale, F):
    return np.polyfit(np.log10(scale),np.log10(F), 1)[0]

