#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 by Araik Tamazian

# Generates surrogate data for non-lynear analysis
#
# Uses algorithm from Schreiber, T., & Schmitz, A. (1996). 
# Improved surrogate data for nonlinearity tests. Physical Review Letters, 77(4), 635.

import numpy as np
import scipy.special
import cmath


def mfdfa(x, q, m, sfmax):
    '''
    Function MF-DFA is used for estimating Hurst parameter of time series based on J.W. Kantelhardt's paper (2002)
        "Multifractal of detrended fluctuation analysis of non stationary time series"
    Input parameters:   1, indata: Time series
                        2, scale:  List or array variable define the sample sizes of the non-overlapping segments
                                for instance: scale = np.logspace(np.log10(10**1),np.log10(10**5),30)
                        3, q: order statistical moment
                        4, m: Polynomial trend (m=1: linear, m=2: quadratic, m=3: cubic)
                        Usually, q=2, m=1
    Output:             1, H: Hurst parameters. It is a float variable
                        2, scale: similar to input scale. It is a list (or array) variable
                        3, F: The fluctuation function. It is a list variable
    '''
    N = len(x)
    scale = np.logspace(np.log10(10**1),np.log10(N/4.0),10)
    scale = scale.astype(int)
    y = np.cumsum(x-x.mean())                       #Equation 1 in paper
    RMSt = []                                       #Temporary RMS variable: contain F(s,v) value
    F = []                                          #F: Fluctuation function
    for i in range(len(scale)):
        ns = int(np.floor(len(y)/scale[i]))         #number of segments: Ns = int(N/s)
        for v in range(ns):
            index_start = v*scale[i]
            index_end = (v+1)*scale[i]
            index = range(index_start,index_end)    #calculate index for each segment
            yv = y[index_start:index_end]           #Extract values of time series for each segments
            c = np.polyfit(index,yv,m)
            fit = np.polyval(c,index)
            RMSt.append(np.sqrt(np.mean((yv-fit)**2))) #Equation 2. But calculating only F(v,s) not F(v,s)**2
        RMS = np.asarray(RMSt)                      #Convert RMSt to array
        qRMS = RMS**q
        F.append(np.mean(qRMS)**(1.0/q))              #Equation 4
        del RMSt[:]                                 #Reset RMSt[:]
    scale = np.asarray(scale)
    F = np.asarray(F)
    C = np.polyfit(np.log10(scale[scale < sfmax]),np.log10(F[scale < sfmax]),1)
    H = C[0]                                        #Hurst parameter
    return(H, scale, F)


def fgnoise(n, h):
    w = np.linspace(2*np.pi/n, np.pi, n/2-1)
    aw = 2*np.sin(np.pi*h)*scipy.special.gamma(2*h+1)*(1-np.cos(w))
    d = -2*h-1
    d1 = -2*h

    a1 = 2*np.pi+w
    a2 = 4*np.pi+w
    a3 = 6*np.pi+w
    a4 = 8*np.pi+w

    b1 = 2*np.pi-w
    b2 = 4*np.pi-w
    b3 = 6*np.pi-w
    b4 = 8*np.pi-w

    bw1 = (a3**d1 + b3**d1 + a4**d1 + b4**d1)/(8*h*np.pi)
    bw = a1**d + b1**d + a2**d + b2**d + a3**d + b3**d + bw1
    sw = aw*(np.abs(w)**(-2*h-1) + bw)

    z = np.zeros(n/2, dtype=complex)
    zph = np.random.uniform(0, 2*np.pi, n/2-1)
    z = np.vectorize(cmath.rect)(sw**0.5, zph)

    z1 = np.zeros(n, dtype=complex)
    z1[0] = 0
    z1[1:n/2] = z
    z1[n/2+1:n] = np.conj(z)

    x = np.fft.ifft(z1).real
    return x


def schriber(x, h, htol, maxitr):
    itr = 0
    h1 = h
    he = 0
    n = len(x)
    x = np.sort(x)
    while((he < h-htol)|(he > h+htol))&(itr < maxitr):
        y = fgnoise(n, h1)
        p = sorted(range(len(y)),key=lambda x:y[x])
        y[p] = x
        he = mfdfa(y,2,2, 10000)[0]
        h1 = h1 + (h1 - he)
        itr = itr +1
    return y, he, itr
