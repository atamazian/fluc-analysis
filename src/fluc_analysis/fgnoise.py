#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019 by Araik Tamazian

import numpy as np
import scipy.special
import cmath

def fgnoise(n, h):
    w = np.linspace(2*np.pi/n, np.pi, int(n/2-1))
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

    z = np.zeros(n//2, dtype=complex)
    zph = np.random.uniform(0, 2*np.pi, n//2-1)
    z = np.vectorize(cmath.rect)(sw**0.5, zph)

    z1 = np.zeros(n, dtype=complex)
    z1[0] = 0
    z1[1:n//2] = z
    z1[n//2+1:n] = np.conj(z)

    x = np.fft.ifft(z1).real
    return x
