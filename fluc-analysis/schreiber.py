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
from mfdfa import mfdfa
from fgnoise import fgnoise

def schriber(x, h, htol, maxitr):
    itr = 0
    h1 = h
    he = 0
    n = len(x)
    x = np.sort(x)
    scale = np.logspace(np.log10(10**1),np.log10(n/4.0),10)
    scale = scale.astype(int)
    while((he < h-htol)|(he > h+htol))&(itr < maxitr):
        y = fgnoise(n, h1)
        p = sorted(range(len(y)),key=lambda x:y[x])
        y[p] = x
        he = mfdfa(y, scale, 2, 1)[0]
        h1 = h1 + (h1 - he)
        itr = itr +1
    return y, he, itr
