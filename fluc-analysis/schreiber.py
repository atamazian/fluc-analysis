#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019 by Araik Tamazian

# Generates surrogate data for non-lynear analysis
#
# Uses algorithm from Schreiber, T., & Schmitz, A. (1996). 
# Improved surrogate data for nonlinearity tests. Physical Review Letters, 77(4), 635.

import numpy as np
import scipy.special
import cmath
from mfdfa import mfdfa, get_hurst, create_logscale
from fgnoise import fgnoise

def schreiber_schmitz(x, h, htol, maxitr):
    itr = 0
    h1 = h
    he = 0
    n = len(x)
    x = np.sort(x)
    scale = create_logscale(1, n/4, 100)
    while((he < h-htol)|(he > h+htol))&(itr < maxitr):
        y = fgnoise(n, h1)
        p = sorted(range(len(y)),key=lambda x:y[x])
        y[p] = x
        he = get_hurst(scale, mfdfa(y, scale, 2, 1)[1])
        h1 = h1 + (h1 - he)
        itr = itr +1
    return y, he, itr
