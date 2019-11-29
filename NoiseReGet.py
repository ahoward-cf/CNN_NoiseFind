#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:27:00 2019

@author: c1649794
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

plt.style.use('/home/phoenixdata/c1649794/Python/alexDefault.mplstyle')

root = '/home/user/c1649794/HDD/Python/NoiseFind/Noisy/'
ntiles = 22000
sideLen = 500
offset = 0000
wvln = np.array((70,160,250,350,500))

maxArray = np.empty((len(wvln),ntiles))
for i in range(ntiles):
    for l in range(len(wvln)):
        w = wvln[l]
        maxArray[l,i] = np.nanmax(fits.getheader(root + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset))['NOISE'])
        
np.savetxt('/home/user/c1649794/HDD/Python/NoiseFind/Noises.txt',maxArray)