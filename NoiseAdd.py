#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:57:30 2019

@author: c1649794
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling.blackbody import blackbody_nu as BB
import astropy.units as u
from astropy.convolution import convolve_fft as convolve
from reproject import reproject_exact

plt.style.use('/home/phoenixdata/c1649794/Python/alexDefault.mplstyle')

root = '/home/user/c1649794/HDD/Python/NoiseFind/'
nDir = 'Noisy/'
obs    = 'Obs/'

ntiles = 22000
offset = 0

wvln = np.array((70,160,250,350,500))

maxSNR = 500
minSNR = 100

maxArray = np.empty((len(wvln),ntiles))

for i in range(ntiles):
    for l in range(len(wvln)):
   
        w = wvln[l]
        maxArray[l,i] = np.nanmax(fits.getdata(root + obs + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset)))        

maxMinSig = np.empty((5,2))
for l in range(len(wvln)):
    maxMinSig[l,0] = np.nanmax(maxArray[l,:]) / minSNR
    maxMinSig[l,1] = np.nanmin(maxArray[l,:]) / maxSNR       

noiseArray = np.empty((ntiles,5))
        
for i in range(ntiles):
    for l in range(len(wvln)):                
        w = wvln[l]
        data = fits.getdata(root + obs + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset))        
        head = fits.getheader(root + obs + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset))
        nLevel = np.random.uniform(maxMinSig[l,1], maxMinSig[l,0])
        noiseArray[i,l] = nLevel
        noise = np.random.normal(loc = 0., scale = nLevel, size = data.shape)
        newData = data.copy() + noise.copy()
        newHead = head.copy()
        newHead['PKSNR'] = np.nanmax(data) / nLevel
        newHead['NOISE'] = nLevel
        outStr = root + nDir + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset)
        fits.writeto(outStr, newData, header = newHead, overwrite = True)
np.savetxt(root + 'Noises.txt', noiseArray)
        