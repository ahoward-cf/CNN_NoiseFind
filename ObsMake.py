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
fields = 'Fields/'
obs    = 'Obs/'

ntiles = 22000
sideLen = 500
h2tog = 2.8 * 1.6735575E-24
offset = 0000

wvln = np.array((70,160,250,350,500))
ares = [9.25925926E-4, 9.25925926E-4, 5.555555556E-4, 9.25925926E-4, 0.001296296296]

def kappa (wvln):
    return 0.1 * (wvln / 300.)**-2.

for i in range(ntiles):
    head = fits.getheader(root + fields + 'Cden/Cden_{:04d}.fits'.format(i+1+offset))
    cden = fits.getdata(root + fields + 'Cden/Cden_{:04d}.fits'.format(i+1+offset)) * 1.0E20
    temp = fits.getdata(root + fields + 'Temp/Temp_{:04d}.fits'.format(i+1+offset))
    for l in range(len(wvln)):
        w = wvln[l]
        beam = fits.getdata(root + obs + 'psf/psf_{:04d}.fits'.format(w))
        k = kappa(w)
        I =  cden * k * h2tog * BB(w * 1E-6 * u.m, temp * u.K).value * 1E17
        Ic = convolve(I, beam, normalize_kernel=True, allow_huge = True)
        headNew = head.copy()
        headNew['NAXIS1'] = 100
        headNew['NAXIS2'] = 100
        headNew['CRPIX1'] = (50-1)/2.
        headNew['CRPIX2'] = (50-1)/2.
        headNew['CDELT1'] = -ares[l]
        headNew['CDELT2'] = ares[l]
        headNew['BUNIT']  = 'MJy/sr'
        Ir = reproject_exact((I,head), headNew, parallel=False)[0]
        outStr = root + obs + '{:04d}/'.format(w) + '{:04d}'.format(w) + '_{:04d}.fits'.format(i+1+offset)
        fits.writeto(outStr, Ir, header = headNew, overwrite = True)
        