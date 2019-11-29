#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:33:02 2019

@author: c1649794
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from gaussian_random_field import scalar_grf
from copy import deepcopy

plt.style.use('/home/phoenixdata/c1649794/Python/alexDefault.mplstyle')

root = '/home/user/c1649794/HDD/Python/NoiseFind/Fields/'
ntiles = 22000
sideLen = 500
offset = 0000

for i in range(ntiles):
    hdr = fits.Header()
    hdr['NAXIS1'] = sideLen
    hdr['NAXIS2'] = sideLen
    hdr['CRPIX1'] = (sideLen-1)/2.
    hdr['CRPIX2'] = (sideLen-1)/2.
    hdr['CRVAL1'] = 0.0
    hdr['CRVAL2'] = 0.0
    hdr['CDELT1'] = -1./3600.
    hdr['CDELT2'] = 1./3600
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    
    fieldBase = scalar_grf((sideLen,sideLen),4)
    fieldCden = deepcopy(fieldBase)
    fieldTemp = deepcopy(fieldBase)
    
    fieldCden.normalise(0.5, exponentiate=True, exp_base = 10.)
    Cden = fieldCden.signal.real * 5.0
    hdr['BUNIT'] = '10^20 cm^-20'
    fits.writeto(root + 'Cden/Cden_{:04d}.fits'.format(i+1+offset), Cden, header = hdr, overwrite = True)
    
    fieldTemp.normalise(0.2)
    Temp = fieldTemp.signal.real + 1.
    Temp = 1. / Temp
    Temp += (1. - Temp.min())
    Temp *= 10.
    hdr['BUNIT'] = 'K'
    fits.writeto(root + 'Temp/Temp_{:04d}.fits'.format(i+1+offset), Temp, header = hdr, overwrite = True)


