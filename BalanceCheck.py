#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:06:23 2019

@author: c1649794
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

plt.style.use('/home/phoenixdata/c1649794/Python/alexDefault.mplstyle')

root = '/home/user/c1649794/HDD/Python/NoiseFind/'

noises = np.loadtxt(root + 'Noises.txt')

plt.figure()
plt.hist(noises[:,0], 100)
plt.title('Balance of Sig at 0070 microns')
plt.savefig('Balance0070.png', dpi = 150, bbox_inches = 'tight')

plt.figure()
plt.hist(noises[:,1], 100)
plt.title('Balance of Sig at 0160 microns')
plt.savefig('Balance0160.png', dpi = 150, bbox_inches = 'tight')

plt.figure()
plt.hist(noises[:,2], 100)
plt.title('Balance of Sig at 0250 microns')
plt.savefig('Balance0250.png', dpi = 150, bbox_inches = 'tight')

plt.figure()
plt.hist(noises[:,3], 100)
plt.title('Balance of Sig at 0350 microns')
plt.savefig('Balance0350.png', dpi = 150, bbox_inches = 'tight')

plt.figure()
plt.hist(noises[:,4], 100)
plt.title('Balance of Sig at 0500 microns')
plt.savefig('Balance500.png', dpi = 150, bbox_inches = 'tight')