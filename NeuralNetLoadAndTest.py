#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:59:01 2019

@author: c1649794
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import datetime

plt.style.use('/home/phoenixdata/c1649794/Python/alexDefault.mplstyle')

root = '/home/user/c1649794/HDD/Python/NoiseFind/Noisy/'
ntiles = 20000
ntot   = 22000
waveband = 70
imgSize = 100
epochs   = 100

test_img = np.empty((ntot-ntiles,imgSize,imgSize,1))
test_sig = np.empty((ntot-ntiles))


print 'Getting {:d} image files from '.format(ntot) + root + '{:04d}/'.format(waveband)
print '{:d} images in Training Pool'.format(ntiles) + ' ({:0.0f}%)'.format(100.*ntiles/ntot)
print '{:d} images in Testing Pool'.format(ntot-ntiles) + ' ({:0.0f}%)'.format(100.*(ntot-ntiles)/ntot)
 
    
for i in range(ntot-ntiles):
    s = '{:04d}/'.format(waveband) + '{:04d}_'.format(waveband) + '{:04d}.fits'.format(i+1+ntiles)
    test_img[i,:,:,0] = fits.getdata(root + s)
    test_sig[i]        = fits.getheader(root + s)['NOISE']
    
print 'Images retrived. Z normalising (min-max normalisation).'

Ztest_img = (test_img - np.min(test_img, axis = (1,2))[:,None,None]) / (
             np.max(np.abs(test_img), axis = (1,2))[:,None,None] - np.min(test_img, axis = (1,2))[:,None,None])

print 'Images normalised. Proceeding to training.'

model = load_model('Model_ntiles20000wvln0070epochs100d2019-09-11t18:55:45')

predictions = model.predict(Ztest_img)

plt.figure(figsize=(8,8))
plt.scatter(test_sig,predictions)
plt.xlabel('True Sig')
plt.ylabel('Predicted Sig')
plt.xlim((0,test_sig.max() + (test_sig.max()*0.1)))
plt.ylim((0,predictions.max() + (predictions.max()*0.1)))
line = np.linspace(0,test_sig.max() + (test_sig.max()*0.1),100)
plt.plot(line,line,'k--')
plt.savefig('TrueVsPred_' + 'ntiles20000wvln0070epochs100d2019-09-11t18:55:45' + '.png',dpi=150, bbox_inches='tight')
