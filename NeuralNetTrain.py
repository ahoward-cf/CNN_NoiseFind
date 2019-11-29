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
from tensorflow.keras.models import Sequential
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
no_conv = 5
no_node = 256


train_img = np.empty((ntiles*4,imgSize,imgSize,1))
train_sig = np.empty((ntiles*4))

test_img = np.empty((ntot-ntiles,imgSize,imgSize,1))
test_sig = np.empty((ntot-ntiles))


print 'Getting {:d} image files from '.format(ntot) + root + '{:04d}/'.format(waveband)
print '{:d} images in Training Pool'.format(ntiles) + ' ({:0.0f}%)'.format(100.*ntiles/ntot)
print '{:d} images in Testing Pool'.format(ntot-ntiles) + ' ({:0.0f}%)'.format(100.*(ntot-ntiles)/ntot)
for i in range(ntiles):
    s = '{:04d}/'.format(waveband) + '{:04d}_'.format(waveband) + '{:04d}.fits'.format(i+1)
    train_img[i,:,:,0] = fits.getdata(root + s)
    train_sig[i]        = fits.getheader(root + s)['NOISE']
    
    train_img[ntiles+i,:,:,0] = np.rot90(train_img[i,:,:,0], 1)
    train_img[2*ntiles+i,:,:,0] = np.rot90(train_img[i,:,:,0], 2)
    train_img[3*ntiles+i,:,:,0] = np.rot90(train_img[i,:,:,0], 3)
    
    train_sig[ntiles+i] = train_sig[i]
    train_sig[2*ntiles+i] = train_sig[i]
    train_sig[3*ntiles+i] = train_sig[i]      
    
#train_img[ntiles:,:,:,0] = np.flip(train_img[:ntiles,:,:,0], axis = 2)
#train_sig[ntiles:] = train_sig[:ntiles]    
    
for i in range(ntot-ntiles):
    s = '{:04d}/'.format(waveband) + '{:04d}_'.format(waveband) + '{:04d}.fits'.format(i+1+ntiles)
    test_img[i,:,:,0] = fits.getdata(root + s)
    test_sig[i]        = fits.getheader(root + s)['NOISE']
    
print 'Images retrived. Z normalising (min-max normalisation).'
   
Ztrain_img = (train_img - np.min(train_img, axis = (1,2))[:,None,None]) / (
             np.max(np.abs(train_img), axis = (1,2))[:,None,None] - np.min(train_img, axis = (1,2))[:,None,None])

Ztest_img = (test_img - np.min(test_img, axis = (1,2))[:,None,None]) / (
             np.max(np.abs(test_img), axis = (1,2))[:,None,None] - np.min(test_img, axis = (1,2))[:,None,None])

print 'Images normalised. Proceeding to training.'

model = Sequential()

for i in range(no_conv):
    if i == 0:
        model.add(Conv2D(no_node,(3,3),input_shape = test_img.shape[1:], data_format='channels_last'))
    else:
       model.add(Conv2D(no_node,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

ostr = ('ntiles{:d}_'.format(ntiles) + 
       'wvln{:04d}_'.format(waveband)+
       'conv{:d}_'.format(no_conv)+
       'nodes{:d}_'.format(no_node)+
       'epochs{:d}_'.format(epochs)  +
       'd'+
       str(datetime.datetime.now().date())+
       't'+
       str(datetime.datetime.now().time().hour)+
       ':'+
       str(datetime.datetime.now().time().minute)+
       ':'+
       str(datetime.datetime.now().time().second)
       )


tensorboard = TensorBoard(log_dir="logs/" + ostr)

model.compile(loss='mse', optimizer = 'sgd', metrics=['accuracy'])

model.fit(Ztrain_img, train_sig, batch_size = 32, epochs=epochs, validation_split = 0.2, callbacks = [tensorboard])

val_loss, val_acc = model.evaluate(Ztest_img,test_sig)
print val_loss
print val_acc

predictions = model.predict(Ztest_img)

model.save('Model_'+ostr)

plt.figure(figsize=(8,8))
plt.scatter(test_sig,predictions)
plt.xlabel('True Sig')
plt.ylabel('Predicted Sig')
plt.xlim((0,test_sig.max() + (test_sig.max()*0.1)))
plt.ylim((0,predictions.max() + (predictions.max()*0.1)))
line = np.linspace(0,test_sig.max() + (test_sig.max()*0.1),100)
plt.plot(line,line,'k--')
plt.savefig('TrueVsPred_' + ostr + '.png',dpi=150, bbox_inches='tight')
