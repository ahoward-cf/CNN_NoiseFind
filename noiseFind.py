# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime, os

# Define data root
root = 'MLData/Noisefind/'



# Redefine the variables for easy loading
no_field = 25000
min_SNR  = 100.
max_SNR  = 500.
sideLen  = 100


fields = np.zeros((no_field, sideLen, sideLen, 1))

# load in the fields and noises
fields[:,:,:,0] = np.load(root+
        'nomSNR' + 
        '{:04.0f}'.format(min_SNR) +
        'to' +
        '{:04.0f}'.format(max_SNR) + 
        '_noField' + 
        '{:04d}'.format(no_field) +
        'fields.npy')
noises = np.loadtxt(root+
        'nomSNR' + 
        '{:04.0f}'.format(min_SNR) +
        'to' +
        '{:04.0f}'.format(max_SNR) + 
        '_noField' + 
        '{:04d}'.format(no_field) +
        'noises.txt')

# Min/Max scale the fields
normFields = (fields - np.nanmin(fields, axis = (1,2))[:,None,None]) / (np.nanmax(fields, axis = (1,2))[:,None,None] - np.nanmin(fields, axis = (1,2))[:,None,None])

# Rotate and stack the fields
normFields = np.vstack((normFields, 
                        np.rot90(normFields, 1, axes = (1,2)), 
                        np.rot90(normFields, 2, axes = (1,2)), 
                        np.rot90(normFields, 3, axes = (1,2))))

# Stack the noises to match
normNoises = np.hstack((noises, noises, noises, noises))

fields = []
noises = []

# Define Training and Testing sets
trainIDX = int(normFields.shape[0] * 0.8)
print(trainIDX)
print(normFields.shape)
print(normFields[:trainIDX,:,:,:].shape)
print(normNoises[:trainIDX].shape)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Define model layers
no_node = 256
no_conv = 5
no_dens = 8
epocs = 50

# Define the string with which to save the model and logs
modelStr = ('nomSNR' + 
            '{:04.0f}'.format(min_SNR) +
            'to' +
            '{:04.0f}'.format(max_SNR) + 
            '_noField' + 
            '{:06d}'.format(normFields.shape[0]) +
            '_trainSet' + 
            '{:06d}'.format(trainIDX) + 
            '_conv' + 
            '{:02d}'.format(no_conv) +
            '_node' + 
            '{:04d}'.format(no_node) + 
            '_dense' + 
            '{:02d}'.format(no_dens) + 
            '_epoc' + 
            '{:04d}_'.format(epocs) +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
print(modelStr)

# Define the log directory
logDir = root + 'logs/'
modDir = ''

# Set up tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(logDir + modelStr)

# Define the model function
def create_model(no_conv, no_node, no_dens, sideLen):
  # Set up a sequential model
  model = tf.keras.models.Sequential()
  # Add the 5 conv layers, with the first one having all the additional data
  for i in range(no_conv):
    if i == 0:
      model.add(tf.keras.layers.Conv2D(no_node,(3,3),
                                       input_shape = (sideLen, sideLen, 1),
                                       data_format = 'channels_last'))
    else:
      model.add(tf.keras.layers.Conv2D(no_node,(3,3)))
    # Add the activation and pooling layers
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
  
  # Flatten the model
  model.add(tf.keras.layers.Flatten())

  # Add dense layers, going from half node size down to 1 if I want 1 number
  current_nodes = no_node
  for i in range(no_dens):
    current_nodes = current_nodes // 2
    model.add(tf.keras.layers.Dense(current_nodes))
  
  return model

# Train model

model = create_model(no_conv, no_node, no_dens, sideLen)

with tf.device('/device:GPU:0'): 
  model.compile(loss = 'mse', optimizer='sgd', metrics = ['accuracy'])
  model.fit(normFields[:trainIDX,:,:,:], normNoises[:trainIDX], batch_size = 32,
            epochs = epocs, validation_split = 0.2, callbacks = [tensorboard])
  
model.save(modDir + modelStr)

# Test to see if it works

predictions = model.predict(normFields[trainIDX:,:,:])

plt.figure()
plt.scatter(normNoises[trainIDX:],predictions)
plt.xlabel('True Noise')
plt.ylabel('Predicted Noise')

