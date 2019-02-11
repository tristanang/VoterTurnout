# Based off set 4 code

import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

## Create your own model here given the constraints in the problem
model = Sequential()

# Add layers
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.002))

## Add a sigmoid layer to output probability
model.add(Dense(1))
model.add(Activation('sigmoid'))

# One-hot encoding the labels, so use 'categorical_crossentropy' as loss.
## You will likely have the best results with RMS prop or Adam as your optimizer.  In the line below we use Adadelta
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop', 
              metrics=['accuracy'])

# Defaults. Found little benefit of changing.
batch_size = 32
epochs = 10

fit = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
    verbose=1)
	
## Problem D
## Printing a summary of the layers and weights in your model
model.summary()

## Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_test, y_test, verbose=0)
in_sample_score = model.evaluate(X_train, y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('In-sample accuracy:', in_sample_score[1])