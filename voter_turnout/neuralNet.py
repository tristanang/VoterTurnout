# Based off set 4 code

import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

def neuralNet(X, y, X_val = None, y_val = None):
    ## Create your own model here given the constraints in the problem
    model = Sequential()
    
    # Add layers
    model.add(Dense(200))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Activation('relu'))   
    
    # Add a sigmoid layer to output probability
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    
    # Binary classification so use binary_crossentropy
    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop', 
                  metrics=['accuracy'])
    
    # Defaults. Found little benefit of changing.
    batch_size = 32
    epochs = 3
    if X_val is not None and y_val is not None:
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=1, 
                  validation_data = (X_val, y_val))
    else:
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=1)
        
    return model

'''
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
	

## Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_test, y_test, verbose=0)
in_sample_score = model.evaluate(X_train, y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('In-sample accuracy:', in_sample_score[1])
'''