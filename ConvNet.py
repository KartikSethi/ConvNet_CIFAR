# ConvNet for CIFAR10 Classification
"""
The architecture which we have built is relatively shallow because of current computational limitations. 
But still it boasts of an impressive maximum accuracy of 70.74% and a final accuracy of 67.62% on running it for 25 epochs. 
There are 5 convolutional layers, 2 max-pooling layers and one final fully-connected layer for final classification. 
The architecture is as follows:

1. CONV layer with 128 3x3 filters, stride 1, RELU activation, the maximum weight value set to 3 and inputs padded with zeroes
2. Dropout layer with dropout set to 0.2
3. CONV layer with 64 5x5 filters, stride 1, RELU activation, the maximum weight value set to 3 and inputs padded with zeroes
4. Dropout layer with dropout set to 0.2
5. CONV layer with 64 3x3 filters, stride 1, RELU activation, the maximum weight value set to 3 and inputs padded with zeroes
6. Dropout layer with dropout set to 0.5
7. 2x2 Max-pooling layer
8. CONV layer with 64 3x3 filters, stride 1, RELU activation, the maximum weight value set to 3 and inputs padded with zeroes
9. Dropout layer with dropout set to 0.2
10. CONV layer with 32 3x3 filters, stride 1, RELU activation, the maximum weight value set to 3 and inputs padded with zeroes
11. Dropout layer with dropout set to 0.5
12. 2x2 Max-pooling layer
13. Flattening layer
14. Fully-connected layer with 10 output nodes with softmax activation
"""

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Nadam
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 97
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Convolution2D(128, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
nadam = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))