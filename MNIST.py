import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras import datasets, layers, models
from keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from imgaug import augmenters as iaa
import imgaug as ia
import sys
from matplotlib import pyplot

#mnist ssoftmax
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
num_classes = 10
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
inputs = Input(shape=(784,))
model = models.Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, use_bias=False, activation=tf.keras.activations.softmax, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
metrics = ['accuracy']
optimizer = tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)
#optimizer = tf.train.AdamOptimizer(1.e-3)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=metrics)
batch_size = 64
epochs = 400
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])






#mnist-svm
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
#x_train = x_train.reshape(50000, 3072)
#x_test = x_test.reshape(10000, 3072)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
def model_1(x_input):
    x = Dense(512, activation='relu',)(x_input)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.8)(x)
    x_out = Dense(256, activation='relu')(x)
    x_out = Dropout(0.8)(x)
    return x_out
def model_2(x_input):
    x = Dense(800, activation='sigmoid')(x_input)
    #x = Dropout(0.0)(x)
    x = Dense(200, activation='sigmoid')(x)
    #x = Dropout(0.0)(x)
    x_out = Dense(12)(x)
    return x_out
inputs = Input(shape=(784,))
x      = model_1(inputs)
x_out  = Dense(10, use_bias=False, activation='linear', name='svm', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
model = Model(inputs, x_out)
def svm_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)
    
    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        regularization_loss = 0.5*(tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss + 0.4*hinge_loss
    
    return categorical_hinge_loss
metrics = ['accuracy']
optimizer = tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)
#optimizer = tf.train.AdamOptimizer(1.e-3)

model.compile(optimizer=optimizer, loss=svm_loss(model.get_layer('svm')), metrics=metrics)
batch_size = 64
epochs = 400

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])