import tensorflow as tf
from keras.initializers import Constant
from keras import models
from keras.models import Model, load_model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Reshape, Conv3D, MaxPooling3D, Conv2D, AveragePooling3D, BatchNormalization, Activation, MaxPooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, LambdaCallback, Callback
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

class Model3D2D:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = None


    def conv_block3D(self):

        inputs = Input(self.input_size)

        ##Convolutional Layer 1
        conv1 = Conv3D(32, (3, 5, 5), activation=None, padding='valid', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(inputs)
        bn1 = BatchNormalization()(conv1)
        bn1 = LeakyReLU(alpha=0.1)(bn1)
        maxPool1 = MaxPooling3D(pool_size=(1, 2, 2))(bn1)

        ##Convolutional Layer 2
        conv2 = Conv3D(64, (2, 5, 5), activation=None, padding='valid', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(maxPool1)
        bn2 = BatchNormalization()(conv2)
        bn2 = LeakyReLU(alpha=0.1)(bn2)
        maxPool2 = MaxPooling3D(pool_size=(1, 2, 2))(bn2)

        average = AveragePooling3D(pool_size=(64, 1, 1), padding='same')(maxPool2)
        reshape = Reshape((42, 42, 64))(average)

        return inputs, reshape


    def conv_block2D(self, last_inputs):
        ##Convolutional Layer 1
        conv1 = Conv2D(64, 5, activation=None, padding='valid', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(last_inputs)
        bn1 = BatchNormalization()(conv1)
        bn1 = LeakyReLU(alpha=0.1)(bn1)
        maxPool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        ##Convolutional Layer 2
        conv2 = Conv2D(128, 5, activation=None, padding='valid', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(maxPool1)
        bn2 = BatchNormalization()(conv2)
        bn2 = LeakyReLU(alpha=0.1)(bn2)
        maxPool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        return maxPool2

    def fullyC(self, lastPool, mean):
        ##Fully-Connected
        flatten = Flatten()(lastPool)
        dense1 = Dense(256, activation=None, kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(0.1))(flatten)
        dense1 = LeakyReLU(alpha=0.1)(dense1)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(64, activation=None, kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(0.1))(dense1)
        dense2 = LeakyReLU(alpha=0.1)(dense2)
        dense2 = Dropout(0.5)(dense2)
        out = Dense(1, activation="linear", bias_initializer=Constant(value=mean))(dense2)
        return out

    def create_model(self, inputs, out):
        self.model = Model(input=inputs, output=out)
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=3e-4),
                           metrics=['mean_squared_error', 'mean_absolute_error'])


    def get_model(self, mean):
        inputs, finalLayer_ConvBlock3D = self.conv_block3D()
        finalLayer_ConvBlock2D = self.conv_block2D(finalLayer_ConvBlock3D)
        finalLayer_maxPool = self.fullyC(finalLayer_ConvBlock2D, mean)
        self.create_model(inputs, finalLayer_maxPool)
        return self.model


