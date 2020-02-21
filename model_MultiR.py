import tensorflow as tf
from keras.initializers import Constant
from keras import models
from keras.models import Model, load_model
from keras import applications
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Concatenate, GlobalAveragePooling2D, Input, Reshape, Conv3D, MaxPooling3D, Conv2D, AveragePooling3D, BatchNormalization, Activation, MaxPooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, LambdaCallback, Callback
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

class ModelMultiR:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = None

    def conv_block3D(self, input_size, firstBlock, last_layer):
        if firstBlock == True:
            inputs = Input(input_size)
        else:
            inputs = last_layer

        ##Branch 1
        conv1 = Conv3D(8, 5, strides=1, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(inputs)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        conv1 = Conv3D(8, 5, strides=1, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(conv1)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        conv1 = Conv3D(8, 5, strides=2, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(conv1)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        ##Branch 2
        conv2 = Conv3D(8, 5, strides=1, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(inputs)
        conv2 = LeakyReLU(alpha=0.1)(conv2)
        conv2 = Conv3D(8, 5, strides=2, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(conv2)
        conv2 = LeakyReLU(alpha=0.1)(conv2)
        ##Branch 3
        conv3 = Conv3D(8, 5, strides=2, activation=None, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(inputs)
        conv3 = LeakyReLU(alpha=0.1)(conv3)
        ##Branch 4
        maxPool = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding='same')(inputs)
        ##Concatenate
        concatenate = Concatenate(axis=-1)([conv1, conv2, conv3, maxPool])
        return inputs, concatenate


    def fullyC(self, lastPool, mean):
        ##Fully-Connected
        flatten = Flatten()(lastPool)
        dense1 = Dense(64, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.1))(flatten)
        dense1 = LeakyReLU(alpha=0.1)(dense1)
        dense1 = Dropout(0.5)(dense1)
        out = Dense(1, activation="linear", bias_initializer=Constant(value=mean))(dense1)
        return out

    def create_model(self, inputs, out):
        self.model = Model(input=inputs, output=out)
        self.model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=3e-4),
                      metrics=['mean_absolute_error'])

    def get_model(self, mean):
        inputs, finalLayer_ConvBlock3D_1 = self.conv_block3D(input_size=(16, 180, 180, 1), firstBlock=True, last_layer=None)
        _, finalLayer_ConvBlock3D_2 = self.conv_block3D(input_size=None, firstBlock=False, last_layer=finalLayer_ConvBlock3D_1)
        _, finalLayer_ConvBlock3D_3 = self.conv_block3D(input_size=None, firstBlock=False, last_layer=finalLayer_ConvBlock3D_2)
        outputs = self.fullyC(finalLayer_ConvBlock3D_3, mean)
        self.model = self.create_model(inputs, outputs)
        return self.model