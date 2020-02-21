from dataset import *
from model_3D2D import *
from utils import *
import numpy as np
from sklearn.model_selection import KFold
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


class ModelManager:

    def __init__(self, k_fold, batch_size, epochs,
                 data_preparer: DataPreparer, model_preparer: Model3D2D):

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_processor = data_preparer
        self.model_processor = model_preparer
        self.X = None
        self.y = None
        self.model = None


    def fit_model(self, X_train, y_train, X_test, y_test, fold_number):

            X_train, y_train = dataAugmentation(X_train, y_train)
            X_train = X_train.reshape(X_train.shape[0], 16, 180, 180, 1)
            X_test = X_test.reshape(X_test.shape[0], 16, 180, 180, 1)
            filepath = ".../Model_{}.h5".format(fold_number)
            model_checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
            history = self.model.fit(X_train, y_train,
                                batch_size=self.batch_size, epochs=self.epochs,
                                validation_data=(X_test, y_test), callbacks=[model_checkpoint])

    
    def run_pipeline(self):

        self.X, self.y = self.data_processor.get_data('Valsava Sinuses')
        kf = KFold(n_splits=self.k_fold)
        for fold_number, (train_index, test_index) in enumerate(kf.split(self.X), 1):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            mean = np.mean(y_train)
            self.model = self.model_processor.get_model(mean)
            self.fit_model(X_train, y_train, X_test, y_test, fold_number)


if __name__ == "__main__":
    data_processor = DataPreparer()
    model_processor = Model3D2D((16, 180, 180, 1))
    manager = ModelManager(k_fold=5, batch_size=8, epochs=50,
                          data_preparer=data_processor, model_preparer=model_processor)
    manager.run_pipeline()