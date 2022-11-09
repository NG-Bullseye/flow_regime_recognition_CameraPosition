import os

import keras_tuner
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

import keras_tuner as kt

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(
            hp.Choice('units', [8, 16, 32]),
            activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse')
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32]),
            **kwargs,
        )

#dropout layer
#l2 regularization
#hyperparameter optimisation: batchsize, dropout rate, regulariation faktor
class LeNet_Hypermodel(kt.HyperModel):
    def build(self, hp):
        HP_DATASHAPE =hp['datashape']
        HP_LABELSHAPE =hp['labelshape']

        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=HP_DATASHAPE,kernel_regularizer=regularizers.l2(hp.Choice("reg"))))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same",kernel_regularizer=regularizers.l2(hp.Choice("reg"))))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500),kernel_regularizer=regularizers.l2(hp.Choice("reg")))
        model.add(Activation("tanh"))

        model.add(Dropout(hp.Choice('drop')))

        # softmax classifier
        model.add(Dense(HP_LABELSHAPE))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch", [8, 16, 32, 64]),
            **kwargs,
        )
