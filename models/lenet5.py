import os
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




class Lenet5_kroger:
  @staticmethod
  def build(data_shape,label_shape,dropout_rate,regularization):
    # initialize the model
    model = Sequential()

    # first CONV => RELU => POOL layer
    model.add(Conv2D(48, (3, 3), padding="same", activation='relu', input_shape=data_shape)) # Assuming 3 color channels
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second CONV => RELU => POOL layer
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third CONV => RELU => POOL layer
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # fourth CONV => RELU => POOL layer
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512, activation='relu'))

    # softmax classifier
    model.add(Dense(label_shape, activation='softmax'))

    # return the constructed network architecture
    return model

class LeNet_baseline:
  @staticmethod
  def build(data_shape, label_shape,dropout_rate,regularization):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=data_shape))
    #model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same", input_shape=data_shape))
    #model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("tanh"))

    # softmax classifier
    model.add(Dense(label_shape))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


class LeNet_reduced:
  @staticmethod
  def build(data_shape, label_shape, dropout_rate, regularization):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(10, (2, 2), padding="same", input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (2, 2), padding="same"))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))

    # add a Flatten layer
    model.add(Flatten())

    # softmax classifier
    model.add(Dense(label_shape))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

class LeNet_drop_reg:
  @staticmethod
  def build(data_shape, label_shape,dropout_rate,regularization):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=data_shape, kernel_regularizer=regularizers.l2(regularization)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same", input_shape=data_shape, kernel_regularizer=regularizers.l2(regularization)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500)) #200
    model.add(Activation("tanh"))
    model.add(Dropout(dropout_rate))

    # softmax classifier
    model.add(Dense(label_shape))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model



# dropout layer
# l2 regularization
# hyperparameter optimisation: batchsize, dropout rate, regulariation faktor



class LeNet_Hypermodel(kt.HyperModel):
  HP_BATCHSIZE = None
  HP_DROPOUT = None
  HP_REGULAIZATION = None

  def getBatchsize(self):
    return self.HP_BATCHSIZE
  def getDropout(self):
    return self.HP_DROPOUT
  def getRegularization(self):
    return self.HP_REGULAIZATION
  def declare_hyperparameters(self, hp):
    self.HP_BATCHSIZE = hp[2]
    self.HP_DROPOUT = hp[1]
    self.HP_REGULAIZATION = hp[0]

  def build(self, hparams):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    os.chdir('../')

    class PrettySafeLoader(yaml.SafeLoader):
      def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    with open('params.yaml', 'r') as stream:
      params = yaml.load(stream, Loader=PrettySafeLoader)
    datashape = params['datashape']
    labelshape = params['labelshape']

    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL layers
    print("Build CNN with Batch="+str(hparams.get(self.HP_BATCHSIZE))+" Drop="+str(hparams.get(self.HP_DROPOUT)) +" Reg="+str(hparams.get(self.HP_REGULAIZATION)))


    model.add(Conv2D(20, (5, 5), padding="same", input_shape=datashape,
                     kernel_regularizer=regularizers.l2(hparams.get(self.HP_REGULAIZATION))))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same", input_shape=datashape,
                     kernel_regularizer=regularizers.l2(hparams.get(self.HP_REGULAIZATION))))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500, kernel_regularizer=regularizers.l2(hparams.get(self.HP_REGULAIZATION))))
    model.add(Activation("tanh"))
    model.add(Dropout(hparams.get(self.HP_DROPOUT)))

    # softmax classifier
    model.add(Dense(labelshape))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

  def fit(self, hp, model, *args, **kwargs):
    return model.fit(
      *args,
      batch_size=hp[self.HP_BATCHSIZE],
      **kwargs,
    )
