from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense


class LeNet:
	@staticmethod
	def build(data_shape, label_shape):
		# initialize the model
		model = Sequential()

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=data_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("tanh"))
		model.add(Dense(250))
		model.add(Activation("tanh"))
		model.add(Dense(40))
		model.add(Activation("tanh"))
		model.add(Dense(10))
		model.add(Activation("tanh"))
		model.add(Dense(2))
		model.add(Activation("tanh"))
		model.add(Dense(label_shape))
		model.add(Activation("tanh"))

		# softmax classifier
		model.add(Dense(label_shape))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
