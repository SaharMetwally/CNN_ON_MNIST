import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from numpy import expand_dims
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = expand_dims(x_train, 3)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
model.save("model")'''

new_model = tf.keras.models.load_model("model")

'''plt.imshow(x_test[4])
plt.show()'''

x_test = expand_dims(x_test, 3)
predictions = new_model.predict([x_test])  # here we need channel dimension so we expand x_test dimension
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))
print(np.argmax(predictions[3]))


'''check if thses predictions are correct ?!'''
print(np.argmax(y_test[0]))
print(np.argmax(y_test[1]))
print(np.argmax(y_test[2]))
print(np.argmax(y_test[3]))
