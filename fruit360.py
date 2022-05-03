import tensorflow as tf
import keras
import os

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from glob import glob

train_path = "Training/"
test_path = "Test/"
msg = tf.constant('Hello, Piipııls dis iz mahmut tuncer remikss!')

img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*')
numberOfClass = len(className)
print(f'Numberofclass', numberOfClass)
# *************** CNN Model *********

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass))# OutputLayer
model.add(Activation("softmax"))


model.compile(loss="categorycal_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"]
              )
batch_size=32

#******** Data Generation-Train-Test**********

