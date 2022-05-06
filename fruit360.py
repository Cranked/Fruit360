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
# ****************CNN Model***************

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))

model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(numberOfClass))  # OutputLayer
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"]
              )
batch_size = 32

# ******** Data Generation-Train-Test**********

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.3,
                                   horizontal_flip=True,
                                   zoom_range=0.3,
                                   )
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=x.shape[:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical"
                                                    )
test_generator = test_datagen.flow_from_directory(train_path,
                                                  target_size=x.shape[:2],
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  class_mode="categorical"
                                                  )

hist=model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1600 // batch_size,
    epochs=100,
    validation_data=test_generator,
    validation_steps=400 // batch_size
)
model.save_weights("deneme.h5")

print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label="Train accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation accuracy")
plt.legend()
plt.show()
#%% save history
import json
with open("deneme.json","w") as f:
    json.dump(hist.history,f)

# Load History
import codecs
with codecs.open("cnn_fruit_hist.json","r",encoding="utf-8") as f:
    h=json.loads(f.read())
    plt.plot(h["loss"], label="Train Loss")
    plt.plot(h["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(h["acc"], label="Train acc")
    plt.plot(h["val_acc"], label="Validation acc")
    plt.legend()
    plt.show()