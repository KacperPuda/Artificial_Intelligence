import tensorflow as tf
import os
import cv2
import imghdr
import imageio.v2 as imageio

import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model
from keras.models import Model

prepareData = False
showData = False
buildModel = False
trainModel = False
evaluate = False
numOfStyles = 5
photoSize = 200
# preparing data -> delete invalid photos
if prepareData:
    data_dir = "smallData"
    print(os.listdir(data_dir))
    for image_class in os.listdir(data_dir):
        print(image_class)
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            try:
                if tip not in ['jpg', 'jpeg', 'bmp', 'png']:
                    print("image is not valid")
                    os.remove(image_path)
            except Exception as a:
                print(a)
# preparing data set

data = tf.keras.utils.image_dataset_from_directory('smallData', image_size=(photoSize, photoSize), batch_size=265)

# 0 - Expressionism
# 1 - Japanese art
# 2 - Realism
# 3 - Rococo
# 4 - Symbolism

# data scaling
data = data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=numOfStyles)))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

if showData:
    fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:15]):
        row = idx // 5
        col = idx % 5
        ax[row, col].imshow(img.astype(int))
        ax[row, col].title.set_text(batch[1][idx])

# creating sets

train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)
print(f"Data train size: {train_size}")
print(f"Data validation size: {val_size}")
print(f"Data test size: {test_size}")

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# model

if buildModel:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(photoSize, photoSize, 3)))
    model.add(MaxPooling2D())
    model.add(Dropout(.15))
    model.add(Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.15))
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.15))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(numOfStyles, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save("Cypuka3.h5")
    # model.summary()
else:
    model = load_model("Cypuka3.h5")

# training

if trainModel:
    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])
    model.save("Cypuka3.h5")
else:
    hist = None
# plot performance
if trainModel:
    fig = plt.figure()
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    fig.suptitle('Loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], label='accuracy')
    plt.plot(hist.history['val_accuracy'], label='val_accuracy')
    fig.suptitle('Accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.show()

# evaluation
if evaluate:
    evaluation = model.evaluate(test)

model.summary()

# new photos prediction
# [Baroque, Expressionism, Japanese art, Realism, Symbolism]

def print_prediction(i):
    if i == 0:
        print("Predicted: Expressionism")
    if i == 1:
        print("Predicted: Japanese art")
    if i == 2:
        print("Predicted: Realism")
    if i == 3:
        print("Predicted: Rococo")
    if i == 4:
        print("Predicted: Symbolism")

print()
img = cv2.imread('newData//expressionism.jpg')
plt.imshow(img)
resize = tf.image.resize(img, (photoSize, photoSize))
img = np.expand_dims(resize / 255, 0)
yhat = model.predict(img)
print(yhat)
print("Expressionism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//expressionism2.jpg')
plt.imshow(img)
resize = tf.image.resize(img, (photoSize, photoSize))
img = np.expand_dims(resize / 255, 0)
yhat = model.predict(img)
print(yhat)
print("Expressionism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//japan.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Japanese art")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//japan2.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Japanese art")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//realism.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Realism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//realism2.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Realism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//symbolism.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Symbolism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//symbolism2.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Symbolism")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//rococo.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Rococo")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

print()
img = cv2.imread('newData//rococo2.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (photoSize, photoSize))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(yhat)
print("Rococo2")
idx_max = np.argmax(yhat)
print_prediction(idx_max)

