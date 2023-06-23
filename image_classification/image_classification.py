import numpy as np
import pandas as pd

from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import cifar10
from keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

#model defining function
def define_model():
  model = models.Sequential()
  model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same'))
  model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Dropout(rate=0.2))
  model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same'))
  model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(3,3)))
  model.add(layers.Dropout(rate=0.2))
  model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same'))
  model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same'))
  model.add(layers.MaxPooling2D(pool_size=(3,3)))
  model.add(layers.Dropout(rate=0.2))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(rate=0.2))
  model.add(layers.Dense(10, activation='softmax'))
  opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

#loading data from cifar10
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

#plotting first 9 images from dataset
images = train_data[0:10]
for i in range(1, 10):
  plt.subplot(3, 3, i)
  plt.imshow(images[i])
  plt.axis("off")
plt.show()

#train and test data prepping
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#model defining and training
model = define_model()
history = model.fit(train_data, train_labels, epochs=30, batch_size=64, validation_data=(test_data, test_labels))

#model evaluation
results = model.evaluate(test_data, test_labels)
print("Loss: ", results[0])
print("Accuracy: ", results[1])

#loss graph plotting
loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1,31)
plt.plot(epochs, loss, 'bo', label='accuracy')
plt.plot(epochs, validation_loss, 'b', label='validation loss')
plt.title('Loss graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#accuracy graph plotting
acc = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
epochs = range(1,31)
plt.plot(epochs, acc, 'bo', label='accuracy')
plt.plot(epochs, validation_accuracy, 'b', label='validation accuracy')
plt.title('Accuracy graph')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()