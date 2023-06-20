import os
import numpy as np
import pandas as pd

from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from matplotlib import pyplot as plt

#characteristics data vectorization function
def vectorize_sequences(sequences, dimension=500):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

#model defining function
def define_model():
  model = models.Sequential()
  model.add(layers.Dense(32, activation='relu', input_shape=(500,)))
  model.add(layers.Dense(8, activation='relu'))
  model.add(layers.Dense(4, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


#loading data from an external file
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'data', 'stackoverflow.csv')
data = pd.read_csv(file_path, sep=',')

#inspecting data structure by viewing a few rows
print(data.head(5))

#viewing data characteristics
print(data.describe())

#splitting data to characteristics data and labels
x = data['post']
y = data['tags']

#viewing possible labels
print(set(y))

#label one-hot-encoding
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

#splitting data to train and test data with a 10% test size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#characterstics data tokenization
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(x_train)
train_data_seq = tokenizer.texts_to_sequences(x_train)
test_data_seq = tokenizer.texts_to_sequences(x_test)

#characteristics data vectorization
train_data_vec = vectorize_sequences(train_data_seq)
test_data_vec = vectorize_sequences(test_data_seq)

#model defining and viewing
model = define_model()
print(model.summary)

#model training with 25% validation split
history = model.fit(train_data_vec, y_train, epochs=8, batch_size=8, validation_split=0.25)

#accuracy graph plotting
acc = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Accuracy graph')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#loss graph plotting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Loss graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#model evaluation
results = model.evaluate(test_data_vec, y_test)
print('Loss: ', results[0])
print('Accuracy: ', results[1] * 100, '%')