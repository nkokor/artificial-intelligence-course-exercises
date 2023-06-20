import os
import numpy as np
import pandas as pd

from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt


#characteristics data vectorization function
def vectorize_sequences(sequences, dimension=4000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

#labels vectorization function
def vectorize_labels(labels):
  results = np.zeros(len(labels))
  for i, label in enumerate(labels):
    if(label.lower() == 'spam'):
      results[i] = 1
    elif(label.lower() == 'ham'):
      results[i] = 0
  return results

#model defining function
def define_model():
  model = models.Sequential()
  model.add(layers.Dense(8, activation='relu', input_shape=(4000,)))
  model.add(layers.Dense(8, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  return model


#loading data from an external file
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'data', 'SpamDetectionData.txt')
data = pd.read_csv(file_path, sep=',')

#inspecting data structure by viewing a few rows
print(data.head(5))

#viewing data characteristics
print(data.describe())

#splitting data to characteristics data and labels
x = data['Message']
y = data['Label']

#removing <p> and </p> tags from data
for i in range(0, len(x)):
  x[i] = x[i].strip('<p>')
  x[i] = x[i].strip('</p>')

#splitting data to train and test data with a 10% test size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#characteristics data tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
train_data_seq = tokenizer.texts_to_sequences(x_train)
test_data_seq = tokenizer.texts_to_sequences(x_test)

#viewing created index and word counts
print(tokenizer.word_index)
print(tokenizer.word_counts)

#data vectorization
train_data_vec = vectorize_sequences(train_data_seq)
test_data_vec = vectorize_sequences(test_data_seq)
train_label_vec = vectorize_labels(y_train)
test_label_vec = vectorize_labels(y_test)

#model defining and training with 30% validation split
model = define_model()
history = model.fit(train_data_vec, train_label_vec, epochs=5, batch_size=128, validation_split=0.3)

#model evaluation
results = model.evaluate(test_data_vec, test_label_vec)
print('Loss: ', results[0])
print('Accuracy: ', results[1] * 100, '%')

#accuracy graph plotting
acc = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, 6)
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
plt.plot(epochs, loss, 'bo', label='loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Loss graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#using the model on unseen data
message = 'Dear students, your final grades will be uploaded today. I hope you all have a great summer. See you in september!'
message_seq = tokenizer.texts_to_sequences([message])
message_vec = vectorize_sequences(message_seq)
prediction = model.predict(message_vec)
print(prediction)