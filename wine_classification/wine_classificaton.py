import numpy as np
import pandas as pd

from keras import models
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#model defining function
def define_model():
   model = models.Sequential()
   model.add(layers.Dense(8, activation='relu',input_shape=(12,)))
   model.add(layers.Dense(8, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

#loading data from external csv files
red_wine_data = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine_data = pd.read_csv('data/winequality-white.csv', sep=';')

#inspecting data structure by viewing a few rows
print(red_wine_data.head(5))
print(white_wine_data.tail(5))

#creating labels for both red and white wine data (1 for red, 0 for white)
red_wine_labels= np.ones(red_wine_data.to_numpy().shape[0])
white_wine_labels = np.zeros(white_wine_data.to_numpy().shape[0])

#merging labels with the corresponding datasets to create one dataset
red_wine_data['label'] = red_wine_labels
white_wine_data['label'] = white_wine_labels
wine_data = pd.concat([red_wine_data, white_wine_data])

#viewing data characteristics
print(wine_data.describe())

#splitting data to characteristics data and labels
x = wine_data.drop(columns=['label'])
y = wine_data['label']

#splitting data to train and test data with a 20% test size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#scaling features of characteristics data using standard scaling method
scaler = StandardScaler().fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

#defining and training the model
model = define_model()
history = model.fit(x_train, y_train, epochs=20, batch_size=16)

#evaluating the model
results = model.evaluate(x_test, y_test)
print('Loss: ', results[0])
print('Accuracy: ', results[1] * 100, '%')