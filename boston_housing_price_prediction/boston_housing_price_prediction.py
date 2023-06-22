import numpy as np
import pandas as pd

from keras import models
from keras import layers
from keras.datasets import boston_housing

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

#model defining function
def define_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])
  return model

#curve smoothing function
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

#loading data from keras datasets
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#inspecting data characteristics
print(pd.DataFrame(train_data).describe())
print(pd.DataFrame(train_targets).describe())

#data scaling
mm = MinMaxScaler()
mm.fit(train_data)
train_data = mm.transform(train_data)
test_data = mm.transform(test_data)

#model defining and training with 10% validation split
model = define_model()
history = model.fit(train_data, train_targets, epochs=500, batch_size=1, validation_split=0.1)

#loss graph plotting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Loss graph')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#MAE graph plotting
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(mae)+1)
plt.plot(epochs, mae, 'bo', label='MAE')
plt.plot(epochs, val_mae, 'b', label='validation MAE')
plt.title('MAE graph')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend()
plt.show()

#MAE graph plotting with smoothing the curve
mae_history = history.history['val_mae']
smooth_mae_history = smooth_curve(mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel(epochs)
plt.ylabel('Validation MAE')
plt.show()

#model evaluating
results = model.evaluate(test_data, test_targets)
print('Loss: ', results[0])
print('MAE: ', results[1])