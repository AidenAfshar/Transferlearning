from typing import Sequence
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.applications.densenet as densenet
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.losses as losses
from tensorflow.keras import Sequential
import pickle
from pprint import pprint
import tensorflow.data as data
import os
from matplotlib import pyplot as plt
import numpy as np

# setting up training and validation data
train, validation = utils.image_dataset_from_directory(
    'butterflies',
    label_mode = 'categorical',
    image_size = (224, 224),
    seed = 1312,
    validation_split = 0.30,
    subset = "both",
)
class_names = train.class_names
train = train.map(lambda x, y : (densenet.preprocess_input(x), y))
validation = validation.map(lambda x, y : (densenet.preprocess_input(x), y))


densenet = densenet.DenseNet121(
    include_top = True, # Includes last dense layers
    weights = "imagenet", # Standard image classification weights
    # classifier_activation = 'softmax',
)
densenet.trainable = False

"""
print(f"{train}")
print(f"{validation}")
print(f"{densenet}")
"""

class MyModel():
  def __init__(self, input_shape):
    super().__init__()
    self.model = Sequential()
    self.model.add(densenet)
    self.model.add(layers.Dense(1024, activation = 'relu'))
    self.model.add(layers.Dense(256, activation = 'relu'))
    self.model.add(layers.Dense(64, activation = 'relu'))
    self.model.add(layers.Dense(10, activation = 'softmax'))
    self.loss = losses.CategoricalCrossentropy()
    self.optimizer = optimizers.SGD(learning_rate = 0.6)

    self.model.compile(
        loss = self.loss,
        optimizer = self.optimizer,
        metrics = ['accuracy']
    )
    def __str__(self):
        self.model.summary()
        return ""
    def save(self, filename):
        self.model.save(filename)

callbacks = [
       callbacks.ModelCheckpoint(
          'checkpoints/checkpoints_{epoch:02d}', # Filepath
          verbose = 2,
          save_freq = 76,
       )
    ]

def createGraph(accuracy, losses, val_accuracy, val_losses):
    plt.plot(np.arange(0, len(accuracy)), accuracy)
    plt.plot(np.arange(0, len(losses)), losses)
    plt.plot(np.arange(0, len(val_accuracy)), val_accuracy)
    plt.plot(np.arange(0, len(val_losses)), val_losses)
    plt.show()

model = MyModel((224,224,3))
trainData = model.model.fit(
    train,
    batch_size = 12,
    epochs = 30,
    verbose = 1,
    validation_data = validation,
    validation_batch_size = 12,
    callbacks=callbacks,
)
history = trainData.history
print(history['accuracy'])
print(len(history['accuracy']))

save_path = 'saves/faces_model_save_2023_02_08__40_epochs'
model.model.save(save_path)
with open(f'{save_path}/class_names.data','wb') as f:
   pickle.dump(class_names, f)
print(f"{model}")

createGraph(
            history['accuracy'],
            history['loss'],
            history["val_accuracy"],history["val_loss"]
            )
