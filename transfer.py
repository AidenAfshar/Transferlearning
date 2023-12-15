from typing import Sequence
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.applications.efficientnet_v2 as efficientnet_v2
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Removes unecessary warnings

# setting up training and validation data
train, validation = utils.image_dataset_from_directory(
    'butterflies',
    label_mode = 'categorical',
    image_size = (224, 224),
    seed = 1239,
    validation_split = 0.30,
    subset = "both",
)
class_names = train.class_names
train = train.map(lambda x, y : (efficientnet_v2.preprocess_input(x), y))
validation = validation.map(lambda x, y : (efficientnet_v2.preprocess_input(x), y))


efficientnet_v2 = efficientnet_v2.EfficientNetV2B0(
    include_top = True, # Includes last dense layers
    weights = "imagenet", # Standard image classification weights
)
efficientnet_v2.trainable = False


class MyModel():
  def __init__(self, input_shape):
    super().__init__()
    self.model = Sequential()
    self.model.add(efficientnet_v2)
    self.model.add(layers.Dense(1024, activation = 'relu'))
    self.model.add(layers.Dense(256, activation = 'relu'))
    self.model.add(layers.Dense(64, activation = 'relu'))
    self.model.add(layers.Dense(10, activation = 'softmax'))
    self.loss = losses.CategoricalCrossentropy()
    self.optimizer = optimizers.SGD(learning_rate = 0.01)

    self.model.compile(
        loss = self.loss,
        optimizer = self.optimizer,
        metrics = ['accuracy']
    )
    def __str__(self):
        self.model.summary()
        return "This model uses EfficientNetV2B0 to detect species of butterflies"
    def save(self, filename):
        self.model.save(filename)

callbacks = [
       callbacks.ModelCheckpoint(
          'checkpoints/checkpoints_{epoch:02d}', # Filepath
          verbose = 2,
          save_freq = 76,
       )
    ]

def createGraphs(accuracy, losses, val_accuracy, val_losses):
    plt.figure(1)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(0, len(accuracy)), accuracy, label="train")
    plt.plot(np.arange(0, len(val_accuracy)), val_accuracy, label="validation")
    plt.figure(2)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(np.arange(0, len(losses)), losses, label="train")
    plt.plot(np.arange(0, len(val_losses)), val_losses, label = "validation")
    plt.show()

def main():
    numEpochs = input("Enter number of epochs: ")
    while True:
        try:
            numEpochs = int(numEpochs)
            if numEpochs < 1:
                numEpochs = input("Please enter an integer above 0: ")
            else:
                break
        except:
            numEpochs = input("Please enter an integer above 0: ")

    model = MyModel((224,224,3))
    trainData = model.model.fit(
        train,
        batch_size = 32,
        epochs = numEpochs,
        verbose = 1,
        validation_data = validation,
        validation_batch_size = 32,
        callbacks=callbacks,
    )
    history = trainData.history

    save_path = 'saves/model_save'
    model.model.save(save_path)
    with open(f'{save_path}/class_names.data','wb') as f:
       pickle.dump(class_names, f)
    print(f"{model}")

    createGraphs(
                history['accuracy'],
                history['loss'],
                history["val_accuracy"],history["val_loss"]
                )
    return

main()
