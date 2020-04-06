from src.image_reader import image_reader
from src.configure import configure
from tensorflow import keras
import tensorflow as tf
import math

""" initialize config file """
config = configure()
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = int(config.setting['image_size'])
epochs = int(config.setting['epochs'])
steps_per_epoch = int(config.setting['steps_per_epoch'])

""" initialize reader """
reader = image_reader(data_folder_path, image_size)
print(len(reader.evaluate_path), len(reader.train_path))

""" create model """
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(image_size, image_size, 3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(math.floor(image_size * image_size/4)))
model.add(keras.layers.Dense(image_size * image_size))
model.add(keras.layers.Reshape((image_size, image_size, 3)))
model.compile(
    optimizer='sgd',
    loss=tf.keras.losses.SquaredHinge(),
    metrics=['accuracy'])
model.summary()
