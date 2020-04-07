from src.image_reader import image_reader
from src.configure import configure
from src.autoencoder import autoencoder
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from os import path

""" initialize config file """
config = configure()
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = int(config.setting['image_size'])
epochs = int(config.setting['epochs'])
steps_per_epoch = int(config.setting['steps_per_epoch'])
code_size = int(config.setting['code_size'])

model = autoencoder(image_size, code_size)

""" load weight """
if path.exists(weight_file_path):
    model.autoencoder.load_weights(weight_file_path)

""" input """
pic_path = input('path : ')

if path.exists(pic_path) and not path.isdir(pic_path):
    model.predict(pic_path)
else:
    print('image file not found')
