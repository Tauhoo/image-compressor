# Image-compressor

Image-compressor is a Python project for compress image file and reconstruct it with autoencoder.

## Installation

Use the package manager [pip](https://help.dreamhost.com/hc/en-us/articles/115000702772-Installing-a-custom-version-of-Python-3) to install.

```bash

pip install tensorflow

pip install matplotlib

pip install numpy

```

And you need to put your data that you want to train.

```
.

├── files
|   └── data
├── src
├── config.ini # config training
├── compress.py
└── index.py

```

- In data folder, Can create sub directory and put image there.

## Config

```
[source]
   data_folder_path = ./files/data # Your data path. you can change it.
   weight_file_path = ./files/weight.h5 # your weight file path. it will be load and update.
[setting]
   code_size = 900 # Size of output node in encode layer
   image_size = 60  # Size of your image that will be resized before trained.
   epochs = 6
   steps_per_epoch = 1000

```

- if you don't have the weight file, you just put the target path that you want to save your weight in weight_file_path.

## Usage

Config your training.

Train by run this command.

```bash

python3.7 index.py

```

Test with CLI.

```bash

python3.7 compress.py

```
