import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class autoencoder:
    def __init__(self, img_size, code_size):
        self.image_size = img_size

        # The encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer((img_size, img_size, 3)))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(code_size))

        # The decoder
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer((code_size,)))
        self.decoder.add(tf.keras.layers.Dense(img_size * img_size * 3))
        self.decoder.add(tf.keras.layers.Reshape((img_size, img_size, 3)))

        inp = tf.keras.layers.Input((img_size, img_size, 3))
        code = self.encoder(inp)
        reconstruction = self.decoder(code)

        self.autoencoder = tf.keras.models.Model(inp, reconstruction)
        self.autoencoder.compile(optimizer='adamax', loss='mse')
        self.autoencoder.summary()

    def predict(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(
            image, [self.image_size, self.image_size])
        image /= 255
        x = tf.expand_dims(image, 0)
        predictions = self.autoencoder.predict(x)

        """ display image """
        predictions = np.array(predictions[0]).reshape(
            (self.image_size, self.image_size, 3))
        plt.imshow(predictions)
        plt.show()
