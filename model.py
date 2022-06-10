import os
from geopy import distance
import tensorflow as tf


def geo_distance(point1: tf.Tensor, point2: tf.Tensor):
    print("POINT1:, ", point1)
    print("POINT2:, ", point2)
    print("POINT1:, ", point1.value_index)
    print("POINT2:, ", point2.value_index)
    print("P2 0", point2[:, 0])
    print("P2 1", point2[:, 1])
    print("WHERE:", tf.where(point1).numpy())
    print("[]:", point1[:])
    p1 = tf.get_static_value(point1)
    print("P!:", p1)
    return distance.distance(point1, point2).km


class Model:
    def __init__(self, dim: int):
        self.model = None
        self.dim = dim

    def build(self):
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(self.dim, 100))

        """
        # From linked article: https://iyatomi-lab.info/sites/default/files/user/CSPA2018%20Proceedings_ito.pdf
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
        self.model.add(tf.keras.layers.Concatenate()) # FIXME concat 1 and 2
        self.model.add(tf.keras.layers.Reshape((64,)))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        """

        # From original article
        # self.model.add(tf.keras.layers.Input(shape=(3, 280, 128)))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu', input_shape=(280, 128, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
        """
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
        """
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(124, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))  # FIXME is it required?
        """
        self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        """
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])  # mean_squared_error
        # VonMisesFisher
        print(self.model.summary())

    def fit(self, test_generator, validation_generator, model_dir):
        if not self.model:
            print("Model is not inited yet")
            return

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model.fit(test_generator, validation_data=validation_generator, batch_size=32, epochs=10, verbose=2)
        self.model.save(model_dir)
