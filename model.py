import tensorflow as tf

class Model:
    def __init__(self, dim):
        self.model = None
        self.dim = dim

    def build(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.dim, 100))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
        #self.model.add(tf.keras.layers.Concatinate())
        #self.model.add(tf.keras.layers.Reshape((64,)))
        #self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        #self.model.add(tf.keras.layers.BatchNormalization())
        #self.model.add(tf.keras.layers.Dropout(0.5))
        #self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # sparse_categorical_crossentropy
        # VonMisesFisher
        print(self.model.summary())


m = Model(128)
m.build()