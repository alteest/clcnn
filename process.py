import os

from convert import Converter
from model import Model
from generator import DataGenerator


class Processor:
    def __init__(self, size):
        self.base_dir = "C:\\Users\\adanilov\\PycharmProjects\\clcnn"
        # self.base_dir = os.curdir
        self.x = []  # FIXME move local to process ?
        self.y = []
        self.converter = Converter(size)
        self.model = Model(size)
        self.model.build()

    def process(self):

        full = DataGenerator.get_data(128, self.base_dir)
        print("FULL :", full.shape)
        if full is None:
            print("Error getting data")
            return

        train = full.sample(frac=0.8, random_state=200)
        test = full.drop(train.index)
        print("TRAIN :", train.shape)
        print("TEST :", test.shape)

        train_generator = DataGenerator(train)
        validation_generator = DataGenerator(test)

        model_dir = os.path.join(self.base_dir, "model")
        self.model.fit(train_generator, validation_generator, model_dir)


if __name__ == "__main__":
    Processor(128).process()
    print("DONE")
