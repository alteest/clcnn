import os
import argparse

from convert import Converter
from model import Model
from generator import DataGenerator


class Processor:
    def __init__(self, size: int, base_dir: str):
        self.base_dir = base_dir
        self.converter = Converter(size)
        self.model = Model(size)
        self.model.build()

    def process(self):

        full = DataGenerator.get_data(128, self.base_dir)
        print("FULL :", full.shape)
        print(full.head())
        if full is None:
            print("Error getting data")
            return

        train = full.sample(frac=0.9, random_state=42)
        test = full.drop(train.index)
        print("TRAIN :", train.shape)
        print("TEST :", test.shape)

        train_generator = DataGenerator(train)
        validation_generator = DataGenerator(test)

        model_dir = os.path.join(self.base_dir, "model")
        self.model.fit(train_generator, validation_generator, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PROCESS')
    parser.add_argument('-b', '--base_dir', help="Base dir where located 'data' folder", type=str, required=True,
                        dest='base_dir')
    parser.add_argument('-d', '--dimension', help="Dimensions", type=int, dest="dim", default=128)
    result = parser.parse_args()

    Processor(result.dim, result.base_dir).process()
    print("DONE")
