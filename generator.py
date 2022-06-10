import os
import csv
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from convert import Converter


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame,
                 batch_size: int = 16,
                 input_size: int = 128,
                 shuffle: bool = True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.nums = len(self.df)
        self.converter = Converter(input_size)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        X = []
        # X2 = []
        y = []
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        # print(f"Ask {index} from {self.df.shape} got batches {batches.shape}")

        for _, row in batches.iterrows():
            text = row["text"]
            raw = self.converter.convert_text(text)
            if raw is not None:
                # print(raw.shape)
                # print(f"Got '{text}' to {raw.shape}")
                # X2.append(raw)
                X.append(raw.tolist())
                y.append((row["latitude"], row["longitude"]))

        Xnumpy = np.array(X)
        # if len(Xnumpy.shape) == 1:
        #     print(f"X : '{len(X)}' : '{len(X[0])}'")
        # print(f"Ask {index} from {self.df.shape} got batches {batches.shape} converted to {Xnumpy.shape}")
        return Xnumpy, np.array(y)

    def __len__(self):
        return self.nums // self.batch_size

    @staticmethod
    def get_data(input_size, base_dir) -> [pd.DataFrame, None]:

        if not os.path.exists(base_dir):
            print(f"Data dir '{base_dir}' does not exist")
            return None

        converter = Converter(input_size)
        df = pd.DataFrame({'text': pd.Series(dtype='str'),
                           'latitude': pd.Series(dtype='float'),
                           'longitude': pd.Series(dtype='float')})
        i = 10
        data_dir = os.path.join(base_dir, "data")
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv'):
                latitude, longitude = fname[:-4].split('_', 1)
                latitude = float(latitude)
                longitude = float(longitude)
                fullname = os.path.join(data_dir, fname)

                # fdata = pd.read_csv(fullname, sep=';')  # FIXME just use cvs reader?
                # for text in fdata['text']:  # FIXME to do in one shot
                with open(fullname, newline='', encoding="utf8") as csvfile:
                    reader = csv.reader(csvfile, delimiter=';')
                    next(reader)
                    for row in reader:
                        text = row[3]
                        text = re.sub(r'https:\/\/t.co\/.{10}', "", text).strip()  # remove links
                        text = re.sub(r'@\w+', "", text).strip()  # remove mentions
                        text = re.sub(r"\s{2,}", " ", text).strip()
                        # print(text)
                        # print(text[-23:])
                        if text[-23:].startswith("https://t.co/"):
                            text = text[:-23].strip()
                        if not text:
                            continue

                        if converter.convert_text(text) is not None:
                            df.loc[len(df.index)] = [text, latitude, longitude]

                i -= 1
                if i < 0:
                    break

        return df


if __name__ == "__main__":
    gdf = DataGenerator.get_data(128, "C:\\Users\\adanilov\\PycharmProjects\\clcnn")
    print(gdf.shape)
    print(gdf.head())
