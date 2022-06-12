import os
import argparse
import numpy as np
import tensorflow as tf


from convert import Converter


class Predictor:
    def __init__(self,  size: int, base_dir: str):
        self.size = size
        self.base_dir = base_dir
        self.model = tf.keras.models.load_model(os.path.join(base_dir, "model"))
        self.converter = Converter(size)

    def predict(self, text: str) -> tuple:
        data = self.converter.convert_text(text)
        if data is not None:
            data = np.array([data.tolist()])
            print(data.shape)
            result = self.model.predict([data])[0]
            return (self.converter.convert_to_geo(result[0], 90),
                    self.converter.convert_to_geo(result[1], 180))
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PROCESS')
    parser.add_argument('-b', '--base_dir', help="Base dir where located 'data' folder", type=str, required=True,
                        dest='base_dir')
    parser.add_argument('-d', '--dimension', help="Dimensions", type=int, dest="dim", default=128)
    args = parser.parse_args()

    predictor = Predictor(args.dim, args.base_dir)
    texts = ["Estamos iniciando el Taller de valoración económica de los servicios ecosistémicos de la Reserva Marina de Galápagos. Gracias por la participación @parquegalapagos @GCRF @HeriotWattUni &amp, Scottish Funding Council https://t.co/YNCejTOn9i"]
    for text in texts:
        print(predictor.predict(text))
    print("DONE")
