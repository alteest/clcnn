import unicodedata
import unidecode
import string
import numpy as np
from slugify import slugify

categories = 'LMNPSZC'

bidirectional = {8: ['L'],
                 9: ['R', 'AL'],
                 10: ['EN', 'ES', 'ET', 'AN', 'CS', 'NSM', 'BN'],
                 11: ['B', 'S', 'WS', 'ON'],
                 }


class Converter:
    def __init__(self, size: int):
        self.size = size

    def convert_char(self, char: string, first: bool) -> np.ndarray:
        # letter, mark, number, punctuation, symbol, separator, or other.
        # https://www.fileformat.info/info/unicode/category/index.htm
        category = unicodedata.category(char)
        data = np.zeros(128, dtype=np.int8)
        if category:
            ind = categories.find(category[0])
            if ind > -1:
                data[ind] = 1
                if ind == 0 and category == 'Lu':
                    data[7] = 1
            else:
                print(f"Unknown category '{category}' for character '{char}'")
        else:
            print(f"Empty category for '{char}'")

        # bidirectional
        # https://www.unicode.org/reports/tr9/#Bidirectional_Character_Types
        bi = unicodedata.bidirectional(char)
        ind = 12
        for i, values in bidirectional.items():
            if bi in values:
                ind = i
                break
        data[ind] = 1

        # FIXME NFD???
        dec = unicodedata.decomposition(char)
        if dec:
            # nfd = unicodedata.normalize('NFD', char)
            for part in dec.split():
                code = int(part, 16)
                ind = (code * 100003) % 16 + 13
                data[ind] = 1

        if char == '#':
            data[29] = 1
        elif char == '@':
            data[30] = 1

        tr = slugify(char)
        if tr:
            ind = ord(tr) - ord('a') + 31
            if 31 <= ind <= 56:
                data[ind] = 1
            else:
                # MAY BE digit
                pass
                # print(f"Invalid transliteration of {tr} char {char}")

        if char in string.printable:
            data[57] = 1
        else:
            ind = (ord(char) * 100003) % 69 + 59
            data[ind] = 1

        if first:
            data[58] = 1

        return data

    def convert_text(self, text: str) -> [list, None]:

        values = None
        for norm_char in unicodedata.normalize("NFKC", text):
            chars = unidecode.unidecode(norm_char)
            first = True
            for ch in chars:
                val = self.convert_char(ch, first)
                first = False
                if values is None:
                    values = np.array([val])
                else:
                    values = np.append(values, [val], axis=0)
            # print(first(ch))
        if values is not None:
            rows = values.shape[0]
            if rows > 280:
                print(f"Got {rows} rows with text '{text}'. SKIPPING")
                return None
            if rows < 280:
                values = np.append(values, np.zeros((280 - rows, 128), np.int8), axis=0)
            return values  # .tolist()
        print(f"ERROR! Got None from '{text}'")
        return None


if __name__ == "__main__":
    converter = Converter(128)
    # test_str = "this Is, test! даè東"
    test_str = "Indemnización a familias víctimas del conflicto armado en Leguízamo  Mediante una labor articulada entre la @UnidadVictimas Regional Putumayo,  la Administración Municipal, se entregó 32 cartas de indemnización a adultos mayores de la localidad. Rubén Velásquez Alvarado, alcalde. https://t.co/RhQlEf9PXv"
    print(converter.convert_text(test_str))
