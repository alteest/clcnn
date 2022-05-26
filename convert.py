import unicodedata
import string
import numpy as np
from slugify import slugify

categories = 'LMNPSZC'

bidirectional = {8: ['L'],
                 9: ['R', 'AL'],
                 10: ['EN', 'ES', 'ET', 'AN', 'CS', 'NSM', 'BN'],
                 11: ['B', 'S', 'WS', 'ON'],
                 }


def convert(char):
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
            print(f"Invalid transliteration of {tr} char {char}")

    if char in string.printable:
        data[57] = 1
    else:
        ind = (ord(char) * 100003) % 69 + 59
        data[ind] = 1

    # TODO 58 : first from transliteration

    return data


test_str = "this Is, test! даè"
values = None
print(values)
for ch in unicodedata.normalize("NFKC", test_str):
    val = first(ch)
    if values is None:
        values = np.array([val])
    else:
        values = np.append(values, [val], axis=0)
    # print(first(ch))
print(values)
print(values.shape)
