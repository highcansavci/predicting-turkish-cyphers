import re
import string
from os.path import exists

from config import TURKISH_DATASET_TXT_PATH, TURKISH_DATASET_URL, TURKISH_ALPHABET, TURKISH_DATASET_TEST_TXT_PATH


def read_and_tokenize(dataset_txt_path):
    with open(dataset_txt_path, 'r', encoding="utf-8") as f:
        content = f.read().replace('\n', '')
    return list(map(str.strip, re.split(r"[.!?]", content)))


def generate_cypher(test_txt_path, dna):
    with open(test_txt_path, 'r', encoding="utf-8") as f:
        content = f.read().replace('\n', '')
        content = content.translate(content.maketrans(TURKISH_ALPHABET, dna))
    return list(map(str.strip, re.split(r"[.!?]", content)))


class Reader:

    def __init__(self):
        if exists(TURKISH_DATASET_TXT_PATH):
            self.dataset_txt_path = TURKISH_DATASET_TXT_PATH
        else:
            print(f"The dataset does not exist. You can get it from here: {TURKISH_DATASET_URL}."
                  f"Special thanks for the preparation of the dataset.")
        self.dataset_test_txt_path = TURKISH_DATASET_TEST_TXT_PATH

    def read_and_tokenize(self):
        return read_and_tokenize(self.dataset_txt_path)

    def generate_cypher(self, dna):
        return generate_cypher(self.dataset_test_txt_path, dna)
