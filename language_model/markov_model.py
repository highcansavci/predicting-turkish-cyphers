import time
import math

import numpy as np

from config import TURKISH_ALPHABET, TURKISH_DATASET_TEST_TXT_PATH, LOWEST_SCORE
from reader.reader import read_and_tokenize
from tokenizer.tokenizer import Tokenizer


class CypherModel:

    def __init__(self):
        self.vector_model = {}
        self.first_order_markov_model = {}
        self.second_order_markov_model = {}

    def fit(self, train_x):
        self.encode(train_x)

    def score(self, test_x, dna):
        return self.calculate_score(test_x, dna)

    def encode(self, train_x):
        self.encode_line(train_x)

    def real_score(self):
        return self.calculate_score(read_and_tokenize(TURKISH_DATASET_TEST_TXT_PATH), TURKISH_ALPHABET)

    def calculate_score(self, test_x, dna):
        # calculate the score of the dna
        score = np.zeros(len(test_x))
        index = 0
        for line in test_x:
            split_string = Tokenizer.tokenize(line)
            if len(split_string) == 0:
                continue
            if dna != TURKISH_ALPHABET:
                split_string = split_string.translate(split_string.maketrans(dna, TURKISH_ALPHABET))
            for i in range(len(split_string)):
                word = split_string[i]
                if i == 0:
                    try:
                        score[index] += self.vector_model[word]
                    except KeyError:
                        score[index] += math.log(LOWEST_SCORE)
                elif i == 1 or i == len(split_string) - 1:
                    try:
                        prev_word = split_string[i - 1]
                        score[index] += self.first_order_markov_model[prev_word][word]
                    except KeyError:
                        score[index] += math.log(LOWEST_SCORE)
                else:
                    try:
                        prev2_word = split_string[i - 2]
                        prev_word = split_string[i - 1]
                        score[index] += self.second_order_markov_model[prev2_word][prev_word][word]
                    except KeyError:
                        score[index] += math.log(LOWEST_SCORE)
            index += 1
        return score.mean()

    def encode_line(self, data):
        # build vector, first_markov and second_markov models
        for line in data:
            split_string = Tokenizer.tokenize(line)
            if len(split_string) == 0:
                continue
            for i in range(len(split_string)):
                word = split_string[i]
                if word not in TURKISH_ALPHABET:
                    continue
                if i == 0:
                    self.vector_model[word] = self.vector_model.get(word, 1) + 1
                elif i == 1 or i == len(split_string) - 1:
                    prev_word = split_string[i - 1]
                    if prev_word not in TURKISH_ALPHABET:
                        continue
                    self.first_order_markov_model[prev_word] = self.first_order_markov_model.get(prev_word, {})
                    self.first_order_markov_model[prev_word][word] = self.first_order_markov_model[prev_word].get(word,
                                                                                                                  1) + 1
                else:
                    prev2_word = split_string[i - 2]
                    if prev2_word not in TURKISH_ALPHABET:
                        continue
                    prev_word = split_string[i - 1]
                    if prev_word not in TURKISH_ALPHABET:
                        continue
                    self.second_order_markov_model[prev2_word] = self.second_order_markov_model.get(prev2_word, {})
                    self.second_order_markov_model[prev2_word][prev_word] = self.second_order_markov_model[prev2_word].get(
                        word, {})
                    self.second_order_markov_model[prev2_word][prev_word][word] \
                        = self.second_order_markov_model[prev2_word][prev_word].get(word, 1) + 1

        # normalization of vector_model
        total_vector_model_size = 10 ** -9
        for _, value in self.vector_model.items():
            total_vector_model_size += value
        for key, _ in self.vector_model.items():
            self.vector_model[key] /= total_vector_model_size
            self.vector_model[key] = math.log(self.vector_model[key])

        # normalization of the first order markov model
        for _, value in self.first_order_markov_model.items():
            total_first_markov_model_size = 10 ** -9
            for _, value_fo in value.items():
                total_first_markov_model_size += value_fo
            for key_fo, _ in value.items():
                value[key_fo] /= total_first_markov_model_size
                value[key_fo] = math.log(value[key_fo])

        # normalization of the second order markov model
        for _, value in self.second_order_markov_model.items():
            for _, value_fo in value.items():
                total_second_order_markov_model_size = 10 ** -9
                for _, value_so in value_fo.items():
                    total_second_order_markov_model_size += value_so
                for key_so, _ in value_fo.items():
                    value_fo[key_so] /= total_second_order_markov_model_size
                    value_fo[key_so] = math.log(value_fo[key_so])
