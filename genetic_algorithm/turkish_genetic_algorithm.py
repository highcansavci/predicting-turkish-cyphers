import copy
import itertools
import time

import numpy as np
import random

from config import TURKISH_ALPHABET, DNA_POOL, OFFSPRING_COUNT, NUM_OF_EPOCH, PRINT_RESULTS


def decode_cypher(test_x, dna):
    for line in test_x:
        line = line.translate(line.maketrans(dna, TURKISH_ALPHABET))
        print(line)


def swap(dna):
    time.sleep(0.1)
    old = random.SystemRandom().randint(0, 29)
    new = random.SystemRandom().randint(0, 29)
    dna_copy = copy.deepcopy(dna)
    while old == new:
        time.sleep(0.1)
        new = random.SystemRandom().randint(0, 29)
    return dna_copy.translate(dna_copy.maketrans(dna_copy[old] + dna_copy[new], dna_copy[new] + dna_copy[old]))


def generate_random_dna():
    perm = np.random.permutation(30)
    dna = ""
    for number in perm:
        dna += TURKISH_ALPHABET[number]
    return dna


def print_accuracy(score_dict, i):
    print(f"Epoch Number: {i}")
    for dna, score in score_dict.items():
        print(f"{dna} has the log-prob {score}")


class GeneticAlgorithm:

    def __init__(self, cypher_model):
        self.cypher_model = cypher_model
        self.dna_pool = []
        self.create_initial_cypher_dna_pool()

    def create_initial_cypher_dna_pool(self):
        for i in range(DNA_POOL):
            self.dna_pool.append(generate_random_dna())

    def execute_algorithm(self, data, offspring_count, epoch_num):
        offspring_dna_pool = []
        for dna in self.dna_pool:
            for i in range(offspring_count):
                offspring_dna_pool.append(swap(dna))
        self.dna_pool.extend(offspring_dna_pool)
        score_dict = {}
        for dna in self.dna_pool:
            score_dict[dna] = self.cypher_model.score(data, dna)
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
        print_accuracy(dict(itertools.islice(score_dict.items(), DNA_POOL)), epoch_num)
        self.dna_pool = list(score_dict.keys())[:DNA_POOL]
        if PRINT_RESULTS or epoch_num == NUM_OF_EPOCH - 1:
            return self.dna_pool[0]

    def predict(self, data, num_of_epoch):
        for i in range(num_of_epoch):
            best_dna = self.execute_algorithm(data, OFFSPRING_COUNT, i)
            decode_cypher(data, best_dna)
