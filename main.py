from config import NUM_OF_EPOCH
from genetic_algorithm.turkish_genetic_algorithm import GeneticAlgorithm, generate_random_dna
from language_model.markov_model import CypherModel
from reader.reader import Reader
import time

if __name__ == '__main__':

    reader = Reader()
    cypher_model = CypherModel()
    genetic_algorithm = GeneticAlgorithm(cypher_model)

    print("***********  Split Data  ****************")
    start_time = time.time()
    sentences = reader.read_and_tokenize()
    start_time = time.time() - start_time
    print(f"Split data completed in {start_time} seconds.")

    print("************* Train Model ***************")
    start_time = time.time()
    cypher_model.fit(sentences)
    start_time = time.time() - start_time
    print(f"Training completed in {start_time} seconds.")

    print("********  Calculate Real Score  *********")
    start_time = time.time()
    real_score = cypher_model.real_score()
    start_time = time.time() - start_time
    print(f"Calculating the real score completed in {start_time} seconds.")

    print("**********  Generate Cypher  ************")
    start_time = time.time()
    random_test_cypher = generate_random_dna()
    test_data = reader.generate_cypher(random_test_cypher)
    start_time = time.time() - start_time
    print(f"Generating a cypher completed in {start_time} seconds.")

    print("**** Calculate Predictions and Scores ****")
    start_time = time.time()
    genetic_algorithm.predict(test_data, NUM_OF_EPOCH)
    start_time = time.time() - start_time
    print(f"Calculating the predictions and scores completed in {start_time} seconds.")


