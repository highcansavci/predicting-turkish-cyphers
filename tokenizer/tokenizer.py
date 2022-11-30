import string
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from reader.reader import read_and_tokenize


class Tokenizer:
    @classmethod
    def tokenize(cls, line):
        return line.lower().translate(str.maketrans('', '', string.punctuation))

    @classmethod
    def custom_tokenizer(cls, line):
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        line = line.lower()
        tokens = nltk.tokenize.word_tokenize(line)
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
        return tokens


class CustomCountVectorizer:

    @classmethod
    def fit(cls, data_instance):
        unique_words = set()

        for each_sentence in data_instance:
            for each_word in Tokenizer.tokenize(each_sentence):
                unique_words.add(each_word)

        vocab = {}
        for index, word in enumerate(sorted(list(unique_words))):
            vocab[word] = index

        return vocab

    @classmethod
    def transform(cls, data_instance):
        vocab = CustomCountVectorizer.fit(data_instance)
        row, col, val = [], [], []

        for idx, sentence in enumerate(data_instance):
            count_word = dict(Counter(Tokenizer.tokenize(sentence)))

            for word, count in count_word.items():
                col_index = vocab.get(word)
                if col_index >= 0:
                    row.append(idx)
                    col.append(col_index)
                    val.append(count)

        return csr_matrix((val, (row, col)), shape=(len(data), len(vocab)))


if __name__ == "__main__":
    data = read_and_tokenize("../raven.txt")
    vectorizer = CountVectorizer()
    sklearn_output = vectorizer.fit_transform(data).toarray()
    custom_output = CustomCountVectorizer.transform(data).toarray()

    print(sklearn_output)
    print("*" * 40)
    print(custom_output)
