import string


class Tokenizer:
    @classmethod
    def tokenize(cls, line):
        return line.lower().translate(str.maketrans('', '', string.punctuation))
