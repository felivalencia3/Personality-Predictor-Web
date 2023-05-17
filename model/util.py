import nltk
import string


def build_bag_of_words_features_filtered(words):
    useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    words = nltk.word_tokenize(words)
    return {word: 1 for word in words if not word in useless_words}


def split_lines_efficient(text: str):
    individual = text.splitlines()
    size = 3
    if len(individual) <= size:
        return individual
    else:
        result = []
        div = len(individual) // size
        rem = len(individual) % size
        for i in range(div):
            build_str = ""
            for j in range(size):
                build_str += individual[i * size + j]
            result.append(build_str)
        last_str = ""
        for k in range(rem):
            last_str += individual[len(individual) - rem + k]
        result.append(last_str)
        return result
