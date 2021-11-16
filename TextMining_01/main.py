# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nltk
import math
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Example function
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Read text file and make ONE string out of it (readline for one line as string, readlines for list of strings)
def read_file(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def binom(k, n, p) -> float:
    return math.pow(p, k) * math.pow((1 - p), n - k)


def log_lambda(n, c1, c2, c3, p, p1, p2) -> float:
    return math.log(binom(c3, c1, p)) + math.log(binom(c2 - c3, n - c1, p)) - math.log(binom(c3, c1, p1)) \
           - math.log(binom(c2 - c3, n - c1, p2))


# Press the green button in the guter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Read file
    example_text = read_file('static_idioms.txt')
    example_text = example_text.lower() # all words to lower case
    # print(example_text)
    stopwords = read_file('stopwords.txt')

    # Tokenize text
    token = nltk.word_tokenize(example_text)

    # Filter stopwords
    filtered_words = [w for w in token if not w.lower() in stopwords]

    # Remove some silly characters
    for w in filtered_words:
        if w.__eq__('-') or w.__eq__('ß') or w.__eq__('é'):
            filtered_words.remove(w)

    # print(filtered_words)

    # Make list of bigrams
    bigrams = nltk.bigrams(filtered_words)

    # Make list of trigrams
    trigrams = list(ngrams(filtered_words, 3))
    # print(trigrams)
    # frequence = nltk.FreqDist(trigrams)
    # for key, value in frequence.items():
    #     print(key, value)

    # Count all words
    wordcount = len(filtered_words)
    print(wordcount)

    # Make a Dict for each word
    wordmap = {}
    for w in filtered_words:
        wordmap[w] = 0

    # Count each word
    for w in filtered_words:
        if w.__eq__(w):
            wordmap[w] = wordmap.get(w) + 1



    print(wordmap)



    # Count bigrams
    frequence = nltk.FreqDist(bigrams)
    for key, value in frequence.items():
        print(key, value)

    # Calc Ps
    # Calc log




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
