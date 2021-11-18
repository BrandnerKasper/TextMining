# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nltk
import math
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer


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


def log_lambda(n, c12, c123, c3, p, p1, p2) -> float:
    # if(n == 0 or c12 == 0 or c2 == 0 or c3 == 0 or p == 0 or p1 == 0 or p2 == 0):
    #     return 0

    b1 = binom(c123, c12, p)
    b2 = binom(c3 - c123, n - c12, p)
    b3 = binom(c123, c12, p1)
    b4 = binom(c3 - c123, n - c12, p2)
    log1 = 0
    if b1 != 0:
        log1 = math.log(b1)
    log2 = 0
    if b2 != 0:
        log2 = math.log(b2)
    log3 = 0
    if b3 != 0:
        log3 = math.log(b3)
    log4 = 0
    if b4 != 0:
        log4 = math.log(b4)

    return log1 + log2 - log3 - log4


# Press the green button in the guter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Read file
    example_text = read_file('static_idioms.txt')
    example_text = example_text.lower() # all words to lower case
    # print(example_text)
    stopwords = read_file('stopwords.txt')

    # Tokenize text
    tokenizer = RegexpTokenizer(r'\w+')
    token = tokenizer.tokenize(example_text)
    # token = nltk.word_tokenize(example_text)

    # Filter stopwords
    filtered_words = [w for w in token if not w.lower() in stopwords]

    # Remove some silly characters
    for w in filtered_words:
        try:
            val = int(w)
            filtered_words.remove(w)
        except ValueError:
            continue
    #print(filtered_words)

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

    #print(wordmap)

    # Count bigrams
    frequence_bigrams = nltk.FreqDist(bigrams)
    #for key, value in frequence_bigrams.items():
        #print(key, value)
        #print(type(key))

    # Count trigrams
    frequence_trigrams = nltk.FreqDist(trigrams)

    f = open("result.txt", "a", encoding='utf-8')
    f.truncate(0)
    # Calc Ps
    for t in trigrams:
        c1 = wordmap.get(t[0])
        c2 = wordmap.get(t[1])
        c3 = wordmap.get(t[2])
        bigram_t = tuple((t[0], t[1]))
        c12 = frequence_bigrams.get(bigram_t)
        c123 = frequence_trigrams.get(t)
        p = c3 / wordcount
        p1 = c123 / c12
        p2 = (c3 - c123) / (wordcount - c12)
        log_end = -2 * log_lambda(wordcount, c12, c123, c3, p, p1, p2)
        f.write(t.__str__() + ',' + '(' + log_end.__str__() + ', ' + c1.__str__() + ', ' + c2.__str__() + ', ' + c3.__str__() + ')')
        print(t.__str__() + ',' + '(' + log_end.__str__() + ', ' + c1.__str__() + ', ' + c2.__str__() + ', ' + c3.__str__() + ')')

    f.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
