import nltk
from nltk.tokenize import RegexpTokenizer
from typing import List, Any, Union, DefaultDict
from collections import defaultdict


def read_file(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_file_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content


# Main method
if __name__ == '__main__':
    print("Hi from Markov :)")

    example = read_file("example_text.txt")
    print(example)

    # Alternative
    example02 = read_file_lines("example_text.txt")
    # map = {}  # word token
    # map2 = {}  # token word
    isStartToken = True
    # aPriori Score
    aPriori: DefaultDict[str, Union[float, int]] = defaultdict(int)  # maybe without default init
    valuesSet = set()
    # Node Scores
    nodeScore: DefaultDict[str, DefaultDict[str, Union[float, int]]] = defaultdict(lambda: defaultdict(int))

    for line in example02:
        if line.isspace():
            isStartToken = True  # leerzeile -> neuer Satz
        else:
            (word, token) = line.strip().split(" ")
            nodeScore[token][word] += 1
            valuesSet.add(token)
            if isStartToken:
                aPriori[token] += 1
                isStartToken = False

    print(map)

    # aPriori Score with Probability
    for v in valuesSet:
        aPriori[v] += 1

    print(aPriori)
    total = sum(aPriori.values())
    aPrioriPercentage = aPriori.copy()
    for o in aPrioriPercentage:
        aPrioriPercentage[o] = aPrioriPercentage[o] / total
    print(aPrioriPercentage)

    # Node Score with Probability
    print(nodeScore)
    nodeScorePercentage = nodeScore.copy()
    for token in nodeScorePercentage:
        total = sum(nodeScorePercentage[token].values())
        for word in nodeScorePercentage[token]:
            nodeScorePercentage[token][word] /= total

    print(nodeScorePercentage)

    # Tokenize text
    # token = RegexpTokenizer(r"\w+-?’?\w*|[\.,]").tokenize(example)
    #
    # keys = []
    # values = []
    # split = True
    # for w in token:
    #     print(w)
    #     if split:
    #         keys.append(w)  # every odd word count is a key
    #         split = False
    #     else:
    #         values.append(w)  # every just word count is a value
    #         split = True

    # for k in keys:
    #     print(k)

    # for v in values:
    #     print(v)

    # wordmap = dict(zip(keys, values))
    # print(wordmap)
