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

    print("Example Text: \n" + read_file("example_text.txt"))
    example = read_file_lines("example_text.txt")
    isStartTag = True
    # aPriori Score
    aPriori: DefaultDict[str, Union[float, int]] = defaultdict(int)
    # Set with all different tags
    tagSet = set()
    # Node Scores
    nodeScore: DefaultDict[str, DefaultDict[str, Union[float, int]]] = defaultdict(lambda: defaultdict(int))

    for line in example:
        if line.isspace():
            isStartTag = True  # empty line -> new sentence
        else:
            (word, tag) = line.strip().split(" ")
            nodeScore[tag][word] += 1
            tagSet.add(tag)
            if isStartTag:
                aPriori[tag] += 1
                isStartTag = False

    # aPriori Score with Probability
    for t in tagSet:
        aPriori[t] += 1

    print("APriori with values: ")
    print(aPriori)
    total = sum(aPriori.values())
    aPrioriPercentage = aPriori.copy()
    for o in aPrioriPercentage:
        aPrioriPercentage[o] = aPrioriPercentage[o] / total
    print("APriori with percentages: ")
    print(aPrioriPercentage)

    # Node Score with Probability
    print("NodeScore with values: ")
    print(nodeScore)
    nodeScorePercentage = nodeScore.copy()
    for tag in nodeScorePercentage:
        total = sum(nodeScorePercentage[tag].values())
        for word in nodeScorePercentage[tag]:
            nodeScorePercentage[tag][word] /= total
    print("NodeScore with percentages: ")
    print(nodeScorePercentage)
