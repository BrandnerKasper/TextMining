import nltk
from nltk.tokenize import RegexpTokenizer
from typing import List, Any, Union, DefaultDict
from collections import defaultdict
import numpy as np


def read_file(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_file_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content

# sentence ['a', 'Idion']
def viterbi(sentence, tag_list, word_probs, transition_probs, aPrioriPercentage) -> float:
    viterbi_mat = [[0] * len(sentence) for i in range(len(tag_list))]
    #for a in range(len(sentence)):
       # for t in tag_list:
          #  if sentence[a][1] == t:
               # aPrioriPercentage
        #viterbi_mat[0][a] = 0
    for i in range(len(tag_list)):
        for j in range(len(sentence)):
            viterbi_mat[i][j] = 1  # max() #word prob#

    print(viterbi_mat)
    return 0


# Main method
if __name__ == '__main__':
    print("Hi from Markov :)")

    #print("Example Text: \n" + read_file("example_text.txt"))

    # text = read_file_lines("example_text.txt")
    text = read_file_lines("BIOformal.txt")
    isStartTag = True
    # aPriori Score
    aPriori: DefaultDict[str, Union[float, int]] = defaultdict(int)
    # Set with all different tags
    tagSet = set()
    # Node Scores
    nodeScore: DefaultDict[str, DefaultDict[str, Union[float, int]]] = defaultdict(lambda: defaultdict(int))

    sentences = []
    sentence = []
    sentence_count = 0
    for line in text:
        if line.isspace():
            isStartTag = True  # empty line -> new sentence
            sentences.append(sentence)
            sentence_count = sentence_count + 1
            sentence = []
        else:
            (word, tag) = line.strip().split(" ")
            sentence.append([word, tag])  # word, tag
            nodeScore[tag][word] += 1
            tagSet.add(tag)
            if isStartTag:
                aPriori[tag] += 1
                isStartTag = False
            if text.index(line) == len(text) - 1:
                sentences.append(sentence)
                sentence_count = sentence_count + 1
                sentence = []

    # aPriori Score with Probability
    for t in tagSet:
        aPriori[t] += 1

    # print("APriori with values: ")
    # print(aPriori)
    total = sum(aPriori.values())
    aPrioriPercentage = aPriori.copy()
    for o in aPrioriPercentage:
        aPrioriPercentage[o] = aPrioriPercentage[o] / total
    # print("APriori with percentages: ")
    # print(aPrioriPercentage)

    # Node Score with Probability
    # print("NodeScore with values: ")
    # print(nodeScore)
    nodeScorePercentage = nodeScore.copy()
    for tag in nodeScorePercentage:
        total = sum(nodeScorePercentage[tag].values())
        for word in nodeScorePercentage[tag]:
            nodeScorePercentage[tag][word] /= total
    # print("NodeScore with percentages: ")
    # print(nodeScorePercentage)

    # transition probabilities
    #       O   C   B   I
    # O     00  01  02  03
    # C     10  11  12  13
    # B     20  21  22  23
    # I     30  31  32  33

    tag_list = []  # set is an unordered collection.
    for t in tagSet:
        tag_list.append(t)

    transition_tags = [[0] * len(tagSet) for i in range(len(tagSet))]

    for s in sentences:
        for w in range(0, len(s) - 1):
            matrix_index_from = -1
            matrix_index_to = -1
            for i in range(0, len(tag_list)):
                if s[w][1] == tag_list[i]:  # find current tag
                    matrix_index_from = i
                if s[w + 1][1] == tag_list[i]:  # next current tag
                    matrix_index_to = i
                if matrix_index_from > -1 and matrix_index_to > -1:
                    break
            transition_tags[matrix_index_from][matrix_index_to] = transition_tags[matrix_index_from][
                                                                      matrix_index_to] + 1

    for t in range(0, len(transition_tags)):
        sum_trans = 0
        for i in range(0, len(transition_tags[t])):
            sum_trans = sum_trans + transition_tags[t][i]
        for u in range(0, len(transition_tags[t])):
            transition_tags[t][u] = transition_tags[t][u] / sum_trans

    print(transition_tags)

    # viterbi mattric
    #       This   is   a   sentence    .
    # O     00     01   02  03          04
    # C     10     11   12  13          14
    # B     20     21   22  23          24
    # I     30     31   32  33          34

    sentence = sentences[0]
    viterbi_mat = [[0] * len(sentence) for i in range(len(tag_list))]
    best_track = [] # sequence with tags

    print(sentence[0][0])
    for t in range(len(tag_list)):
        viterbi_mat[t][0] = aPrioriPercentage.get(tag_list[t])
        node_score = nodeScorePercentage.get(tag_list[t]).get(sentence[0][0])
        if node_score is None:
            node_score = 1
        viterbi_mat[t][0] = np.log(aPrioriPercentage.get(tag_list[t])) + np.log(node_score) # plus or what or log or foo

    for i in range(len(tag_list)):
        for j in range(len(sentence) - 1):
            node_score = nodeScorePercentage.get(tag_list[i]).get(sentence[j + 1][0])
            if node_score is None:
                node_score = 1
            max_current = 0
            for k in range(len(tag_list)):
                node_score_before = nodeScorePercentage.get(tag_list[k]).get(sentence[j][0])  # node score before
                if node_score_before is None:
                    node_score_before = 1
                transition_score = transition_tags[k][i]  # not sure
                if transition_score == 0:
                    transition_score = 1
                curr_score = np.log(node_score_before) + np.log(transition_score)
                if curr_score > max_current:
                    max_current = curr_score
            viterbi_mat[i][j + 1] = max_current + np.log(node_score)

    max_prob = 0
    last_i = -1
    for i in range(len(tag_list)):
        if viterbi_mat[i][len(sentence)-1] > max_prob:
            max_prob = viterbi_mat[i][len(sentence)-1]
            last_i = 1

    for i in range(len(tag_list)):
        line = ""
        for j in range(len(sentence)):
            # line += i.__str__() + "|" + j.__str__() + ": " + viterbi_mat[i][j].__str__() + " "
            line += viterbi_mat[i][j].__str__() + " "
        print(line)
    print(max_prob.__str__() + " " + last_i.__str__())
    # print(nodeScorePercentage)
    print(sentences[0])

    #viterbi(sentences[0], tag_list, 5, 5, aPrioriPercentage)

    print("End of Markov")
