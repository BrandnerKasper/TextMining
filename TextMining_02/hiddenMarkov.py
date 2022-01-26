import os
from typing import List, Any, Union, DefaultDict
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json
from zipfile import ZipFile, ZIP_DEFLATED
import argparse


def trainmodel(trainingdatafilepath: str, zipfilepath: str):
    # ----------TRAIN MODEL--------------------------------------------
    # Training data
    text = read_file_lines(trainingdatafilepath)

    # All Sentences as List of words with their tag or empty line
    sentences = create_sentence_list(text)

    # Set with all different tags
    tag_set = create_tag_set(text)
    # List with the different!! tags
    tag_list = create_tag_list(tag_set)

    # aPriori Score
    apriori_scores = calculate_a_priori_probability(text)
    print("APriori with percentages: ")
    print(apriori_scores)

    # WordCount of training data
    wordcount = calculate_word_count(text)

    # Node Score with Probability
    node_scores = calculate_node_scores_probability(text)
    # print("NodeScore with percentages: ")
    # print(nodeScorePercentage)

    print("The different Tags: ")
    print(tag_list)

    transition_scores = calculate_transition_scores(tag_list, sentences)
    print("Here go the transitions: ")
    print(transition_scores)

    # transition probabilities
    #       O   C   B   I
    # O     00  01  02  03
    # C     10  11  12  13
    # B     20  21  22  23
    # I     30  31  32  33

    for i in range(0, len(transition_scores)):
        print(tag_list[i])
        for j in range(0, len(transition_scores[i])):
            print(transition_scores[i][j])

    # Dump files
    model_dictionary = {
        "tag_list": tag_list,
        "apriori": apriori_scores,
        "node_scores": node_scores,
        "transition_scores": transition_scores,
        "word_count": wordcount
    }

    open("model.json", "w").close()
    with open("model.json", 'w') as file_object:
        json.dump(model_dictionary, file_object, indent=1)

    # create a ZipFile object
    zipObj = ZipFile(zipfilepath, 'w', compression=ZIP_DEFLATED)
    zipObj.write('model.json')
    zipObj.close()

    os.remove("model.json")


def read_file(filepath)\
        -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_file_lines(filepath)\
        -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content


def create_sentence_list(text: List[str])\
        -> List[List[list]]:
    sentences_list = []
    sentence = []
    print("\nCreating list of sentences with (word,tag) pair")
    for line in tqdm(text):
        if line.isspace():
            sentences_list.append(sentence)
            sentence = []
        else:
            (word, tag) = line.strip().split(" ")
            sentence.append([word.lower(), tag])  # word, tag
            if text.index(line) == len(text) - 1:
                sentences_list.append(sentence)
                sentence = []
    return sentences_list


def create_tag_set(text: List[str])\
        -> set:
    tag_set = set()
    for line in text:
        if line.isspace():
            pass
        else:
            (word, tag) = line.strip().split(" ")
            tag_set.add(tag)
    return tag_set


def create_tag_list(tag_set: set) \
        -> List[str]:
    tag_list = []  # set is an unordered collection.
    for t in tag_set:
        tag_list.append(t)
    return tag_list


def calculate_word_count(text: List[str])\
        -> int:
    word_count = 0
    known_words = []
    for line in text:
        if line.isspace():
            pass
        else:
            (word, tag) = line.strip().split(" ")
            if word.lower() not in known_words:
                known_words.append(word.lower())
                word_count += 1
    return word_count


def calculate_a_priori_probability(text: List[str])\
        -> DefaultDict[str, Union[float, int]]:
    # Set with all different tags
    tag_set = set()
    a_priori: DefaultDict[str, Union[float, int]] = defaultdict(int)
    is_start_tag = True

    for line in text:
        if line.isspace():
            is_start_tag = True
        else:
            (word, tag) = line.strip().split(" ")
            tag_set.add(tag)
            if is_start_tag:
                a_priori[tag] += 1
                is_start_tag = False

    # aPriori add-one-smoothing
    for t in tag_set:
        a_priori[t] += 1

    total_values = sum(a_priori.values())
    a_priori_percentage = a_priori.copy()
    for o in a_priori_percentage:
        a_priori_percentage[o] = a_priori_percentage[o] / total_values

    return a_priori_percentage


def calculate_node_scores_probability(text: List[str])\
        -> DefaultDict[str, DefaultDict[str, Union[float, int]]]:
    # Node Scores
    node_score: DefaultDict[str, DefaultDict[str, Union[float, int]]] = defaultdict(lambda: defaultdict(int))

    print("Calculating Node Scores")
    for line in text:
        if line.isspace():
            pass
        else:
            (word, tag) = line.strip().split(" ")
            node_score[tag][word.lower()] += 1

    node_score_percentage = node_score.copy()
    for tag in node_score_percentage:
        total_value = sum(node_score_percentage[tag].values())
        for word in node_score_percentage[tag]:
            node_score_percentage[tag][word] /= total_value

    return node_score_percentage


def calculate_transition_scores(tag_list: List[str], sentence_list: List[List[list]])\
        -> List[List[int]]:
    # Calculate +1 to transitions not seen before!
    transition_tags = [[1] * len(tag_list) for i in range(len(tag_list))]

    for s in sentence_list:
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

    return transition_tags


def classify(sentence: str, tag_list: List[str], node_scores: DefaultDict[str, DefaultDict[str, Union[float, int]]],
             transition_scores: List[List[int]], apriori_scores: DefaultDict[str, Union[float, int]], word_count: int):
    # ----------HMM AND VITERBI--------------------------------------------
    # viterbi matrix with index from where value came from
    #       This    is      a       sentence    .
    # O     00, 0   01, 0   02, 0   03, 0       04, 0
    # C     10, 0   11, 0   12, 0   13, 0       14, 0
    # B     20, 0   21, 0   22, 0   23, 0       24, 0
    # I     30, 0   31, 0   32, 0   33, 0       34, 0

    # An example sentence to calculate viterbi on
    # sentence = "The sound was music to her ears ."
    sentence.lower()
    sentence = sentence.split(" ")

    # Viterbi Matrix
    viterbi_matrix = calculate_viterbi_matrix(sentence, tag_list, node_scores, transition_scores, apriori_scores, word_count)

    # The most probable tag combination of the given sentence
    best_track = calculate_best_tag_combination(sentence, tag_list, viterbi_matrix)

    # Calculate Viterbi Score
    viterbi_score = calculate_viterbi_score(viterbi_matrix, tag_list, sentence)

    print("Reference: " + str(sentence))
    print("Our Solution: " + str(best_track))
    print("Our max_prob: " + str(viterbi_score))


def calculate_viterbi_matrix(sentence: List[str], tag_list: List[str], node_scores: DefaultDict[str, DefaultDict[str, Union[float, int]]],
                             transition_scores: List[List[int]], apriori_scores: DefaultDict[str, Union[float, int]], word_count: int)\
        -> List[List[List[int]]]:
    viterbi_matrix = [[[0, 0]] * len(sentence) for i in range(len(tag_list))]

    # Viterbi Score for first word
    for t in range(len(tag_list)):
        node_score = node_scores.get(tag_list[t]).get(sentence[0])
        if node_score is None:
            node_score = 1 / (1 + word_count)
        viterbi_matrix[t][0] = [np.log(apriori_scores.get(tag_list[t])) + np.log(node_score), -1]

    # Viterbi score for every other word
    for j in range(len(sentence) - 1):
        for i in range(len(tag_list)):
            node_score = node_scores.get(tag_list[i]).get(sentence[j + 1])
            if node_score is None:
                node_score = 1 / (1 + word_count)
            max_current = -100000000000000000
            index_origin = -1
            for k in range(len(tag_list)):
                node_score_before = viterbi_matrix[k][j][0]
                transition_score = transition_scores[k][i]
                curr_score = node_score_before + np.log(transition_score)
                if curr_score > max_current:
                    max_current = curr_score
                    index_origin = k
            viterbi_matrix[i][j + 1] = [max_current + np.log(node_score), index_origin]

    return viterbi_matrix


def calculate_best_tag_combination(sentence: List[str], tag_list: List[str], viterbi_matrix: List[List[List[int]]])\
        -> List[str]:
    max_prob = -10000000000000
    last_i = -1
    for i in range(len(tag_list)):
        curr_prob = viterbi_matrix[i][len(sentence) - 1][0]
        if curr_prob > max_prob:
            last_i = i
            max_prob = curr_prob

    best_tag_combination = [tag_list[last_i]]  # sequence with solution tags reversed
    for j in range(len(sentence) - 1, 0, -1):
        index_before = viterbi_matrix[last_i][j][1]
        best_tag_combination.append(tag_list[index_before])
        last_i = index_before

    best_tag_combination.reverse()  # sequence with solution tags

    return best_tag_combination


def calculate_viterbi_score(viterbi_matrix: List[List[List[int]]], tag_list: List[str], sentence: List[str]) -> float:
    max_prob = -10000000000000

    for i in range(len(tag_list)):
        curr_prob = viterbi_matrix[i][len(sentence) - 1][0]
        if curr_prob > max_prob:
            max_prob = curr_prob

    return max_prob


# Main method
if __name__ == '__main__':
    print("Hi from Markov :)")

    # Add parser to read command line arguments
    parser = argparse.ArgumentParser(description='Train a hidden markov model')
    parser.add_argument('command',
                        help='Either call "train" or "classify". '
                             'Train needs a training data file and a model .zip file to save.'
                             'Classify needs valid model file.')
    parser.add_argument('arg1',
                        help='Using Train: Path to a txt file containing the text for training the model.\n'
                             'Using Classify: Path to the trained model file')
    parser.add_argument('arg2',
                        help='Using Train: Path to a .zip file to save the trained model\n'
                             'Using Classify: The sentence to be classified')
    args = parser.parse_args()

    if args.command == "train":
        trainmodel(args.arg1, args.arg2)
    elif args.command == "classify":
        # zipObj.extractall()
        with ZipFile(args.arg1, 'r') as modelzip:
            with modelzip.open("model.json") as modeljson:
                model_dictionary = json.load(modeljson)
        classify(args.arg2, model_dictionary["tag_list"], model_dictionary["node_scores"], model_dictionary["transition_scores"], model_dictionary["apriori"], model_dictionary["word_count"])

    print("End of Markov ^^")
