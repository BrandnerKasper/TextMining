from __future__ import annotations
from typing import List, Any, Union, DefaultDict
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def read_file(filepath) \
        -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_file_lines(filepath) \
        -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content


def create_tag_set(text) \
        -> set:
    tag_set_local = set()
    for line in text:
        if line.isspace():
            pass
        else:
            (word, tag) = line.strip().split(" ")
            tag_set_local.add(tag)
    return tag_set_local


def create_word_label_dic(text, tag_set) \
        -> dict[str, dict[str, int]]:
    print("\nWord : Label")
    word_label_dic_local = {}
    for line in tqdm(text):
        if line.isspace():
            continue
        else:
            (word, tag) = line.strip().split(" ")
            # word and label
            if word not in word_label_dic_local:
                word_label_dic_local[word] = {}
                for t in tag_set:
                    word_label_dic_local[word][t] = 0
            word_label_dic_local[word][tag] += 1
    return word_label_dic_local


# give a tagset .-.
def create_word_previous_label_dic(text, tag_set_with_start_token) \
        -> dict[(str, str), dict[str, int]]:
    print("\nWord - Previous Label : Label")
    word_previous_label_dic_local = {}
    last_tag = "<S>"
    for line in tqdm(text):
        if line.isspace():
            last_tag = "<S>"
        else:
            # the actual combination of last tag, current tag and word we found
            (word, tag) = line.strip().split(" ")
            if (word, last_tag) not in word_previous_label_dic_local:
                word_previous_label_dic_local[word, last_tag] = {}
                # init all combinations of word last tag (we found) and all the different curr tags
                for t in tag_set_with_start_token:
                    word_previous_label_dic_local[word, last_tag][t] = 0
            word_previous_label_dic_local[word, last_tag][tag] += 1
            last_tag = tag
            # init all combinations of word plus all different last tags with all diff curr tags
            for ltag in tag_set_with_start_token:
                if (word, ltag) not in word_previous_label_dic_local:
                    word_previous_label_dic_local[word, ltag] = {}
                    for ctag in tag_set_with_start_token:
                        word_previous_label_dic_local[word, ltag][ctag] = 0

    return word_previous_label_dic_local


# possible to use for 2 different features
def create_feature_vector_from_dic(example_dic) -> (list, dict[int, (str, str) or (str, str, str)]):
    # feature_vector = []
    # for (key, value) in example_dic.items():
    #     feature_vector.append(value)
    pass


def learning_function():
    pass


# Main method
def main():
    # TRAIN MODEL
    # Read text
    text = read_file_lines("BIO_formals_blank.txt")

    # Set with all different tags
    tag_set = create_tag_set(text)
    tag_set_with_start_token = set()
    tag_set_with_start_token.add("<S>")
    tag_set_with_start_token |= tag_set
    print(tag_set)
    print(tag_set_with_start_token)

    # words and tags and previous
    word_label_dic = create_word_label_dic(text, tag_set)
    word_previous_label_dic = create_word_previous_label_dic(text, tag_set_with_start_token)
    # print(word_label_dic)
    # print(word_previous_label_dic)


if __name__ == '__main__':
    print("Hi from Maximum Entropie :)")
    main()
    print("End of Maximum Entropie ^^")
