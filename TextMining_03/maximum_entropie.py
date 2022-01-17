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


def create_tag_set() \
        -> set:
    tag_set_local = set()
    for line in text:
        if line.isspace():
            pass
        else:
            (word, tag) = line.strip().split(" ")
            tag_set_local.add(tag)
    return tag_set_local


def create_word_label_dic():
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


def create_word_previous_label_dic():
    print("\nWord - Previous Label : Label")
    word_previous_label_dic_local = {}
    last_tag = ''
    for line in tqdm(text):
        if line.isspace():
            last_tag = "<S>"
        else:
            (word, tag) = line.strip().split(" ")
            if (word, last_tag) not in word_previous_label_dic_local:
                word_previous_label_dic_local[word, last_tag] = {}
                for t in tag_set:
                    word_previous_label_dic_local[word, last_tag][t] = 0
            word_previous_label_dic_local[word, last_tag][tag] += 1
            last_tag = tag
    return word_previous_label_dic_local


def learning_function():


    return 0


# Main method
if __name__ == '__main__':
    print("Hi from Maximum Entropie :)")

    # TRAIN MODEL
    # Read text
    text = read_file_lines("BIO_formals_blank.txt")

    # Set with all different tags
    tag_set = create_tag_set()
    # print(tag_set)

    # words and tags and previous
    word_label_dic = create_word_label_dic()
    word_previous_label_dic = create_word_previous_label_dic()
    # print(word_label_dic)
    # print(word_previous_label_dic)




    # CLASSIFY




    print("End of Maximum Entropie ^^")
