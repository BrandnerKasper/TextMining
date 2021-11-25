import nltk
import math
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
import operator
import argparse


def read_file(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def binom(k, n, p) -> float:
    return math.pow(p, k) * math.pow((1 - p), n - k)


def binom_part1(k, n, p) -> float:
    if p == 0:
        return 0
    return k * math.log(p)


def binom_part2(k, n, p) -> float:
    if p == 1:
        return 0
    return (n-k) * math.log(1-p)


def log_lambda(n, c12, c123, c3, p, p1, p2) -> float:
    b11 = binom_part1(c123, c12, p)
    b12 = binom_part2(c123, c12, p)
    b21 = binom_part1(c3 - c123, n - c12, p)
    b22 = binom_part2(c3 - c123, n - c12, p)
    b31 = binom_part1(c123, c12, p1)
    b32 = binom_part2(c123, c12, p1)
    b41 = binom_part1(c3 - c123, n - c12, p2)
    b42 = binom_part2(c3 - c123, n - c12, p2)
    return (b11 + b12) + (b21 + b22) - (b31 + b32) - (b41 + b42)


# Main method
if __name__ == '__main__':

    # Add parser to read command line arguments
    parser = argparse.ArgumentParser(description='Create the 20 most likely Trigrams from a given text file.')
    parser.add_argument('textfile', metavar='TEXTFILE', type=str,
                        help='Path to a txt file containing the text to parse to Trigrams.')
    args = parser.parse_args()

    # Read file
    example_text = read_file(args.textfile).lower()
    # example_text = read_file('static_idioms.txt').lower()
    stopwords = read_file('stopwords.txt').splitlines()

    # Tokenize text
    token = RegexpTokenizer(r"\w+").tokenize(example_text)

    # Filter stopwords
    filtered_words = [w for w in token if not w.lower() in stopwords]

    # Remove some silly characters # Would be more efficient with regex
    for w in filtered_words:
        try:
            if 'é' in w or 'è' in w or 'ö' in w or 'ä' in w or 'ü' in w or 'â' in w or 'ê' in w or 'ô' in w:
                filtered_words.remove(w)
                continue
            val = int(w)
            filtered_words.remove(w)
        except ValueError:
            continue

    # Make lists of bigrams and trigrams
    bigrams = nltk.bigrams(filtered_words)
    trigrams = list(ngrams(filtered_words, 3))

    # Count all words
    wordcount = len(filtered_words)

    # Make a dict for each word
    wordmap = {}
    for w in filtered_words:
        wordmap[w] = 0

    # Count words, bigrams, trigrams
    for w in filtered_words:
        if w.__eq__(w):
            wordmap[w] = wordmap.get(w) + 1
    frequence_bigrams = nltk.FreqDist(bigrams)
    frequence_trigrams = nltk.FreqDist(trigrams)

    dic = {'empty': 0}
    # Calculate likelihood for every trigram
    for t in trigrams:
        c1 = wordmap.get(t[0])
        c2 = wordmap.get(t[1])
        c3 = wordmap.get(t[2])
        bigram_t = tuple((t[0], t[1]))
        c12 = frequence_bigrams.get(bigram_t)
        c123 = frequence_trigrams.get(t)
        p = c3 / wordcount
        p1 = c123 / c12
        p2 = (c3 - c123) / (wordcount - 1 - c12)
        log_end = -2 * log_lambda(wordcount, c12, c123, c3, p, p1, p2)

        # Store score as value and the print statement as key
        value = log_end
        key = t.__str__() + ',' + '(' + log_end.__str__() + ', ' + c12.__str__() + ', ' + c3.__str__() + ', ' + c123.__str__() + ')'
        dic.update({key: value})

    # Sort the dictionary by score(value)
    sorted_d = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))

    # Print best 20
    counter = 0
    for item in sorted_d:
        if counter > 19:
            break
        print(item)
        counter = counter + 1
