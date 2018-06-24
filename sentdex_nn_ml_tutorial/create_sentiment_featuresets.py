"""
0. The data set consists of two documents, each containing about 5k sample phrases.
    The samples in one document have been labeled as having a positive sentiment, and the other a negative sentiment.
1. Textual data ingestion: Bag of words
    a. Create ordered lexicon (vocabulary) from unique words in all the training set.
    b. Vectorize each sample in the training or test set according to the lexicon.
2. Neural net classifier

Memory considerations:
1. Model (computational graph) vs. RAM:
    a. Number of hidden layers
    b. Number of nodes in each hidden layer
2. Lexicon size vs. RAM
    a. Some words can be considered equivalent, decreasing lexicon size:
        1) Stemming: run ~ run-s ~ run-ning
        2) Lemmatizing: run ~ ran
    a. Some words cannot be considered equivalent:
        1) Tense: like vs. used to like vs. liked

Therefore:
0.5 Natural Language (pre-)Processing:
    a. Tokenize samples to split each into a list of words
    b. Stem (or better yet, Lemmatize) the words (using NLTK's WordNet lookup)
    c. Ignore pitfalls involved in tense, etc.

3. Shuffling and counting...
4. Persist model using pickle.dump
"""

from collections import Counter
from functools import partial
from logging import DEBUG, basicConfig, debug, info
from os.path import join as path_join, sep
from pickle import dump, load
from random import shuffle
from sys import stdout
from typing import Tuple

from nltk import WordNetLemmatizer, download, word_tokenize
from numpy import array, ndarray, zeros
from tqdm import tqdm as tqdm_

download(['punkt', 'wordnet'])

tqdm = partial(tqdm_, file=stdout, mininterval=2)

max_lines = int(1e7)


def process_sample(sample: str):
    # TODO: Handle special characters, etc.
    return word_tokenize(sample.lower())


def create_bag_of_words(filenames: list):
    debug("filenames = %s" % str(filenames))
    n_lines = 0
    bag_of_words = []
    for filename in filenames:
        with open(filename) as file:
            contents = file.readlines()
            for line in tqdm(contents[:max_lines], desc="appending to bag-of-words from %s" % filename, unit="line"):
                words = process_sample(line)
                n_lines += 1
                bag_of_words += words
    info("len(bag_of_words) = %d" % len(bag_of_words))
    return n_lines, bag_of_words


lemmatizer = WordNetLemmatizer()


def create_word_counts(bag_of_words: list) -> dict:
    bag_of_lemmas = [lemmatizer.lemmatize(word) for word in tqdm(bag_of_words, desc="lemmatizing", unit="word")]
    info("len(bag_of_lemmas) = %d" % len(bag_of_lemmas))
    word_counts = Counter(bag_of_lemmas)
    return word_counts


def create_lexicon(filenames: list, rare=50, common=1000) -> Tuple[int, list]:
    n_lines, bag_of_words = create_bag_of_words(filenames)
    word_counts = create_word_counts(bag_of_words)

    lexicon = []
    for word, count in tqdm(word_counts.items(), desc="appending to lexicon from word counts", unit="word"):
        if rare < count < common:
            lexicon.append(word)
    info("len(lexicon) = %d" % len(lexicon))
    return n_lines, lexicon


def create_design_matrix_for_label(samples_filename, lexicon: list, is_positive_sentiment: bool):
    debug("samples_filename = %s" % samples_filename)
    design_matrix = []  # list of features+label pairs
    with open(samples_filename) as file:
        contents = file.readlines()
        for line in tqdm(contents[:max_lines], desc="creating design matrix from %s" % samples_filename, unit="line"):
            words = process_sample(line)

            words = [lemmatizer.lemmatize(word) for word in tqdm(words, desc="lemmatizing", unit="word")]

            # vectorize
            features = zeros(len(lexicon))
            for word in tqdm(words, desc="vectorizing", unit="word"):
                if word in lexicon:
                    index_of_word_in_lexicon = lexicon.index(word)
                    features[index_of_word_in_lexicon] += 1

            label = array([1, 0]) if is_positive_sentiment else array([0, 1])
            design_matrix.append([features, label])
    return design_matrix


def create_design_matrix(pos_filename, neg_filename, test_fraction=0.3):
    n_lines, lexicon = create_lexicon([pos_filename, neg_filename])

    design_matrix_positives = create_design_matrix_for_label(pos_filename, lexicon, True)
    design_matrix_negatives = create_design_matrix_for_label(neg_filename, lexicon, False)

    design_matrix = design_matrix_positives + design_matrix_negatives
    shuffle(design_matrix)
    design_matrix = array(design_matrix)

    n_test_samples = int(len(design_matrix) * test_fraction)

    x, y = design_matrix[:, 0], design_matrix[:, 1]
    x_train, y_train = x[:-n_test_samples], y[:-n_test_samples]
    x_test, y_test = x[-n_test_samples:], y[-n_test_samples:]

    return reallocate_ndarray(x_train), reallocate_ndarray(y_train), \
           reallocate_ndarray(x_test), reallocate_ndarray(y_test)


def reallocate_ndarray(a):
    a_ = ndarray([len(a), len(a[0])])
    for ndx, sample in enumerate(a):
        a_[ndx] = sample
    return a_


DATA_DIR = sep.join(("data", ""))
pickle_filepath = path_join(DATA_DIR, "sentiment_data.pickle")


def pickle_labeled_samples():
    debug("DATA_DIR = %s" % DATA_DIR)
    pos = path_join(DATA_DIR, "pos.txt")
    neg = path_join(DATA_DIR, "neg.txt")
    x_train, y_train, x_test, y_test = create_design_matrix(pos, neg)

    debug("pickle_filepath = %s" % pickle_filepath)
    with open(pickle_filepath, "wb") as pickle_file:
        dump([x_train, y_train, x_test, y_test], pickle_file)  # 140MB


def unpickle_design_matrix():
    with open(pickle_filepath, "rb") as file:
        design_matrix = load(file)
    x_train, y_train, x_test, y_test = design_matrix
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    basicConfig(level=DEBUG, stream=stdout)
    pickle_labeled_samples()
