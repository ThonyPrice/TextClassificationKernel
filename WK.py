#!/usr/bin/env python
# encoding: utf-8

# TODO: Function that takes a list of strings and processes every string in it

import itertools
import numpy as np
import os

from collections import Counter
from math import floor
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
This class provides conversion of text documents into a gram matrix
to be used in an SVM for text classification purposes. Example:
>>> WK([text1, text2,...]).gram_matrix
'''

class WK(object):

    def __init__(self, docs):
        n = len(docs)
        # Join all text to find unique words and word frequency
        all_words = Counter(word_tokenize(" ".join(docs)))
        all_words = self.filterFewOccurences(all_words, 2)
        df = self.document_frequency(all_words, docs)
        unique_words = [word for word in df.keys()]
        # Calculate feature vector and use these to calc gram matrix
        feature_vectors = self.featureVectors(docs, unique_words, df, n)
        m = np.zeros((n, n))
        for idx, (doc1, doc2) in enumerate(itertools.product(feature_vectors, feature_vectors)):
            m[floor(idx/n)][idx%n] = np.dot(doc1, doc2) / \
                ( np.dot(doc1, doc1) * np.dot(doc2, doc2) )**0.5
        self.gram_matrix = m

    def filterFewOccurences(self, words, limit):
        return {x : words[x] for x in words if words[x] > limit}

    def document_frequency(self, words, docs):
        for word in words.keys():
            words[word] = sum([1 for doc in docs if word in doc])
        return words

    def featureVectors(self, docs, unique_words, df, n):
        vectors = []
        for doc in docs:
            doc_count = Counter(word_tokenize(doc))
            v = [
                log( 1 + doc_count[word] ) *
                log( n / df[word] ) for word in unique_words
            ]
            vectors.append(np.asarray(v))
        return vectors

# Currently used for testing purposes,
# remove main() when done with testing.
def main():
    indata = [
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
        "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.",
        "There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again."
    ]
    result = WK(indata).gram_matrix
    print('Result: ', result)
    print('~*~ End Of Word Kernel ~*~')

if __name__ == '__main__':
    main()
