#!/usr/bin/env python
# encoding: utf-8

# TODO: Include NLTK packages to git repository
# TODO: Investigate if it's reasonable to perform stemming on dataset
# TODO: Make sure each document one get one feature vector
# TODO: Each pair of vectors get a value decided by the dot
#       product of their feature vectors
# TODO: Put all values into a gram matrix
# TODO: Return Gram matrix instead ESSENTIAL before testing with SVM

from collections import Counter
import itertools
from math import log
import os

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


'''
This class provides conversion of text documents into a feature vector
to be used in an SVM for text classification purposes. Initiate class
with a path to a directory with documents that should be convertet to a
feature vector for the Word Kernel (WK). The feature vector can be
collected immedeatly as:

>>> WK(/path/to/file).featureVector

Notice (dependencies):
The current version requires NLTK (use pip install),
decuments must be .txt files (easily modified),
document encoding must be UTF-8.
'''


class WK(object):

    def __init__(self, directory):
        docs = self.filesFrom(directory)
        n = len(docs)
        content_list = self.extract(docs)
        content_list = self.process(content_list)
        separate_doc_counts = [Counter(content) for content in content_list]
        total_count = Counter(list(itertools.chain.from_iterable(content_list)))
        vector = [
            log(1 + total_count[x])
            # Sum below is document frequency
            * (n / sum([1 for count in separate_doc_counts if x in count]))
            for x in total_count
            if total_count[x] > 3
        ]
        self.featureVector = np.asarray(vector)

    def filesFrom(self, directory):
        return [
            directory + file_name
            for file_name in os.listdir(directory)
            if file_name.endswith(".txt")
        ]

    def extract(self, docs):
        return [self.open(document) for document in docs]

    def open(self, path):
        with open(path, 'r', encoding='utf-8') as myfile:
            return myfile.read()

    def process(self, docs):
        return [self.rmStopWords(document) for document in docs]

    def rmStopWords(self, text):
        signs = [
            '.', ',', ':', ';', '-', '?', '--'
            '!', "'", '(', ')', '[', ']'
        ]
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        return [
            w for w in word_tokens
            if not w in stop_words
            and w not in signs
        ]

    def featureVector(self, text_list):
        unique_words_and_ocurrences = Counter(text_list)
        ocurrences = list(unique_words_and_ocurrences.values())
        return np.asarray(ocurrences)


# Currently used for testing purposes,
# remove main() when done with testing.
def main():
    path = './test_docs/'
    result = WK(path).featureVector
    print('Result: ', result)
    print('~*~ End Of Word Kernel ~*~')

if __name__ == '__main__':
    main()
