#!/usr/bin/env python
# encoding: utf-8

# TODO: Include NLTK packages to git repository
# TODO: Investigate if it's reasonable to perform stemming on dataset

import numpy as np
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
This class provides conversion of text document into a feature vector
to be used in an SVM for text classification purposes. Initiate class with
a path to a file which should be convertet to a feature vectore for the
Word Kernel (WK). The feature vector can be collected immedeatly as:

>>> WK(/path/to/file).featureVector

'''
class WK(object):

    def __init__(self, directory):
        docs = self.filesFrom(directory))
        for document in docs:
            file_content = self.open(document)
            processed_text_list = self.rmStopWords(file_content)
        self.featureVector = self.featureVector(processed_text_list)

    def filesFrom(self, directory):
        return [
            file_name for file_name in os.listdir(directory)
            if file_name.endswith(".txt")
        ]

    def open(self, path):
        with open(path, 'r', encoding='utf-8') as myfile:
            return myfile.read()

    def rmStopWords(self, text):
        signs = [
            '.', ',', ':', ';', '-', '?',
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
    result = WK(path).vector
    print('Result: ', result)
    print('~*~ End Of Word Kernel ~*~')

if __name__ == '__main__':
    main()
