#!/usr/bin/env python
# encoding: utf-8
"""
WK.py

This class provides conversion of text document into a feature vector
to be used in an SVM for text classification purposes.
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class WK(object):

    def __init__(self, doc_path):
        file_content = self.openFile(doc_path)
        print(file_content)
        processed_text = self.rmStopWords(file_content)

    def openFile(self, path):
        with open(path, 'r', encoding='utf-8') as myfile:
            text = myfile.read()
        return text

    def rmStopWords(self, text):
        return

# Currently osed for testing purposes,
# remove main() when done with testing.
def main():
    path = 'test_docs/alice_ch1.txt'
    result = WK(path)
    return

if __name__ == '__main__':
    main()
