#!/usr/bin/env python
# encoding: utf-8
"""
WK.py

This class provides conversion of text document into a feature vector
to be used in an SVM for text classification purposes.
"""


class WK(object):

    def __init__(self, doc_path):
        file_content = self.openFile(doc_path)

    def openFile(self, path):
        return


# Currently osed for testing purposes,
# remove main() when done with testing.
def main():
    path = 'test_docs/alice_ch1.txt'
    result = WK(path)
    return

if __name__ == '__main__':
    main()
