#!/usr/bin/env python
# encoding: utf-8

import itertools
import numpy as np

from collections import Counter
from math import floor
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
Word kernel for text classification. Initialize class by:
>>> WK_kernel = WK([text1, text2,...])
'''

signs = [
    ',', '.', ';', ':', '?', '!', '-', '--', "'", '(', ')',
    '[', ']', '&', "'s", "''", '>', '``'
]
stop_words = stopwords.words("english") + signs

class WK(object):

    def __init__(self, docs):
        print("WK initialization started...")
        self.n = len(docs)
        self.docs = docs
        tokenized_docs = [word_tokenize(doc) for doc in docs]
        all_words = list(itertools.chain.from_iterable(tokenized_docs))
        filtered_words = [word for word in all_words if word not in stop_words]
        count_words = Counter(filtered_words)
        count_words = self.filterFewOccurences(count_words, 3)
        self.df = self.document_frequency(count_words, tokenized_docs)
        self.unique_words = self.df
        print("WK initialization complete")

    def kernel(self):
        def kernel_func(doc1, doc2):
            v = self.vectorize(doc1)
            w = self.vectorize(doc2)
            normalize = ( np.dot(v, v) * np.dot(w, w) )**0.5
            return np.dot(v, w)
        return kernel_func

    def vectorize(self, doc):
        v = []
        doc_count = Counter(word_tokenize(doc))
        v = [
            log( 1. + doc_count[word] ) *
            log( self.n / self.df[word] )
            for word in self.df.keys()
        ]
        norm = np.linalg.norm(v)
        if norm != 0:
            return np.divide(np.asarray(v),np.linalg.norm(v))
        return np.asarray(v)

    def featureVectors(self, docs):
        vectors = []
        for doc in docs:
            doc_count = Counter(word_tokenize(doc))
            v = [
                log( 1 + doc_count[word] ) *
                log( self.n / self.df[word] )
                for word in self.unique_words.keys()
            ]
            vectors.append(np.asarray(v))
        return vectors

    def filterFewOccurences(self, words, limit):
        for key, count in itertools.dropwhile(lambda key_count:
                key_count[1] > limit, words.most_common()):
            del words[key]
        return words

    def document_frequency(self, words, docs):
        for unique_word in words.keys():
            count = 0
            for doc in docs:
                if unique_word in doc:
                    count += 1
            words[unique_word] = count
        return words

    def gram_matrix(self):
        unique_words = [word for word in self.df.keys()]
        # Calculate feature vector and use these to calc gram matrix
        feature_vectors = self.featureVectors(self.docs)
        m = np.zeros((n, n))
        for idx, (doc1, doc2) in enumerate(itertools.product(feature_vectors, feature_vectors)):
            m[floor(idx/n)][idx%n] = np.dot(doc1, doc2) / \
                ( np.dot(doc1, doc1) * np.dot(doc2, doc2) )**0.5
        return m

    def __str__(self):
        return "WK"

# ___ This code if for testing purposes! ___

# Comment out main() when done with testing.
# def main():
#     indata = [
#         "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
#         "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.",
#         "There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again."
#     ]
#     text1 = "There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."
#     text2 = "So she was considering in her own mind (as well as she could, for hot day made her feel very sleepy and stupid), whether pleasure of making a daisy-chain would be worth trouble of getting up and picking daisies, when suddenly a White Rabbit with pink eyes ran close by her."
#     WK_kernel = WK(indata)
#     kernel = WK_kernel.kernel()
#     v = kernel(text1, text2)
#     print(v)
# #     print('Gram matrix:\n', WK_kernel.gram_matrix(indata))
# #     print('Feature vector:\n', WK_kernel.vectorize(text))
# #     print('~*~ End Of Word Kernel ~*~')
# #
# if __name__ == '__main__':
#     main()
