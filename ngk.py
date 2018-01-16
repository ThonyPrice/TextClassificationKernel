# N-Gram kernel
from collections import Counter
from math import sqrt
import itertools as it
import numpy as np
import reut
from nltk.corpus import reuters

class NGK():
    """
    Class for N-gram kernel
    Can compare the similarity between two documents (sequences).
    Is initialized with the number of n-grams wanted.
    """

    def __init__(self, n):
        self.n = n

    def vectorize(self, doc):
        """Returns the documented converted into a set containing its n-grams."""
        words = []
        for i in range(self.n):
            words.append(doc[:i])
        for i in range(len(doc)):
            words.append(doc[i:i+self.n])

        return words

    def similarity(self, doc1, doc2):
        """Returns the similarity between two documents."""
        return self.cosine_similarity(doc1, doc2)

    def jaccard_similarity(self, doc1, doc2):
        g1, g2 = set(self.vectorize(doc1)), set(self.vectorize(doc2))
        # Uses the Jaccard similarity coefficient
        # Should probably use something else, but there's like a dozen different ways to do it.
        # Researching what is best.
        return len(g1.intersection(g2)) / len(g1.union(g2))

    def cosine_similarity(self, g1, g2):
        
        # Counts the occurance of each ngram
        # Basically a sparse vector of every ngram and the times they occur
        doc1_wordfreq, doc2_wordfreq = Counter(g1), Counter(g2)

        # Get the shared ngrams, since they are the one showing that they are similar
        # two documents with no shared ngrams can't be similiar, in
        # cosine-similarity terms their vectors are orthogonal.
        shared_ngrams = set(doc1_wordfreq.keys()).intersection(set(doc2_wordfreq.keys()))

        # Dot product between the two vectors
        dot_product = sum([doc1_wordfreq[k] * doc2_wordfreq[k] for k in shared_ngrams])
        g1_length = np.sqrt(np.sum(np.power(np.asarray(list(doc1_wordfreq.values())), 2)))
        g2_length = np.sqrt(np.sum(np.power(np.asarray(list(doc2_wordfreq.values())), 2)))

        return dot_product / (g1_length * g2_length)

    def gram_matrix(self, docs):
        gram_matrix = np.zeros((len(docs), len(docs)))

        # Do the gramization once for each document, previously did it for every combination.
        vectorized = [list(self.vectorize(doc)) for doc in docs]

        for ((i, gram1), (j, gram2)) in it.combinations_with_replacement(enumerate(vectorized), 2):
            similarity = self.cosine_similarity(gram1, gram2)
            gram_matrix[i, j] = similarity
            gram_matrix[j, i] = similarity
        return gram_matrix


if __name__ == '__main__':
    
    print("NGK Start")
    ngk = NGK(5)
    print("EARN")
    train,test, docs = reut.load_docs("earn")
    print("CORN")
    train,test, docs = reut.load_docs("corn")

    gram_matrix = ngk.gram_matrix(docs['train'][:10])
    print(gram_matrix)

    train,test, docs = reut.load_docs()

    gram_matrix = ngk.gram_matrix(docs['train'][:500])

    print("NGK End")