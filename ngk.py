# N-Gram kernel
from collections import Counter
from math import sqrt


class NGK():
    """
    Class for N-gram kernel
    Can compare the similarity between two documents (sequences).
    Is initialized with the number of n-grams wanted.
    """

    def __init__(self, n):
        self.n = n

    def gramize(self, doc):
        """Returns the documented converted into a set containing its n-grams."""
        words = list()
        for i in range(self.n):
            words.append(doc[:i])
        for i in range(len(doc)):
            words.append(doc[i:i+self.n])
        return words

    def similarity(self, doc1, doc2):
        """Returns the similarity between two documents."""
        return self.cosine_similarity(doc1, doc2)

    def jaccard_similarity(self, doc1, doc2):
        g1, g2 = set(self.gramize(doc1)), set(self.gramize(doc2))
        # Uses the Jaccard similarity coefficient
        # Should probably use something else, but there's like a dozen different ways to do it.
        # Researching what is best.
        return len(g1.intersection(g2)) / len(g1.union(g2))

    def cosine_similarity(self, doc1, doc2):
        g1, g2 = self.gramize(doc1), self.gramize(doc2)
        # Counts the occurance of each ngram
        # Basically a sparse vector of every ngram and the times they occur
        doc1_wordfreq, doc2_wordfreq = Counter(g1), Counter(g2)

        # Get the shared ngrams, since they are the one showing that they are similar
        # two documents with no shared ngrams can't be similiar, in
        # cosine-similarity terms their vectors are orthogonal.
        shared_ngrams = set(doc1_wordfreq.keys()).intersection(set(doc2_wordfreq.keys()))

        # Dot product between the two vectors
        dot_product = sum([doc1_wordfreq[k] * doc2_wordfreq[k] for k in shared_ngrams])
        g1_length = sqrt(sum([doc1_wordfreq[k]**2 for k in doc1_wordfreq.keys()]))
        g2_length = sqrt(sum([doc2_wordfreq[k]**2 for k in doc2_wordfreq.keys()]))

        return dot_product / (g1_length * g2_length)


if __name__ == '__main__':
    ngk = NGK(5)
    print("Grams: {}".format(ngk.gramize("Test sentence")))

    print("Jaccard Similarity: {}".format(ngk.jaccard_similarity(
        "the quick brown fox jumps high", "the quick red fox jumps high",)))
    print("Jaccard Similarity: {}".format(ngk.jaccard_similarity(
        "the quick brown fox jumps high", "the lazy red fox jumps low",)))
    print("Jaccard Similarity: {}".format(ngk.jaccard_similarity(
        "the quick brown fox jumps high", "a bright red doctor isn't high",)))
    print("Jaccard Similarity: {}".format(ngk.jaccard_similarity(
        "the quick brown fox jumps high", "a bright red doctor is very smart",)))

    print()

    # Cosine similarity gives better results.
    print("Cosine Similarity: {}".format(ngk.cosine_similarity(
        "the quick brown fox jumps high", "the quick red fox jumps high",)))
    print("Cosine Similarity: {}".format(ngk.cosine_similarity(
        "the quick brown fox jumps high", "the lazy red fox jumps low",)))
    print("Cosine Similarity: {}".format(ngk.cosine_similarity(
        "the quick brown fox jumps high", "a bright red doctor isn't high",)))
    print("Cosine Similarity: {}".format(ngk.cosine_similarity(
        "the quick brown fox jumps high", "a bright red doctor is very smart",)))
