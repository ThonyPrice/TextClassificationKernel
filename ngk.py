# N-Gram kernel


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
        words = set()
        for i in range(self.n):
            words.add(doc[:i])
        for i in range(len(doc)):
            words.add(doc[i:i+self.n])
        return words

    def similarity(self, doc1, doc2):
        """Returns the similarity between two documents."""
        g1, g2 = self.gramize(doc1), self.gramize(doc2)
        # Uses the Jaccard similarity coefficient
        # Should probably use something else, but there's like a dozen different ways to do it.
        # Researching what is best.
        return len(g1.intersection(g2)) / len(g1.union(g2))


if __name__ == '__main__':
    ngk = NGK(3)
    print("Grams: {}".format(ngk.gramize("Test sentence")))
    print("Similarity: {}".format(ngk.similarity(
        "the quick brown fox jumps high", "the quick red fox jumps high",)))
    print("Similarity: {}".format(ngk.similarity(
        "the quick brown fox jumps high", "the lazy red fox jumps low",)))
    print("Similarity: {}".format(ngk.similarity(
        "the quick brown fox jumps high", "a bright red doctor isn't high",)))
    print("Similarity: {}".format(ngk.similarity(
        "the quick brown fox jumps high", "a bright red doctor is very smart",)))
