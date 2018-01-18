# SSK kernel
from __future__ import division
from collections import Counter
from math import sqrt
import itertools as it
import glob
import numpy as np
import reut
from nltk.corpus import reuters
import functools


class sSK():
    """
    Class for SSK kernel
    Can compare the similarity between two documents (sequences).
    Is initialized with the number of n-grams wanted.
    """

    def __init__(self, n, l):
        self.n = n
        self.l = l

    def kernel(self):
        def kernel_func(doc1, doc2):
            return self.normalizedSSK(doc1, doc2, self.n, self.l)
        return kernel_func

    @functools.lru_cache(maxsize=300)
    def SSK(self, s,t,n,l):
    # l is the Lambda value for the kernel
        lenS = len(s)
        lenT = len(t)
        # Initialize the K''Array
        K2 = np.zeros((lenS+1, lenT+1))

        # Initialize the K' array 
        K1 = np.zeros((lenS, lenT))

        # Initialize the array storing the different values of the kernel depending on n
        K = np.zeros(n+1)

        for i in range(1,lenS+1):
            for j in range(1,lenT+1):
                if s[i-1] == t[j-1]:
                    K2[i][j] = l*l

        # Start calculating the value of the kernel
        for p in range(2,n+1):
            for i in range(1,lenS):
                for j in range(1,lenT):
                    K1[i,j] = K2[i,j] + l * K1[i-1,j] + l * K1[i,j-1] - (l*l) * K1[i-1,j-1]

                    if s[i-1] == t[j-1]:
                        K2[i,j] = (l*l) * K1[i-1,j-1]
                        K[p] = K[p] + K2[i,j]
        return K[n]

    def normalizedSSK(self, s,t,n,l):
        ssk = self.SSK(s, t, n, l)

        # Normalize ssk
        sskF1 = self.SSK(s, s, n, l)
        sskF2 = self.SSK(t, t, n, l)
        den = pow(sskF1*sskF2, 0.5)
        normSSK = ssk/den
        
        print(normSSK)

        return normSSK
        
    def __str__(self):
        return "SSK"