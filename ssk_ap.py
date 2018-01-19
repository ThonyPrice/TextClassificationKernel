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
import operator

class approxSSK():
    """
    Class for SSK kernel
    Can compare the similarity between two documents (sequences).
    Is initialized with the number of n-grams wanted.
    """

    def __init__(self, n, l, nrFeatures):
        self.n = n
        self.l = l
        self.nrFeatures = nrFeatures

    def kernel(self):
        def kernel_func(doc1, doc2):
            stringList = [doc1, doc2]
            topFeatures = self.occuranceOfSubstring(stringList, self.n, self.nrFeatures)
            result = self.normalizedApproxSSK(doc1, doc2, self.n, self.l, topFeatures)
            print(result)
            return result
        return kernel_func

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

    def normalizedApproxSSK(self, s,t,n,l, topFeatures):
        ssk = self.aproximateSSK(s, t, topFeatures, n, l)
        #print("SSK" + str(ssk))
    # Normalize ssk
        topFeatures = self.occuranceOfSubstring([s], self.n, self.nrFeatures)
        sskF1 = self.aproximateSSK(s, s, topFeatures, n, l)
        #print(str(sskF1))
        #print ("ssk1: " + str(sskF1))
        topFeatures = self.occuranceOfSubstring([t], self.n, self.nrFeatures)
        sskF2 = self.aproximateSSK(t, t, topFeatures, n, l)
        #print(str(sskF2))
        #print ("ssk2: " + str(sskF2))
        den = pow(sskF1*sskF2, 0.5)
        #den = sskF1*sskF2
        normSSK = ssk/den
        return normSSK

    def aproximateSSK(self, s, t, topFeatures, n, l):
        K = 0
    #   1. loop over the most occuring features which are given by the topFeatures list
    #   calculate the approximated kernel acording to formula (4) on page 437 in report
        for si in topFeatures:
            K1 = self.SSK(s,si,n,l)
            K2 = self.SSK(t,si,n,l)
            K += K1 * K2
        return K

    def occuranceOfSubstring(self, stringList, subStringLength, nrFeatures):
        n = subStringLength

        subStringDict = dict()
    #   1. Enumerate all sub-strings of lenght n, joint for all of the documents
        for string in stringList:
            for i in range(0, len(string) - n):
                key = string[i:i+n]
                if key in subStringDict:    
                    subStringDict[key] += 1
                else:
                    subStringDict[key] = 1 

    #   2. Extract the top features in accordance with the nrFeatures variable
    #   by converting the dictionary in to tuples and sort them in accordance with 
    #   observations from the texts.
        sorted_sequences = sorted(subStringDict.items(), key=operator.itemgetter(1))
        sorted_sequences.reverse()

        if nrFeatures < len(sorted_sequences):
            sorted_sequences = sorted_sequences[0:nrFeatures]

    #   3. Extracts the sub-string from the touples, which will be used to evaluate the approximate kernel
        topFeatures = list()
        for t in sorted_sequences:
            (key, val) = t
            topFeatures.append(key+"1")

        return topFeatures
        
    def __str__(self):
        return "Approx SSK"

if __name__ == '__main__':
    s = "difhdjkddhgkdjg kgkdsgkad kldalkjgig nbhbhu udfjdbgvu  oh gnooag"
    t = "fghdsghfg hgdhfka uhdvjiuebb jdhjg gdjg idjga aj dj ids ijds gnoa"
 
    ap = approxSSK(2, 0.5, 100)

    #ssk = ap.SSK(s, "hg",2,0.5)

    #print(str(ssk) + "SSK value")

    
    topFeatures = ap.occuranceOfSubstring([s,t], 2, 20)
    print (topFeatures)
    apssk = ap.normalizedApproxSSK(s,t,2,0.5, topFeatures)