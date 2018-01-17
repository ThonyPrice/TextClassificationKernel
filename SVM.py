from cvxopt.solvers import qp
from cvxopt.base import matrix
from nltk.corpus import reuters, stopwords
from WK import WK

import numpy, pylab, random, math


def main():

    train1, test1, docs1 = load_docs('corn')
    train2, test2, docs2 = load_docs('crude')
    train1 = [(d, -1.) for d in docs1['train'][:20]]
    train2 = [(d, 1.) for d in docs2['train'][:20]]
    data = train1+train2
    random.shuffle(train1+train2)

    wk_kernel = WK([d[0] for d in data])

    pMatrix = pMatrixCreator(data, wk_kernel)
    q,h = qhVectorCreator(len(data))
    gMatrix = gMatrixCreator(len(data))
    r = qp(matrix(pMatrix), matrix(q), matrix(gMatrix), matrix(h))
    alpha = list(r['x'])
    param = nonZero(alpha, data)
    test_samples = docs1['test'][:10] # belong to class -1
    for sample in test_samples:
        print("Indicator: ", ind(param, sample, wk_kernel))
    test_samples = docs2['test'][:10] # belong to class 1
    print('***')
    for sample in test_samples:
        print("Indicator: ", ind(param, sample, wk_kernel))

def nonZero(alpha, data):
    nonZero = []
    for i in range(len(alpha)):
        if alpha[i] > 0.00001:
            d = data[i][0]
            t = data[i][1]
            nonZero.append((alpha[i],d,t))
    return nonZero

def ind(param, xs, kernel):
    sum = 0
    for (alpha, x,t) in param:
        sum += alpha*t*kernel.kernel(xs,x)
    return sum

def linearKernel(vectorX, vectorY):
    if(len(vectorX)!=len(vectorY)):
        print("Vector length not equal.")
        return 0
    scalar = 0
    for i in range(0,len(vectorX)):
        scalar += vectorX[i]*vectorY[i]
    return scalar+1.0

def pMatrixCreator(dataSet, kernel):
    n = len(dataSet)
    pMatrix = [[0.0 for x in range(n)] for y in range(n)]
    for i in range(n):
        x = dataSet[i]
        for j in range(n):
            y = dataSet[j]
            pMatrix[i][j] = x[1]*y[1]*kernel.kernel(x[0] ,y[0])
    return pMatrix

def qhVectorCreator(n):
    q = [-1.0]*n
    h = [0.0]*n
    return [q],[h]

def gMatrixCreator(n):
    gMatrix = [[0.0 for x in range(n)] for y in range(n)]
    for i in range(n):
        gMatrix[i][i] = -1.0
    return gMatrix

def generateData():
    classA = [(random.normalvariate(-1.5 ,1),
        random.normalvariate(0.5 ,1),
        1.0)
        for i in range(5)] + \
        [(random.normalvariate(1.5 ,1),
        random.normalvariate(0.5 ,1),
        1.0)
        for i in range(5)]
    classB = [(random.normalvariate(0.0 ,0.5),
        random.normalvariate(-0.5 ,0.5),
        -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    return data

def load_docs(label=None, config={}):
	if not label:
		documents = reuters.fileids()
	else:
		documents = reuters.fileids(label)
	test = [d for d in documents if d.startswith('test/')]
	train = [d for d in documents if d.startswith('training/')]
	docs = {}
	docs['train'] = [reuters.raw(doc_id) for doc_id in train]
	docs['test'] = [reuters.raw(doc_id) for doc_id in test]
	return train, test, docs





'''
pylab.hold(True)
pylab.plot( [p[0] for p in classA],
            [p[1] for p in classA],'bo')
pylab.plot( [p[0] for p in classB],
            [p[1] for p in classB], 'ro')
pylab.show()
'''

if __name__ == '__main__':
    main()
