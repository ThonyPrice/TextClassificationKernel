from cvxopt.solvers import qp
from cvxopt.base import matrix
from cvxopt.solvers import options
from nltk.corpus import reuters
import numpy
import reut
from ngk import NGK
from pprint import pprint

def p_mat(data, kernel):
    P = [[data[i].label*data[j].label*kernel(data[i].doc, data[j].doc) for j in range(len(data))] for i in range(len(data))]
    return P


def svm(data, kernel):
    q = numpy.array([-1.0 for i in range(len(data))])
    P = p_mat(data, kernel)
    h = numpy.array([0.0 for i in range(len(data))])
    G = [[-1.0 if i == j else 0.0 for j in range(len(data))] for i in range(len(data))]

    # print(P)
    # print(q)
    result = qp(matrix(P), matrix(q), matrix(G), matrix(h))

    alphas = list(result["x"])
    nonzero_alpha_data = [(alphas[i], data[i]) for i in range(len(data)) if alphas[i] > 10**-5]
    # print("Alpha data pairs")
    # for e in nonzero_alpha_data:
    #     print(e)
    zero_alpha_data = [data[i] for i in range(len(data)) if alphas[i] <= 10**-5]

    return result, alphas, zero_alpha_data, nonzero_alpha_data

def svm_for_label(training_data, kernel, label):
    label_adjusted_data = list(map(lambda x: reut.Doc(x.doc, (1 if x.label == label else -1)), training_data))
    return svm(label_adjusted_data, kernel)

def predict(kernel, NAD, newpoint):
    # NAD[i][0] = alpha, NAD[i][0] = LABEL, NAD[i][1] = data
    calc = [NAD[i][0] * NAD[i][1].label* kernel(newpoint, NAD[i][1].doc) for i in range(len(NAD))]
    # print("predict vector", calc)
    s = sum(calc)
    return s

if __name__ == '__main__':
    docmap = reut.load_docs_with_labels(["earn","corn","acq","crude"])
    ngk = NGK(3)
    training_data = docmap['earn']['train'][:20] + docmap['corn']['train'][:10] + docmap['acq']['train'][:15]
    
    r,a,zad,nzad = svm_for_label(training_data, ngk.kernel(), "earn")
    # print(r)
    # print(a)
    # print(zad)
    pprint(nzad)
    vals = []
    for i in range(30):
        new_doc = docmap['earn']['test'][i]
        print("PREDICTION: ", new_doc.label)
        # print("Text: ",new_doc[1])
        vals.append(predict(ngk.kernel(), nzad, new_doc.doc))
    print(vals)
    correct = sum(map(lambda x : x > 0, vals))
    print("Accuracy", correct/len(vals))