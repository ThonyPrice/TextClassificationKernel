from cvxopt.solvers import qp
from cvxopt.base import matrix
from cvxopt.solvers import options
from nltk.corpus import reuters
import numpy
import reut
from ngk import NGK
from pprint import pprint
import random
import pickle
from WK import WK
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

def calc_accuracy_precision_recall(test_data, kernel, nzad):
    correct = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    for doc in test_data:
        print("PREDICTION: ", doc.label)
        prediction = (predict(kernel.kernel(), nzad, doc.doc))
        print("pre: ", prediction)
        if doc.label == "earn" and prediction > 0:
            correct += 1
            true_pos += 1
        elif doc.label == "earn" and prediction <= 0:
            false_neg += 1
        elif doc.label != "earn" and prediction < 0:
            correct += 1
            true_neg += 1
        else:
            false_pos += 1
    print("Correct",correct)
    print("len(test_data)",len(test_data))
    print("true_pos",true_pos)
    print("false_pos",false_pos)
    print("false_neg",false_neg)
    return (correct/(len(test_data)), true_pos/(true_pos+false_pos), true_pos/(true_pos+false_neg))


def get_training_data(docmap):
    # Using the splits in the paper, #380 docs
    return docmap['earn']['train'][:152] + docmap['corn']['train'][:38] + docmap['acq']['train'][:114] + docmap['crude']['train'][:76]


def get_test_data(docmap):
    # Using the splits in the paper # 90 docs
    return  docmap['earn']['test'][:40] + docmap['corn']['test'][:10] + docmap['acq']['test'][:25] + docmap['crude']['test'][:15]


def do_kernel(docmap, kernelClass):
    try:
        nzad = pickle.load(open("nonzero_alpha_data"+str(kernelClass)+".p","rb"))
    except IOError as e:
        r,a,zad,nzad = svm_for_label(training_data, kernelClass.kernel(), "earn")
    pickle.dump(nzad, open("nonzero_alpha_data"+str(kernelClass)+".p", "wb"))

    return calc_accuracy_precision_recall(test_data, kernelClass, nzad)

if __name__ == '__main__':
    docmap = reut.load_docs_with_labels(["earn","corn","acq","crude"])

    # ~*~ Select wich kernel to run here ~*~
    kernel = NGK(5) # n-gram kernel
    # kernel = WK([d.doc for d in get_training_data(docmap)]) word kernel

    training_data = get_training_data(docmap)
    test_data = get_test_data(docmap)
    accuracy, precision, recall = do_kernel(docmap, kernel)
    # accuracy, precision, recall = do_kernel(docmap, wk)

    print("Accuracy", accuracy)
    print("Precision:", precision)
    print("Recall: ", recall)
