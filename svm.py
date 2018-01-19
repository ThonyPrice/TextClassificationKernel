import numpy
import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix
from cvxopt.solvers import options
from nltk.corpus import reuters
import reut
from ngk import NGK
from pprint import pprint
import random
import pickle
from WK import WK
import ssk
import os
cvxopt.solvers.options['show_progress'] = False
seed = 10000
random.seed(seed)
os.makedirs(os.path.dirname("kernelpmats/"), exist_ok=True)  # Makes path for kernel data.


def p_mat(data, kernel):
    P = [[0 for j in range(len(data))] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(i, len(data)):
            print("\rDoc pair ({},{})".format(i, j), end="")
            P[i][j] = data[i].label * data[j].label * \
                kernel(data[i].doc, data[j].doc)
            P[j][i] = P[i][j]

    print("\r", end="")

    # P = [[data[i].label*data[j].label*kernel(data[i].doc, data[j].doc)
    #       for j in range(len(data))] for i in range(len(data))]
    return P


def svm(data, kernel):
    q = numpy.array([-1.0 for i in range(len(data))])
    P = p_mat(data, kernel)
    h = numpy.array([0.0 for i in range(len(data))])
    G = [
        [-1.0 if i == j else 0.0 for j in range(len(data))] for i in range(len(data))]

    # print(P)
    # print(q)

    result = qp(matrix(P), matrix(q), matrix(G), matrix(h))

    alphas = list(result["x"])
    nonzero_alpha_data = [(alphas[i], data[i])
                          for i in range(len(data)) if alphas[i] > 10**-5]
    # print("Alpha data pairs")
    # for e in nonzero_alpha_data:
    #     print(e)
    zero_alpha_data = [data[i]
                       for i in range(len(data)) if alphas[i] <= 10**-5]

    return result, alphas, zero_alpha_data, nonzero_alpha_data


def svm_for_label(training_data, kernel, label):
    label_adjusted_data = list(map(lambda x: reut.Doc(
        x.doc, (1 if x.label == label else -1)), training_data))
    return svm(label_adjusted_data, kernel)


def predict(kernel, NAD, newpoint):
    # NAD[i][0] = alpha, NAD[i][0] = LABEL, NAD[i][1] = data
    calc = [NAD[i][0] * NAD[i][1].label *
            kernel(newpoint, NAD[i][1].doc) for i in range(len(NAD))]
    # print("predict vector", calc)
    s = sum(calc)
    return s


def calc_accuracy_precision_recall(test_data, kernel, nzad, label="earn"):
    """ Returns (Accuracy, Precision, Recall, F1-Score) """
    correct = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    for doc in test_data:
        # print("PREDICTION: ", doc.label)
        prediction = (predict(kernel.kernel(), nzad, doc.doc))
        # print("pre: ", prediction)
        if doc.label == label and prediction > 0:
            correct += 1
            true_pos += 1
        elif doc.label == label and prediction <= 0:
            false_neg += 1
        elif doc.label != label and prediction < 0:
            correct += 1
            true_neg += 1
        else:
            false_pos += 1
    # print("Correct", correct)
    # print("len(test_data)", len(test_data))
    # print("true_pos", true_pos)
    # print("false_pos", false_pos)
    # print("false_neg", false_neg)
    # print("true_neg", true_neg)
    acc = correct / (len(test_data))
    prec = true_pos / (true_pos + false_pos)
    rec = true_pos / (true_pos + false_neg)
    f1 = 2 * (prec * rec) / (prec + rec)
    return (acc, prec, rec, f1)


def get_training_data(docmap, earn=152, corn=38, acq=114, crude=76):
    # Using the splits in the paper, #380 docs
    # return docmap['earn']['train'][:152] + docmap['corn']['train'][:38] +
    # docmap['acq']['train'][:114] + docmap['crude']['train'][:76]
    return random.sample(docmap['earn']['train'], earn) + random.sample(docmap['corn']['train'], corn) + random.sample(docmap['acq']['train'], acq) + random.sample(docmap['crude']['train'], crude)


def get_test_data(docmap, earn=40, corn=10, acq=25, crude=15):
    # Using the splits in the paper # 90 docs
    # return docmap['earn']['test'][:40] + docmap['corn']['test'][:10] +
    # docmap['acq']['test'][:25] + docmap['crude']['test'][:15]
    return random.sample(docmap['earn']['test'], earn) + random.sample(docmap['corn']['test'], corn) + random.sample(docmap['acq']['test'], acq) + random.sample(docmap['crude']['test'], crude)


def do_kernel(docmap, kernelClass, label="earn"):
    global seed
    try:
        print("Loading file ", "kernelpmats/nonzero_alpha_data_{}_{}_seed{}.p".format(
            label, str(kernelClass), seed), end="\r")
        nzad = pickle.load(
            open("kernelpmats/nonzero_alpha_data_{}_{}_seed{}.p".format(label, str(kernelClass), seed), "rb"))
        get_training_data(docmap)  # To ensure seed is the same
    except IOError as e:
        r, a, zad, nzad = svm_for_label(
            get_training_data(docmap), kernelClass.kernel(), label)
        pickle.dump(nzad, open(
            "kernelpmats/nonzero_alpha_data_{}_{}_seed{}.p".format(label, str(kernelClass), seed), "wb"))

    return calc_accuracy_precision_recall(get_test_data(docmap), kernelClass, nzad, label)


def do_n_runs(docmap, kernel, n):
    """Always runs over the same randomized set of documents (starts with fixed seed and updates that)"""
    global seed
    seed = 10000
    random.seed(seed)
    for j in range(10):
        acc, prec, rec, f1 = do_kernel(docmap, ngk)
        print(" " * 100, end="\r")  # Erases load output
        percentages.append((acc, prec, rec))
        print(acc, end="\t")
        print(f1, end="\t")
        print(prec, end="\t")
        print(rec)
        seed = random.randint(1000, 15125111)
        random.seed(seed)


def do_ngrams(docmap):
    for i in range(3, 15):
        print("NGRAM KERNEL {}".format(i))
        p
        ngk = NGK(i)
        do_n_runs(docmap, ngk, 10)


def do_ssk(docmap):
    global seed
    for n in range(3, 14):
        sskk = ssk.sSK(n, 0.5)
        print("".format(str(sskk)))
        do_n_runs(docmap, sskk, 10)

    for i in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]:
        sskk = ssk.sSK(5, i)
        print("".format(str(sskk)))
        do_n_runs(docmap, sskk, 10)


if __name__ == '__main__':
    docmap = reut.load_docs_with_labels(["earn", "corn", "acq", "crude"])

    # ~*~ Select wich kernel to run here ~*~
    # kernel = NGK(5) # n-gram kernel
    kernel = WK([d.doc for d in get_training_data(docmap)])  # word kernel
    # kernel = ssk.sSK(5, 0.05)
    training_data = get_training_data(docmap)
    # test_data = get_test_data(docmap)
    # accuracy, precision, recall = do_kernel(docmap, kernel)
    print("WORD KERNEL")
    # acc, prec, rec = do_kernel(docmap, kernel)
    # print("Accuracy", acc)
    # print("Precision:", prec)
    # print("Recall: ", rec)
    # print("Accuracy", accuracy)
    # print("Precision:", precision)
    # print("Recall: ", recall)

    do_ngrams(docmap)
    # do_ssk(docmap)
