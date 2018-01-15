# To run it, ex:
# import run_test as run
# run.play([3,4,5,6], ["earn", "corn", "acq", "crude"])
# The first list contains the lengths for NGK and the second list the labels to be shown
# plotting is for the whole dataset(the 90 categories)
# To get the output tables over all labels:
# import nltk
# from nltk.corpus import reuters
# run.play([3,4,5,6], reuters.categories())

import reut as r
import test as t
from nltk.corpus import reuters
import plot as pt

def play(ngram, list):
	config = {}
	train, test, docs = r.load_docs(config)

	# Obtain WK processed documents
	xs_wk = r.WK(docs)
	ys = r.targets(train, test)

	# put xs and ys together
	data_wk = r.data(xs_wk, ys)

	# Get index from desired categories
	listNum = []
	for i in list:
		for w in range(len(reuters.categories())):
			if i == reuters.categories()[w]:
				listNum += [w]

	# Performs testing over all specified lengths for NGK
	for j in range(len(ngram)):
		xs_ngk = r.NGK(docs, ngram[j])
		data_ngk = r.data(xs_ngk, ys)
		name = "NGK length "+str(ngram[j])
		t.test(data_ngk, list, listNum, name)

	# Performs testing for WK and gets the final lists of scores
	n, f, p, a = t.test(data_wk, list, listNum, "WK")

	# Plots scores for all classifiers and all lengths for NGK
	pt.plot(n, f, p, a)
