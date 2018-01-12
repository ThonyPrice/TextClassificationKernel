# To run it, ex:
# import run_test as run
# run.play(3, ["earn", "corn", "acq", "crude"])
# To get the scores over all labels:
# import nltk
# from nltk.corpus import reuters
# run.play(3, reuters.categories())

import reut as r
import test as t
import nltk
from nltk.corpus import reuters
import plot as pt

def play(ngram, list):
	config = {}
	train, test, docs = r.load_docs(config)
	# Obtain WK processed documents	
	xs_wk = r.WK(docs)
	# Obtain NGK processed documents
	# xs_ngk = r.NGK(docs, ngram)
	ys = r.targets(train, test)
	# put xs and ys together for each case
	data_wk = r.data(xs_wk, ys)
	# data_ngk = r.data(xs_ngk, ys)
	# Get index from desired categories  
	listNum = []
	for i in list:
		for w in range(len(reuters.categories())):
			if i == reuters.categories()[w]:
				listNum += [w]
	# Testing
	for j in range(len(ngram)):
		xs_ngk = r.NGK(docs, ngram[j])
		data_ngk = r.data(xs_ngk, ys)
		name = "NGK length "+str(ngram[j])	
		t.test(data_ngk, list, listNum, name) 

	n, f, p, a = t.test(data_wk, list, listNum, "WK")	
	pt.plot(n, f, p, a)
