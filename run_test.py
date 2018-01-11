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

def play(i1, i2, list):
	config = {}
	train, test, docs = r.load_docs(config)
	# Obtain WK processed documents	
	xs_wk = r.WK(docs)
	# Obtain NGK processed documents
	xs_ngk = r.NGK(docs, ngram)
	ys = r.targets(train, test)
	# put xs and ys together for each case
	data_wk = r.data(xs_wk, ys)
	data_ngk = r.data(xs_ngk, ys)
	# Get index from desired categories  
	listNum = []
	for i in list:
		for w in range(len(reuters.categories())):
			if i == reuters.categories()[w]:
				listNum += [w]
	# Testing
	t.test(data_wk, list, listNum, "WK")
	t.test(data_ngk, list, listNum, "NGK")
