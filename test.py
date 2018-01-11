from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import linear_model
import nltk
from nltk.corpus import reuters
import time
# Here we import SKK kernel file. 
# import SKKpyfile as SKK

def test(data, listLabels, listNum, k):
	data = data

	xs = {'train': data['x_train'], 'test': data['x_test']}
	ys = {'train': data['y_train'], 'test': data['y_test']}

	# Classifiers to use with the data	
	classifiers = [
			(k + ' LinearSVC', OneVsRestClassifier(LinearSVC(random_state=42))),
			,(k + ' SVM SVC linear', OneVsRestClassifier(SVC(kernel="linear", cache_size=200, random_state=42)))
			#,(k + 'SKK kernel', OneVsRestClassifier(SVC(kernel=SKK.kernel(data), cache_size=200, random_state=42, decision_function_shape='ovr')))
			#,(k + 'Linear with SGD', OneVsRestClassifier(linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)))
			]

	for clf_name, classifier in classifiers:
		t0 = time.time()
		classifier.fit(xs['train'], ys['train'])
		t1 = time.time()
		preds = classifier.predict(data['x_test'])
		t2 = time.time()
		y_true = ys['test']
		y_pred = preds
		print("-"*20,("{clf_name:<16}").format(clf_name=clf_name),"-"*20)
		# Classification report for each classifier	
		print(classification_report(y_true, y_pred, target_names=listLabels, labels=listNum))
