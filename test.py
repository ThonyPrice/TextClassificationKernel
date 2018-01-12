from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, precision_score, fbeta_score, classification_report, accuracy_score
from sklearn import linear_model
import nltk
from nltk.corpus import reuters
import time

# Here we import SKK kernel file. 
# import SKKpyfile as SKK

p, a, f, clf = [], [], [], []

def test(data, listLabels, listNum, k):
	data = data

	xs = {'train': data['x_train'], 'test': data['x_test']}
	ys = {'train': data['y_train'], 'test': data['y_test']}

	# Classifiers to use with the data	
	classifiers = [
			(k + ' LinearSVC', OneVsRestClassifier(LinearSVC(loss='hinge',random_state=42)))
			#,(k + ' SVM SVC linear', OneVsRestClassifier(SVC(kernel="linear", cache_size=200, random_state=42)))
			#,(k + 'SKK kernel', OneVsRestClassifier(SVC(kernel=SKK.kernel(data), cache_size=200, random_state=42, decision_function_shape='ovr')))
			#,(k + 'Linear with SGD', OneVsRestClassifier(linear_model.SGDClassifier(penalty='l2',alpha=0.001, random_state=42, max_iter=100)))
			]

	for index in range(len(classifiers)):
		clf_name, classifier = classifiers[index]	
		t0 = time.time()
		classifier.fit(xs['train'], ys['train'])
		t1 = time.time()
		preds = classifier.predict(data['x_test'])
		preds[preds >= 0.5] = 1
		preds[preds < 0.5] = 0
		t2 = time.time()
		y_true = ys['test']
		y_pred = preds
		prec = precision_score(y_true, y_pred, average='weighted')
		f1 = fbeta_score(y_true, y_pred, beta=1, average='weighted')
		acc = accuracy_score(y_true, y_pred)
		p.append(prec), f.append(f1), a.append(acc), clf.append(clf_name) 
		print("classifier: ",clf_name,"\tf1: ",f1,"\tprecision: ",prec,"\taccuracy: ",acc)
		print("-"*20,("{clf_name:<16}").format(clf_name=clf_name),"-"*20)
		# Classification report for each classifier	
		print(classification_report(y_true, y_pred, target_names=listLabels, labels=listNum))

	return clf, f, p, a
