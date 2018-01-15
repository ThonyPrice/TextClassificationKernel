from nltk.corpus import reuters, stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

n_classes = 90
labels = reuters.categories()
signs = [',', '.', ';', ':', '?', '!', '-', '--', "'", '(', ')', '[', ']']
stop_words = stopwords.words("english") + signs

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
	
#WK documents processing
def WK(docs):
	vectorizer = CountVectorizer(stop_words=stop_words)	
	tfidf_transformer = TfidfTransformer()
	xs = {'train': [], 'test': []}
	xs['train'] = vectorizer.fit_transform(docs['train'])
	xs['test'] = vectorizer.transform(docs['test'])
	xs_tfidf = {'train': [], 'test': []}
	xs_tfidf['train'] = tfidf_transformer.fit_transform(xs['train'])
	xs_tfidf['test'] = tfidf_transformer.transform(xs['test'])
	return xs_tfidf

# NGK documents processing
# param ngram_size is the grams to be considered. (Sequence length)
def NGK(docs, ngram_size):
	vectorizer = CountVectorizer(stop_words=stop_words, ngram_range = (1, ngram_size))
	tfidf_transformer = TfidfTransformer()	
	xs = {'train': [], 'test': []}
	xs['train'] = vectorizer.fit_transform(docs['train'])
	xs['test'] = vectorizer.transform(docs['test'])
	xs_tfidf = {'train': [], 'test': []}
	xs_tfidf['train'] = tfidf_transformer.fit_transform(xs['train'])
	xs_tfidf['test'] = tfidf_transformer.transform(xs['test'])
	return xs_tfidf
	
def targets(train, test):
	mlb = MultiLabelBinarizer()
	ys = {'train': [], 'test': []}
	ys['train'] = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train])
	ys['test'] = mlb.transform([reuters.categories(doc_id) for doc_id in test])
	return ys
	
def data(xs,ys):
	data = {'x_train': xs['train'], 'y_train': ys['train'],'x_test': xs['test'], 'y_test': ys['test'],'labels': globals()["labels"]}
	return data
