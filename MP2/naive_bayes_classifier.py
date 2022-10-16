'''''''''''''''''''''''
sources:
https://medium.com/@eiki1212/natural-language-processing-naive-bayes-classification-in-python-e934365cf40c

'''''''''''''''''''''''

# Import CountVectorizer class. 
# CountVectorizer converts text data to matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load dict from file with data
train_file = open("train.pkl", "rb")
train = pickle.load(train_file)

print(train.keys())

# Create dictionary and transform to feature vectors.
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(train["data"])

# TF-IDF vectorize.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Create model(naive bayes) and training. 
clf = MultinomialNB().fit(X_train_tfidf, train["target"])

# Create test documents and vectorize.
docs_new = ['Great piece of crap!', 'pretty good, matches picture', "amazing"]
X_new_counts = count_vector.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Execute prediction(classification).
predicted = clf.predict(X_new_tfidf)

# Show predicted data.
for doc, category in zip(docs_new, predicted):
    print("{0} => {1}".format(doc, train["target_names"][category]))
    