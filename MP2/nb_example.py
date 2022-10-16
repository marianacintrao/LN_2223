'''''''''''''''''''''''
sources:
https://medium.com/@eiki1212/natural-language-processing-naive-bayes-classification-in-python-e934365cf40c

'''''''''''''''''''''''

# Get training data.
from sklearn.datasets import fetch_20newsgroups
news_groups_train = fetch_20newsgroups(subset="train")

# Create dictionary and transform to feature vectors.
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(news_groups_train.data)

# TF-IDF vectorize.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Create model(naive bayes) and training. 
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, news_groups_train.target)

# Create test documents and vectorize.
docs_new = ['God is love', 'OpenGL on the GPU is fast', "United states goes to Iraq"]
X_new_counts = count_vector.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Execute prediction(classification).
predicted = clf.predict(X_new_tfidf)

# Show predicted data.
for doc, category in zip(docs_new, predicted):
    print("{0} => {1}".format(doc, news_groups_train.target_names[category]))
    