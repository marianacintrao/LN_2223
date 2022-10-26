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
from nltk import ngrams


def classify(train_data, target, target_names, test_data):
    # Create dictionary and transform to feature vectors.
    count_vector = CountVectorizer()
    X_train_counts = count_vector.fit_transform(train_data)

    # TF-IDF vectorize.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Create model(naive bayes) and training. 
    clf = MultinomialNB().fit(X_train_tfidf, target)
    
    # Create test documents and vectorize.
    X_new_counts = count_vector.transform(test_data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # Execute prediction(classification).
    predicted = clf.predict(X_new_tfidf)

    # Show predicted data.
    predicted_data = []
    for text, category in zip(test_data, predicted):
        predicted_data.append((text, target_names[category]))
    return tuple(predicted_data)

if __name__ == "__main__":
    # load dict from file with data
    train_file = open("train.pkl", "rb")
    train = pickle.load(train_file)
    test = ['Great piece of crap!', 'pretty good, matches picture', "amazing"]
    predicted_data = classify(train["data"], train["target"], train["target_names"], test)
    for text, category in predicted_data:
        print("{0} => {1}".format(text, category))


