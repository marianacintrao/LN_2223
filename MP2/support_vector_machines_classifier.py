from re import T
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def classify(train_data, train_target, target_names, test_data, POLY=True, RBF=False):

    count_vector = CountVectorizer()
    X_train_counts = count_vector.fit_transform(train_data)

    # TF-IDF vectorize.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train_tfidf, train_target)
    poly = svm.SVC(kernel='poly', degree=5, C=1).fit(X_train_tfidf, train_target)

    # Create test documents and vectorize.
    X_new_counts = count_vector.transform(test_data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    if POLY:
        poly_pred = poly.predict(X_new_tfidf)
        poly_predicted_data = []
        for text, category in zip(test_data, poly_pred):
            poly_predicted_data.append((text, target_names[category]))
        return tuple(poly_predicted_data)
    elif RBF:
        rbf_pred = rbf.predict(X_new_tfidf)
        rbf_predicted_data = []
        for text, category in zip(test_data, rbf_pred):
            rbf_predicted_data.append((text, target_names[category]))

        return tuple(rbf_predicted_data)




if __name__ == "__main__":
    # load dict from file with data
    train_file = open("train.pkl", "rb")
    train = pickle.load(train_file)
    test = ['Great piece of crap!', 'pretty good, matches picture', "amazing"]

    train_data = train["data"]
    train_target = train["target"]
    target_names = train["target_names"]

    test = ['Great piece of crap!', 'pretty good, matches picture', "amazing"]

    prediction = classify(train_data, train_target, target_names, test)

    for text, category in prediction:
        print("{0} => {1}".format(text, category))

