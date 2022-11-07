import sys 
import re
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import jaccard_distance
from nltk.metrics.distance  import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Initialize the stemmer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
correct_words = words.words()

target_names = [
    "=Poor=", 
    "=Unsatisfactory=", 
    "=Good=", 
    "=VeryGood=", 
    "=Excellent="
    ]

def transform_data(data, REMOVE_STOPWORDS=False, PORTERSTEMMER=False, LANCASTERSTEMMER=False, LEMMATIZATION=True):
    new_data = []
    for text in data:
        # Tokenize the sentence
        words = word_tokenize(text)
        new_words = []
        for word in words:
            if (REMOVE_STOPWORDS and word.lower() not in stop_words) or not REMOVE_STOPWORDS:
                if PORTERSTEMMER:
                    word = porter.stem(word)
                elif LANCASTERSTEMMER:
                    word = lancaster.stem(word)
                elif LEMMATIZATION:
                    word = lemmatizer.lemmatize(word)
                new_words.append(word)
        new_data.append(' '.join(new_words))
    return new_data

def classify(train_data, target, target_names, test_data):
    count_vector = CountVectorizer()
    X_train_counts = count_vector.fit_transform(train_data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, target)
    
    X_new_counts = count_vector.transform(test_data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    predicted_data = []
    for text, category in zip(test_data, predicted):
        predicted_data.append((text, target_names[category]))
    return tuple(predicted_data)

if __name__ == '__main__':
    ### input >>> python reviews.py –test test.txt –train train.txt > results.txt
    test_filename = sys.argv[2]
    train_filename = sys.argv[4]

    train_data = []
    targets = []
    file = open(train_filename, 'r')
    Lines = file.readlines()
    for line in Lines:
        target_name, text = re.split('\t', line, maxsplit=1)
        train_data.append(text)
        target = target_names.index(target_name)
        targets.append(target)

    test_data = []

    file = open(test_filename, 'r')
    Lines = file.readlines()
    for line in Lines:
        test_data.append(line)

    train_data = transform_data(train_data)
    test_data = transform_data(test_data)

    predicted_data = classify(train_data, targets, target_names, test_data)

    for _, category in predicted_data:
        print(category)

