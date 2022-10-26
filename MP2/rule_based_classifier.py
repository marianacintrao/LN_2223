'''''''''''''''''''''
sources:
https://monkeylearn.com/text-classification/
https://medium.com/tokopedia-data/step-by-step-text-classification-fa439608e79e

- stopwords
https://levelup.gitconnected.com/how-to-remove-stopwords-from-text-in-python-9e9fbfcbca8d
- stemming
https://www.datacamp.com/tutorial/stemming-lemmatization-python
- autocorrect
https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/

'''''''''''''''''''''
from curses.ascii import isalpha
import pickle
import nltk

#nltk.download('vader_lexicon')

from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import jaccard_distance
from nltk.metrics.distance  import edit_distance
from nltk.util import ngrams


def sentiment(sentence):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(sentence)
    if polarity['neg'] > 0.9:
        return "==Poor=="
    elif polarity['pos'] > 0.9:
        return "==Excellent=="
    elif polarity['neu'] > 0.8:
        return "==Good=="
    elif polarity['pos'] > polarity['neg']:
        return "==VeryGood=="
    else:
        return "==Unsatisfactory=="


# stopwords from NLTK
stop_words = stopwords.words('english')# my new custom stopwords

correct_words = words.words()

def classify(data, STOP_WORD_REMOVAL=False, AUTOCORRECT=False):
    classification_list = []
    for text in data:
        words = word_tokenize(text)
        if STOP_WORD_REMOVAL:
            new_words = []
            for word in words:
                if word.lower() not in stop_words:
                    new_words.append(word)
            words = new_words
        if AUTOCORRECT:
            new_words = []
            for word in words:
                if word.isalpha():
                    temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
                    new_words.append(sorted(temp, key = lambda val:val[0])[0][1])
                else:
                    new_words.append(word)
            words = new_words
        text = ' '.join(w for w in words)
        classification_list.append((text, sentiment(text)))
    return classification_list

if __name__ == "__main__":
    # load dict from file with data
    train_file = open("train.pkl", "rb")
    train = pickle.load(train_file)

    # ========================== #

    for text in train["data"][:6]:
        filtered_list = []
        # Tokenize the sentence
        print(sentiment(text), text)
        words = word_tokenize(text)
                
    # print(filtered_list)

