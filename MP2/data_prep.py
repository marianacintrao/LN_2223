
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
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import jaccard_distance
from nltk.metrics.distance  import edit_distance
from nltk.util import ngrams
import pickle

#print(stopwords.words('english'))
'''
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 
'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
"it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
"haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
'''

# stopwords from NLTK
stop_words = stopwords.words('english')# my new custom stopwords
extra_stop_words = [
    "'m", "'s", "n't", "'ve",
    "&", "=",
    #"!", "?", ",", ".", ";", "...", "....", "-", ":",
    ] # add the new custom stopwrds to my stopwords
stop_words.extend(extra_stop_words)



# Initialize the stemmer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

PORTERSTEMMER = False
LANCASTERSTEMMER = False
LEMMATIZATION = True
AUTOCORRECT = False

correct_words = words.words()

def transform_data(data, REMOVE_STOPWORDS=False, PORTERSTEMMER=False, LANCASTERSTEMMER=False, LEMMATIZATION=False):
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
        


if __name__ == "__main__":
    # load dict from file with data
    # load dict from file with data
    train_file = open("train.pkl", "rb")
    train = pickle.load(train_file)
    
    data = train["data"][:6]
    print(data)
    new_data = transform_data(data)
    print(new_data)