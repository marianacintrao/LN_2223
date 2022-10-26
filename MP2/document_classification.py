import random
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import brown

nltk.download('brown')

tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]

file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])

train_set = brown.tagged_sents(categories='news')
test_set = brown.tagged_sents(categories='fiction')

'''
classifier = NaiveBayesClassifier.train(train_set) 
print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set)))
'''

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)