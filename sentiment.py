#This file uses the pickes values obtained after training the tweets and classify the tweets based based on voting the classifiers.
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


#Created a Class that inherits ClassifierI to vote the results obtained from different classiifiers
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
#Method to calculate the confidence
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("Output/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features_f = open("Output/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


#Method to find the features from the given text
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#Open all the pickes of the classifiers.
open_file = open("Output/originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("Output/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("Output/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Output/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Output/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Output/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()



voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

