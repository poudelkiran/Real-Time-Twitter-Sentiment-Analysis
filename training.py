#This File trains the classier and pickles to save the classifier so that it can be used while testing and running the real examples.

#import the libraries
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

#import classifiers
#Lets try with each classifiers and ensemble them to obtain the final result
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


#Lets create a Classifier Class cakked VoteClassifier 
# Inheriting from NLTK's ClassifierI.
#Assigning the list of classifiers that are passed to our class to self.classifiers

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers

#Creating our own classify method.
#votes contains the result from each classifier. We find the mode of the result.
    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

#Defining the confidence
#Confidence is the ratio of length of mode votes to total length of votes
    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# Defining and Accessing the corporas.
# In total, approx 10,000 feeds to be trained and tested on.
#We have separate files called positive.txt and negative.txt containing the tweets.
short_pos = open("Data/positive.txt","r").read()
short_neg = open("Data/negative.txt","r").read()

all_words = []
documents = []


#According to the Part of Speech Tagging called POS Tag,  j is adject, r is adverb, v is verb and so on. For more info check out POS Tagging.
#Lets filter out only the Adject from the tweets.
allowed_word_types = ["J"]

#For each split in new line, save them as a list with positive or negative 
for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
#Tokenize sentences to words
    words = word_tokenize(p)
#Find the POS Tagging of each word.
    pos = nltk.pos_tag(words)
#If the word is 'adject' i.e. allowed type put it in a list of all_words
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

#Same as above is done for the negatice tweets.    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


#Lets now pickle the documents and save it in a file called documents.pickle
save_documents = open("Output/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

#Lets find out frequency of each words in all_words list
all_words = nltk.FreqDist(all_words)


#Top 5000 words are selected as the word_features
word_features = list(all_words.keys())[:5000]


#Lets pickle word_features 
save_word_features = open("Output/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


#Finding the feature from the text
def find_features(document):
#Split documents in each word.
    words = word_tokenize(document)
    features = {}
#If the words in word_features are on document, make the feature True else make it Fals.e  
    for w in word_features:
        features[w] = (w in words)

    return features
#Feature contains list of all the word_features with True if they are in document or False if not.



featuresets = [(find_features(rev), category) for (rev, category) in documents]

#Shuffle the featuresets randomly
random.shuffle(featuresets)
print(len(featuresets))

#Partition training and testing set for the training and testing purpose. 
testing_set = featuresets[10000:]
training_set = featuresets[:10000]

# Pickling the featuresets.
save_features = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()


#Training and Pickling the classifier.

#Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
#Display top 15 most informative features of the classifier
classifier.show_most_informative_features(15)
#Pickle the classifier 
save_classifier = open("Output/originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#Multinomial Bayes Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open("Output/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

#Bernoulli Navive Bayes Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open("Output/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

#Logistic Regression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("Output/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


#Linear SVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("Output/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


#SGDC Classifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)
save_classifier = open("Output/SGDC_classifier.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

# Voting classifier.
# Creates a voting mechanism using the above classifiers
voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

#Print the accuracy of the ensemble classifier
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)