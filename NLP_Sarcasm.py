 #NLP

# Importing libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset =  pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 26709):  # 26709 is the numbers of headlines in the dataset
    review = re.sub('!', ' exclamation', dataset['headline'][i]) # replace '!' with the word 'exclamation' to use it as a dimension
    review = review.replace('?', ' inquiry') # re.sub didn't work with '?'
    matches = re.findall(r'\'(.+?)\'', review) # detects quoted text from string
    if matches: # if the string contains a quotation
        review += ' quotation' # add word 'quotation' to use it as a dimension
    review = re.sub('[^a-z]', ' ', review) # removes all symbols besides lowercase letters (everything is lowercase to begin with)
    review = review.split() # split the string into a list of single words for PorterStemmer
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # transform words to their stems and remove stopwords
    review = ' '.join(review) # join the list back to a string
    corpus.append(review) # add it to the corpus

# Creating the bag of words model
y = dataset.iloc[:, 2] # extracting the dependant outcomes
from sklearn.feature_extraction.text import CountVectorizer
features_n = range(100, 3000, 100) # list of different max vectors to try
scores = []
for i in features_n:
    cv = CountVectorizer(max_features = i) # setting the max number different features
    X = cv.fit_transform(corpus).toarray() # transforming list of strings into a matrix of token counts
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    error_rate = (FP+FN)/(TP+TN+FP+FN) # calculating the error rate based on confusion matrix results
    scores.append(error_rate)
    
#Printing out the optimal max features value and plot the results
optimal_n = features_n[scores.index(min(scores))]
print ("The optimal number of max vectors is %d" % optimal_n + " with an error rate of %.3f" % min(scores))
plt.plot(features_n, scores)
plt.xlabel('Number of Max Vectors')
plt.ylabel('Error Rate')
plt.show()