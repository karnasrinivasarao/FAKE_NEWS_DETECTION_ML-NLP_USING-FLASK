import matplotlib
from textblob import TextBlob
import re
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
from time import strftime
import pickle

# Set up random seed, for reproductability of randomness
np.random.seed(18)
# 1. Basic data preparation and first classifier
# Import dataset
df1 = pd.read_csv('FNDdataset.csv')
# FAKE NEWS dataset, decision to select only columns "text" and "target1"
df1 = df1[['text', 'target1']]
# Getting rid of empty lines
df1 = df1[df1.text.isna() == False]
length_df1 = len(df1)
# Build sublist of original df1, contains # lines picked at random, out of 10863 possible
random_indexes = list(np.random.choice(length_df1 - 2, 3000, replace=False))
df1 = df1.iloc[random_indexes]
# Function dissects text i, attributes polarity scores, positive/negative/neutral, polarity or not, and subject
def sentiment_analyzer(dataframe):
    sid = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(i) for i in dataframe.text]
    compounds = np.array([i['compound'] for i in scores], dtype='float32')
    abs_compounds = np.array([np.sqrt(i ** 2) for i in compounds], dtype='float32')
    negs = np.array([i['neg'] for i in scores], dtype='float32')
    poss = np.array([i['pos'] for i in scores], dtype='float32')
    neus = np.array([i['neu'] for i in scores], dtype='float32')
    sent = dataframe['text'].apply(lambda x: TextBlob(x).sentiment)
    pol = np.array([s[0] for s in sent], dtype='float32')
    abs_pol = np.array([np.sqrt(i ** 2) for i in pol], dtype='float32')
    subj = np.array([s[1] for s in sent], dtype='float32')

    return compounds, abs_compounds, negs, poss, neus, sent, pol, abs_pol, subj


compounds, abs_compounds, negs, poss, neus, sent, pol, abs_pol, subj = sentiment_analyzer(df1)

# Adding columns to df1
df1['compounds'] = compounds
df1['abs_compounds'] = abs_compounds
df1['negs'] = negs
df1['neus'] = neus
df1['poss'] = poss
df1['pol'] = pol
df1['abs_pol'] = abs_pol
df1['subj'] = subj
# newly created variables
X = df1[['compounds', 'negs', 'neus', 'poss', 'pol', 'subj']]
y = df1['target1']

# First classifier
lrxtrain, lrxtest, lrytrain, lrytest = train_test_split(X, y)
lr = LogisticRegression()
lr.fit(lrxtrain, lrytrain)
lrpreds = lr.predict(lrxtest)
accuracy = accuracy_score(lrytest, lrpreds)
f1 = f1_score(lrytest, lrpreds)

# First attempt gives accuracy and f1 score
print(accuracy, f1)

x_values = df1[['text', 'compounds', 'abs_compounds', 'negs', 'neus', 'poss', 'pol', 'abs_pol', 'subj']]
y_values = df1['target1']
xtrain, xtest, ytrain, ytest = train_test_split(x_values, y_values)

# 2. Improving our classifier
# Cleans article from numbers, capital letters, punctuation and spaces for better classifier results
def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()
#nltk.download('vader_lexicon')
# Stop_words will ignore common english words which are noise (the / a / an / etc.)
# Max_df / min_df : ignore words which frequencies are above/under those thresholds
bow = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=998, max_df=1.0, min_df=1, binary=False)

training_data = bow.fit_transform(xtrain.text)
test_data = bow.transform(xtest.text)

dftrain = pd.DataFrame(training_data.toarray())
dftrain.columns = bow.get_feature_names()

dftest = pd.DataFrame(test_data.toarray())
dftest.columns = bow.get_feature_names()
# Second classifier
lr2 = LogisticRegression()
lr2.fit(dftrain, ytrain)
lr2_preds = lr2.predict(dftest)
accuracy = accuracy_score(ytest, lr2_preds)
f1 = f1_score(ytest, lr2_preds)
# second attempt gives accuracy and f1 score
print(accuracy, f1)


pickle.dump(lr2, open("model1.pkl", "wb"), protocol=2)
pickle.dump(clean_article, open("clean_article.pkl", 'wb'))
pickle.dump(bow, open("bow2.pkl", 'wb'), protocol=2)

print('\n')

end_time = strftime("%Y-%m-%d %H:%M:%S")
