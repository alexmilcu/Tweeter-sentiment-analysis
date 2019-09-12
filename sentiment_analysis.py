import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import wordninja
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import re
import numpy as np
import matplotlib.pyplot as plt

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


def clean_tweet(tweet):
    # tweet tokenizer pt a detecta emoji-uri, tag-uri, eliminare username, reducere nr litere
    tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tweets = tw_tokenizer.tokenize(tweet)
    tweets = " ".join(tweets)
    tweets = tweets.lower()
    # inlocuire abrevieri
    # tweets = tweets.replace(' u ', 'you')
    # tweets = tweets.replace(' ur', 'your')
    # tweets = tweets.replace('gr8', 'great')
    # word ninja pt tag-uri cu 2 sau mai multe cuvinte legate
    tweets = wordninja.split(tweets)
    tweets = " ".join(tweets)
    # eliminare
    tweets = re.sub('[^a-zA-Z]', ' ', tweets)
    tweets = tweets.split()
    # eliminare stopwords
    tweets = [tweet for tweet in tweets if tweet not in stop_words]
    tweets = " ".join(tweets)
    return tweets


def hashtag(tweet):
    tweets = " ".join(filter(lambda x: x[0] == '#', tweet.split()))
    tweets = re.sub('[^a-zA-Z]', ' ', tweets)
    tweets = tweets.lower()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = "".join(tweets)
    return tweets


df = pd.read_csv('train.csv')

cleaned_tweet = pd.DataFrame(columns=['tweet'])
cleaned_tweet['tweet'] = df['tweet'].apply(clean_tweet)
cleaned_tweet.to_csv('cleaned_tweet.csv')

hashtags = pd.DataFrame(columns=['hashtag'])
hashtags['hashtag'] = df['tweet'].apply(hashtag)
hashtags['label'] = df['label']
hashtags.to_csv('hashtags.csv')

# numar tweet uri rasiste/non rasiste
print(len(df[df.label == 0]), 'Not racist/sexist Tweets')
print(len(df[df.label == 1]), 'Racist/sexist Tweets')

# cele mai frecvente hashtag uri
all_hashtags = FreqDist(list((' '.join(hashtags.loc[:, 'hashtag'])).split())).most_common(10)
hatred_hashtags = FreqDist(list(' '.join(hashtags[hashtags.label == 1].hashtag.values).split())).most_common(10)

print(all_hashtags)

# grafice tweet uri
plt.figure(figsize=(14, 6))
ax = plt.subplot(121)
pd.DataFrame(all_hashtags, columns=['hashtag', 'Count']).set_index('hashtag').plot.barh(ax=ax, fontsize=12)
plt.xlabel('# occurrences')
plt.title('Hashtags in all tweets', size=13)
ax = plt.subplot(122)
pd.DataFrame(hatred_hashtags, columns=['hashtag', 'Count']).set_index('hashtag').plot.barh(ax=ax, fontsize=12)
plt.xlabel('# occurrences')
plt.ylabel('')
plt.title('Hashtags in hatred tweets', size=13)
plt.show()

text = []

for i in range(df.shape[0]):
    text.append(cleaned_tweet['tweet'][i])

text = np.asarray(text)

features = cleaned_tweet['tweet'].values
labels = df['label'].values

shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.33)

for train_index, test_index in shuffle_stratified.split(features, labels):
    f_train, f_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

cv = CountVectorizer(analyzer='word', stop_words='english')
cv.fit(f_train)

X_train = cv.transform(f_train)
X_test = cv.transform(f_test)

classifier1 = LogisticRegression(C=10)
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)

print(f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

for train_index, test_index in shuffle_stratified.split(text, labels):
    f1_train, f1_test = text[train_index], text[test_index]
    y1_train, y1_test = labels[train_index], labels[test_index]

tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=10, stop_words='english')
tfidf.fit(f1_train)

X1_train = tfidf.transform(f1_train)
X1_test = tfidf.transform(f1_test)

classifier2 = LogisticRegression(C=10)
classifier2.fit(X1_train, y1_train)

y1_pred = classifier2.predict(X1_test)

print(f1_score(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))
print(confusion_matrix(y1_test, y1_pred))

test = pd.read_csv('test.csv')
test['cleaned_tweet'] = test['tweet'].apply(clean_tweet)

test_text = []
for i in range(test.shape[0]):
    test_text.append(test['cleaned_tweet'][i])

Test_X = tfidf.transform(test_text)

pred_Y = classifier2.predict(Test_X)
prob_Y = classifier2.predict_proba(Test_X)

test['score_linear_regression'] = prob_Y[:, 1]
test['prediction_linear_regression'] = pred_Y

classifier3 = MultinomialNB(alpha=0.01)
classifier3.fit(X_train, y_train)

y2_pred = classifier3.predict(X_test)

print(f1_score(y_test, y2_pred))
print(classification_report(y_test, y2_pred))
print(confusion_matrix(y_test, y2_pred))

Test1_X = cv.transform(test['cleaned_tweet'].values)

pred2_Y = classifier3.predict(Test1_X)
prob2_Y = classifier3.predict_proba(Test1_X)

test['score_naive_bayes'] = prob2_Y[:, 1]
test['prediction_naive_bayes'] = pred2_Y

parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 1, 10, 100]}
classifier4 = SVC(kernel='rbf')
svc = GridSearchCV(classifier4, param_grid=parameters, cv=5, n_jobs=-1)
svc.fit(X1_train, y1_train)

print("Best parameter found is {} with F1 score of {:.2f}".format(
    svc.best_params_,
    svc.best_score_
))

classifier4 = SVC(kernel='rbf', C=svc.best_params_['C'], gamma=svc.best_params_['gamma'])
classifier4.fit(X1_train, y1_train)

y3_pred = classifier4.predict(X1_test)

print(f1_score(y1_test, y3_pred))
print(classification_report(y1_test, y3_pred))
print(confusion_matrix(y1_test, y3_pred))

pred3_Y = classifier4.predict(Test_X)

test['prediction_SVM'] = pred3_Y

test.to_csv('predictions.csv')
