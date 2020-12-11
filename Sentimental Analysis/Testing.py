import re
import nltk
import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import wx.richtext

import wx
import wx.xrc
import tkinter as tk
from tkinter import messagebox as ms
from tkinter import StringVar
from tkinter import *
import sqlite3

from nltk.stem.porter import PorterStemmer

consumer_key = 'anOUeVGDoLGR85ynVNTifEwNS'
consumer_secret = 'hzhImyM8WQ2F7a1VOv1jfGaKk4Bde5xI4Z6GQlbOdTq23MZWue'
access_token = '1329654620570800128-5jhnDjtsbBcjUXBuHhyj0Dhg5izg5v'
access_token_secret = 'T28s2IDVP5lLFM1LTMs0c6muyZw7MF2w1d6sIz54xfivL'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


with sqlite3.connect('quit.db') as db:
    c = db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS user (username TEXT NOT NULL ,password TEX NOT NULL);')
db.commit()
db.close()

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def get_tweets(query, count=3000):
    tweets = []

    try:
        fetched_tweets = api.search(q=query, count=count)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet['text'] = tweet.text
            parsed_tweet['sentiment'] = get_tweet_sentiment(tweet.text)
            if tweet.retweet_count > 0:
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
            else:
                tweets.append(parsed_tweet)
        return tweets

    except tweepy.TweepError as e:
        print("Error : " + str(e))


def get_tweet_sentiment(tweet):
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment 
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    # tweet = re.sub(r'\s+', ' ', tweet)
    # words = tweet.split()

    # for word in words:
    #     word = preprocess_word(word)
    #     if is_valid_word(word):
    #         if use_stemmer:
    #             word = str(porter_stemmer.stem(word))
    #         processed_tweet.append(word)

    return tweet


analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

class Pc(tk.Frame):
        def __init__(self, parent):
            tk.Frame.__init__(self, parent)
            self.parent = parent
            # Some Usefull variables
            self.username = StringVar()
            self.password = StringVar()
            self.n_username = StringVar()
            self.n_password = StringVar()
            # Create Widgets
            self.widgets()


        # Login Function
        def login(self):
            # Establish Connection
            with sqlite3.connect('quit.db') as db:
                c = db.cursor()

            # Find user If there is any take proper action
            find_user = 'SELECT * FROM user WHERE username = ? and password = ?'
            c.execute(find_user, [(self.username.get()), (self.password.get())])
            result = c.fetchall()
            if result:
                self.logf.pack_forget()
                self.head['text'] = self.username.get() + '\n Loged In'
                self.head['pady'] = 150
                main()
            else:
                ms.showerror('Oops!', 'Username Not Found.')

        def new_user(self):
            # Establish Connection
            with sqlite3.connect('quit.db') as db:
                c = db.cursor()

            # Find Existing username if any take proper action
            find_user = ('SELECT * FROM user WHERE username = ?')
            c.execute(find_user, [(self.username.get())])
            if c.fetchall():
                ms.showerror('Error!', 'Username Taken Try a Diffrent One.')
            else:
                ms.showinfo('Success!', 'Account Created!')
                self.log()

            # Create New Account
            insert = 'INSERT INTO user(username,password) VALUES(?,?)'
            c.execute(insert, [(self.n_username.get()), (self.n_password.get())])
            db.commit()

            # Frame Packing Methords

        def log(self):
            self.username.set('')
            self.password.set('')
            self.crf.pack_forget()
            self.head['text'] = 'LOGIN'
            self.logf.pack()

        def cr(self):
            self.n_username.set('')
            self.n_password.set('')
            self.logf.pack_forget()
            self.head['text'] = 'Create Account'
            self.crf.pack()

        # Draw Widgets
        def widgets(self):
            self.head = Label(self.parent, text='LOGIN', font=('', 35), pady=10)
            self.head.pack()
            self.logf = Frame(self.parent, padx=10, pady=10)
            Label(self.logf, text='Username: ', font=('', 20), pady=5, padx=5).grid(sticky=W)
            Entry(self.logf, textvariable=self.username, bd=5, font=('', 15)).grid(row=0, column=1)
            Label(self.logf, text='Password: ', font=('', 20), pady=5, padx=5).grid(sticky=W)
            Entry(self.logf, textvariable=self.password, bd=5, font=('', 15), show='*').grid(row=1, column=1)
            Button(self.logf, text=' Login ', bd=3, font=('', 15), padx=5, pady=5, command=self.login).grid()
            Button(self.logf, text=' Create Account ', bd=3, font=('', 15), padx=5, pady=5, command=self.cr).grid(row=2,
                                                                                                                  column=1)
            self.logf.pack()

            self.crf = Frame(self.parent, padx=10, pady=10)
            Label(self.crf, text='Username: ', font=('', 20), pady=5, padx=5).grid(sticky=W)
            Entry(self.crf, textvariable=self.n_username, bd=5, font=('', 15)).grid(row=0, column=1)
            Label(self.crf, text='Password: ', font=('', 20), pady=5, padx=5).grid(sticky=W)
            Entry(self.crf, textvariable=self.n_password, bd=5, font=('', 15), show='*').grid(row=1, column=1)
            Button(self.crf, text='Create Account', bd=3, font=('', 15), padx=5, pady=5, command=self.new_user).grid()
            Button(self.crf, text='Go to Login', bd=3, font=('', 15), padx=5, pady=5, command=self.log).grid(row=2,
                                                                                                             column=1)

    # create window and application object

def main():

    class MyFrame2(wx.Frame):

        global path

        def __init__(self, parent):
            wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                              size=wx.Size(500, 300), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

            self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)

            bSizer1 = wx.BoxSizer(wx.VERTICAL)

            self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, u"Select CSV File", wx.DefaultPosition, wx.DefaultSize,
                                               0)
            self.m_staticText1.Wrap(-1)
            bSizer1.Add(self.m_staticText1, 0, wx.ALL, 5)

            self.m_filePicker1 = wx.FilePickerCtrl(self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.csv",
                                                   wx.DefaultPosition, wx.Size(500, -1), wx.FLP_DEFAULT_STYLE)
            bSizer1.Add(self.m_filePicker1, 0, wx.ALL, 5)

            self.m_button1 = wx.Button(self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0)
            bSizer1.Add(self.m_button1, 0, wx.ALL, 5)

            self.SetSizer(bSizer1)
            self.Layout()

            self.Centre(wx.BOTH)

            # Connect Events
            self.m_button1.Bind(wx.EVT_BUTTON, self.submit)

        def __del__(self):
            pass

        # Virtual event handlers, overide them in your derived class
        def submit(self, event):
            self.path = self.m_filePicker1.Path

            self.Close()

    app = wx.App(False)

    frame = MyFrame2(None)
    frame.Show(True)
    app.MainLoop()

    data = pd.read_csv(frame.path, encoding= 'unicode_escape')
    # Keeping only the necessary columns
    data = data[['text', 'sentiment']]

    data = data[data.sentiment != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    print(data[data['sentiment'] == 'Positive'].size)
    print(data[data['sentiment'] == 'Negative'].size)


    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')
    ##pos tagging
    for j in data["text"]:
        tokenized = sent_tokenize(j)
        for i in tokenized:
            print(i)
            wordsList = nltk.word_tokenize(i)
            wordsList = [w for w in wordsList if not w in stop_words]
            tagged = nltk.pos_tag(wordsList)
            print(tagged)



    # Stemming
    ps = PorterStemmer()
    for j in data["text"]:
        print(j)
        words = word_tokenize(j)
        for w in words:
            print(ps.stem(w))



    #ALGORITHM

    max_fatures = 20000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    x = tokenizer.texts_to_sequences(data['text'].values)
    x = pad_sequences(x)

    embed_dim = 128
    lstm_out = 256

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=x.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    batch_size = 128
    model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, verbose=2)

    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    #score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    #print("score: %.2f" % (score))
    #print("acc: %.2f" % (acc))

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):

        result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct / pos_cnt * 100, "%")
    print("neg_acc", neg_correct / neg_cnt * 100, "%")

    class MyFrame5(wx.Frame):

        def __init__(self, parent):
            wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                              size=wx.Size(603, 397), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

            self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)

            bSizer3 = wx.BoxSizer(wx.VERTICAL)

            self.m_staticText4 = wx.StaticText(self, wx.ID_ANY, u"Enter Text", wx.DefaultPosition, wx.DefaultSize, 0)
            self.m_staticText4.Wrap(-1)
            bSizer3.Add(self.m_staticText4, 0, wx.ALL, 5)

            self.m_textCtrl3 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(500, 100), 0)
            bSizer3.Add(self.m_textCtrl3, 0, wx.ALL, 5)

            self.m_button3 = wx.Button(self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0)
            bSizer3.Add(self.m_button3, 0, wx.ALL, 5)

            self.m_staticText5 = wx.StaticText(self, wx.ID_ANY, u"View Polarity", wx.DefaultPosition, wx.DefaultSize, 0)
            self.m_staticText5.Wrap(-1)
            bSizer3.Add(self.m_staticText5, 0, wx.ALL, 5)

            self.m_textCtrl4 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(300, -1), 0)
            bSizer3.Add(self.m_textCtrl4, 0, wx.ALL, 5)

            self.SetSizer(bSizer3)
            self.Layout()

            self.Centre(wx.BOTH)

            # Connect Events
            self.m_button3.Bind(wx.EVT_BUTTON, self.click)

        def __del__(self):
            pass

        # Virtual event handlers, overide them in your derived class
        def click(self, event):
            txt1 = self.m_textCtrl3.Value
            # twt = 'Meetings: Because none of us is as dumb as all of us.'
            # vectorizing the tweet by the pre-fitted tokenizer instance
            twt = tokenizer.texts_to_sequences(txt1)
            # padding the tweet to have exactly the same shape as `embedding_2` input
            twt = pad_sequences(twt, maxlen=31, dtype='int32', value=0)
            # print(twt)
            sentiment = model.predict(twt, batch_size=1, verbose=2)[0]
            if (np.argmax(sentiment) == 0):
                self.m_textCtrl4.SetValue(str("negative"))
            elif (np.argmax(sentiment) == 1):
                self.m_textCtrl4.SetValue(str("positive"))

            event.Skip()

    app5 = wx.App(False)

    frame = MyFrame5(None)
    frame.Show(True)
    app5.MainLoop()


    tweets = get_tweets(query='Donald Trump', count=200)
    dataset = pd.DataFrame(tweets)
    analyser = SentimentIntensityAnalyzer()
    # dataset_df = pd.read_csv('sentiment.csv')
    # dataset=pd.DataFrame(dataset_df)
    pos = 0
    neg = 0
    neu = 0
    a = dataset['sentiment'].count()

    for i in range(0, a):
        data = {}
        review = dataset['text'][i]
        review1 = preprocess_tweet(review)
        data = sentiment_analyzer_scores(review1)
        if (data['pos'] > data['neg']):
            pos = pos + 1
        elif (data['pos'] == data['neg']):
            neu = neu + 1
        else:
            neg = neg + 1

    print(pos)
    print(neg)
    print(neu)


if __name__ == '__main__':
    root = tk.Tk()
    run = Pc(root)
    root.mainloop()
    #main()

