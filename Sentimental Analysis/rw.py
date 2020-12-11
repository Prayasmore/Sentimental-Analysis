from textblob import TextBlob
from tweepy import OAuthHandler
import tweepy
from nltk.stem.porter import PorterStemmer
import wx.xrc
import wx
import matplotlib.pyplot as plt
import wx.richtext
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import sqlite3 as sqlite3
stop_words = set(stopwords.words('english'))


consumer_key = 'anOUeVGDoLGR85ynVNTifEwNS'
consumer_secret = 'hzhImyM8WQ2F7a1VOv1jfGaKk4Bde5xI4Z6GQlbOdTq23MZWue'
access_token = '1329654620570800128-5jhnDjtsbBcjUXBuHhyj0Dhg5izg5v'
access_token_secret = 'T28s2IDVP5lLFM1LTMs0c6muyZw7MF2w1d6sIz54xfivL'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


#global sqlite3
global p_cnt
global n_cnt
p_cnt = 0
n_cnt = 0


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
    ####NEW------------------#####
    twt = tokenizer.texts_to_sequences(tweet)
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    print(twt)
    sentiment = model.predict(twt, batch_size=1, verbose=2)[0]
    if(np.argmax(sentiment) == 0):
                n_cnt = n_cnt + 1
    elif (np.argmax(sentiment) == 1):
                p_cnt = p_cnt + 1

    # print("RESULT--->>>" + result)

    ###

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

# Login Function


def login(self):

    # Find user if there is any problem and take proper action
    find_user = 'SELECT * FROM user WHERE username = ? and password = ?'
    c.execute(find_user, [(self.username.get() + self.password.get())])
    result = c.fetchall()
    if result:
        self.logf.pack_forget()
        self.head['text'] = self.username.get() + '\n Logged In'
        self.head['pady'] = 150
        main()
    else:
        ms.showerror('Oops!','Username Not Found')

def new_user(self):
    # # Establish Connection
    # with sqlite3.connect('quit.db') as db:
    #     c = db.cursor()
        
    # Find Existing username if any take proper action
    find_user = 'SELECT * FROM user WHERE username = ?'
    c.execute(find_user, [(self.username.get())])
    c.fetchall()
    if result:
        ms.showerror('Error','Username Taken Try a Different One.')
    else:
        ms.showerror('Success!','Account Created!')
        self.log()
    

def main():
    #sqlite3 = sqlite3.connect('quit.db')
    # Establish Connection
    with sqlite3.connect('quit.db') as db :
        c = db.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS user (username TEXT NOT NULL, password TEXT NOT NULL);')
        db.commit()
        #db.close()

    
    
    class MyFrame1 ( wx.Frame ):
        
        global path
        def __init__( self, parent ):
            wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
            
            self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
            
            bSizer1 = wx.BoxSizer( wx.VERTICAL )
            
            self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"Select CSV File", wx.DefaultPosition, wx.DefaultSize, 0 )
            self.m_staticText1.Wrap( -1 )
            bSizer1.Add( self.m_staticText1, 0, wx.ALL, 5 )
            
            self.m_filePicker1 = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.csv", wx.DefaultPosition, wx.Size( 500,-1 ), wx.FLP_DEFAULT_STYLE )
            bSizer1.Add( self.m_filePicker1, 0, wx.ALL, 5 )
            
            self.m_button1 = wx.Button( self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
            bSizer1.Add( self.m_button1, 0, wx.ALL, 5 )
            
            
            
            self.SetSizer( bSizer1 )
            self.Layout()
            
            self.Centre( wx.BOTH )
            
            # Connect Events
            self.m_button1.Bind( wx.EVT_BUTTON, self.submit )
        
        def __del__( self ):
            pass
        
        
        # Virtual event handlers, overide them in your derived class
        def submit( self, event ):
            self.path = self.m_filePicker1.Path
            self.Close()          


    app = wx.App(False)
    
    
    frame = MyFrame1(None)
    frame.Show(True)
    app.MainLoop()
    


    data = pd.read_csv(frame.path,  encoding= 'unicode_escape')
    # Keeping only the necessary columns
    data = data[['text','sentiment']]

    data = data[data.sentiment != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    print(data[ data['sentiment'] == 'Positive'].size)
    print(data[ data['sentiment'] == 'Negative'].size)

    for idx,row in data.iterrows():
        row[0] = row[0].replace('rt',' ')
    # pos tagging
    for j in data["text"]:
        tokenized = sent_tokenize(j) 
        for i in tokenized: 
            print(i)
            wordsList = nltk.word_tokenize(i)
            wordsList = [w for w in wordsList if not w in stop_words]        
            tagged = nltk.pos_tag(wordsList)     
            print(tagged)


    ps = PorterStemmer()
    for j in data["text"]:
        print(j)
        words = word_tokenize(j)
        for w in words:
            print(ps.stem(w))



    
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)




    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())




    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)



    batch_size = 32
    model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    # score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    # print("score: %.2f" % (score))
    # print("acc: %.2f" % (acc))







    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):
        
        result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
    
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
        
        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1


    pos_acc = (pos_correct/pos_cnt*100)
    neg_acc = (neg_correct/neg_cnt*100)
    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")
    labels = ['Positive', 'Negative']
    # sizes = [5, neg_per, neu_per]
    sizes = [pos_acc, neg_acc]
    colors = ['yellowgreen', 'gold' ]
    patches, texts = plt.pie(sizes, colors=colors, shadow=True,  labels=labels, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    class MyFrame4 ( wx.Frame ):
    	
        def __init__( self, parent ):
            wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 603,397 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
            
            self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
            
            bSizer3 = wx.BoxSizer( wx.VERTICAL )
            
            #self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"Enter Text", wx.DefaultPosition, wx.DefaultSize, 0 )
            #self.m_staticText4.Wrap( -1 )
            #bSizer3.Add( self.m_staticText4, 0, wx.ALL, 4 )
            
            #self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 500,100 ), 0 )
            #bSizer3.Add( self.m_textCtrl3, 0, wx.ALL, 4 )

            self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Search tweet", wx.DefaultPosition, wx.DefaultSize, 0 )
            self.m_staticText6.Wrap( -1 )
            bSizer3.Add( self.m_staticText6, 0, wx.ALL, 4 )
            
            self.m_textCtrl5 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 500,100 ), 0 )
            bSizer3.Add( self.m_textCtrl5, 0, wx.ALL, 4 )
            
            self.m_button3 = wx.Button( self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
            bSizer3.Add( self.m_button3, 0, wx.ALL, 4 )
            
            #self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"View Polarity", wx.DefaultPosition, wx.DefaultSize, 0 )
            #self.m_staticText5.Wrap( -1 )
            #bSizer3.Add( self.m_staticText5, 0, wx.ALL, 4 )
            
            #self.m_textCtrl4 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 300,-1 ), 0 )
            #bSizer3.Add( self.m_textCtrl4, 0, wx.ALL, 4 )
            
            
            self.SetSizer( bSizer3 )
            self.Layout()
            
            self.Centre( wx.BOTH )
            
            # Connect Events
            self.m_button3.Bind( wx.EVT_BUTTON, self.click )
        
        def __del__( self ):
            pass
        
        
        # Virtual event handlers, overide them in your derived class
        def click( self, event ):
            txt1=self.m_textCtrl5.Value
            global txt_search
            txt_search = self.m_textCtrl5.Value
            # twt = 'Meetings: Because none of us is as dumb as all of us.'
            # vectorizing the tweet by the pre-fitted tokenizer instance
            twt = tokenizer.texts_to_sequences(txt1)
            # padding the tweet to have exactly the same shape as `embedding_2` input
            twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
            # print(twt)
            sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
            if(np.argmax(sentiment) == 0):
                self.m_textCtrl4.SetValue(str("negative"))
            elif (np.argmax(sentiment) == 1):
                self.m_textCtrl4.SetValue(str("positive"))
            
   
            
            event.Skip()
            
    def click(self, event):
	    txt1 = self.m_textCtrl4.Value

    app4 = wx.App(False)   
    frame = MyFrame4(None)
    frame.Show(True)
    app4.MainLoop()


    class MyFrame5 ( wx.Frame ):
    	
        def __init__( self, parent ):
            wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 603,397 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
            
            self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
            
            bSizer3 = wx.BoxSizer( wx.VERTICAL )
            
            self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"Enter Text", wx.DefaultPosition, wx.DefaultSize, 0 )
            self.m_staticText4.Wrap( -1 )
            bSizer3.Add( self.m_staticText4, 0, wx.ALL, 5 )
            
            self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 500,100 ), 0 )
            bSizer3.Add( self.m_textCtrl3, 0, wx.ALL, 5 )

            #self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Search tweet", wx.DefaultPosition, wx.DefaultSize, 0 )
            #self.m_staticText6.Wrap( -1 )
            #bSizer3.Add( self.m_staticText6, 0, wx.ALL, 5 )
            
            #self.m_textCtrl5 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 500,100 ), 0 )
            #bSizer3.Add( self.m_textCtrl5, 0, wx.ALL, 5 )
            
            self.m_button3 = wx.Button( self, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
            bSizer3.Add( self.m_button3, 0, wx.ALL, 5 )
            
            self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"View Polarity", wx.DefaultPosition, wx.DefaultSize, 0 )
            self.m_staticText5.Wrap( -1 )
            bSizer3.Add( self.m_staticText5, 0, wx.ALL, 5 )
            
            self.m_textCtrl4 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 300,-1 ), 0 )
            bSizer3.Add( self.m_textCtrl4, 0, wx.ALL, 5 )
            
            
            self.SetSizer( bSizer3 )
            self.Layout()
            
            self.Centre( wx.BOTH )
            
            # Connect Events
            self.m_button3.Bind( wx.EVT_BUTTON, self.click )
        
        def __del__( self ):
            pass
        
        
        # Virtual event handlers, overide them in your derived class
        def click( self, event ):
            txt1=self.m_textCtrl3.Value
            global txt_search
            txt_search=self.m_textCtrl3.Value
            # twt = 'Meetings: Because none of us is as dumb as all of us.'
            # vectorizing the tweet by the pre-fitted tokenizer instance
            twt = tokenizer.texts_to_sequences(txt1)
            # padding the tweet to have exactly the same shape as `embedding_2` input
            twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
            # print(twt)
            sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
            if(np.argmax(sentiment) == 0):
                self.m_textCtrl4.SetValue(str("negative"))
            elif (np.argmax(sentiment) == 1):
                self.m_textCtrl4.SetValue(str("positive"))
   


            event.Skip()
            
    
    app5 = wx.App(False)   

    frame = MyFrame5(None)
    frame.Show(True)
    app5.MainLoop()

    tweets = []
    n_cnt = 0
    p_cnt = 0
    try:
        fetched_tweets = api.search(q = txt_search, count = 200)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet['text'] = tweet.text
            twt = tokenizer.texts_to_sequences(tweet.text) 
            twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
            # print(twt)
            sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
            if(np.argmax(sentiment) <= 0.3):
                n_cnt += 1
            elif (np.argmax(sentiment) == 1):
                p_cnt += 1

    except tweepy.TweepError as e: 
        print("Error : " + str(e))
        
    print("Final_Result_Positive--->" + str(p_cnt))
    print("Final_Result_negative--->" + str(n_cnt))
    labels = ['Positive', 'Negative']
    # sizes = [5, neg_per, neu_per]
    sizes = [p_cnt, n_cnt]
    colors = ['yellowgreen', 'gold' ]
    patches, texts = plt.pie(sizes, colors=colors, shadow=True,  labels=labels, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    main()

