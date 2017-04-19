# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys,os
import numpy as np

import keras
import pandas as pd
import re
import copy #can delete

import string
import matplotlib.pyplot as plt


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras import regularizers

from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.cross_validation import cross_val_score,cross_val_predict,KFold,StratifiedKFold

np.random.seed(1)

MAX_SEQUENCE_LENGTH = 30 #max number of sentences in a message
MAX_NB_WORDS = 20000 #cap vocabulary
GLOVE_FILE = '/Users/nookiebiz4/Downloads/glove/glove.twitter.27B.100d.txt'
EMBEDDING_DIM = 100 #size of word vector 
TWITTER_FILE = '/Users/nookiebiz4/583_proj2/training-Obama-Romney-tweets.xlsx'
JAR_FILE = '/Users/nookiebiz4/Downloads/stanford-postagger-2016-10-31/stanford-postagger.jar'
MODEL_FILE = '/Users/nookiebiz4/Downloads/stanford-postagger-2016-10-31/models/english-left3words-distsim.tagger'
TOKENIZER = 'keras' #or use nltk
STEMMER = 'wordnet' #or use snowball or porter


def get_Ytrue_Ypred(model,x,y):
    #Y matrix is [1,0,0] for class 0, [0,1,0] for class 1, [0,0,1] for class -1
    convert_to_label ={0:0,1:1,2:-1}
    model_predictions = model.predict(x)
    y_pred = np.zeros(len(y))
    y_true = np.zeros(len(y))
    #errors = 0.0
    for i in range(len(y)):
        y_pred[i] = convert_to_label[np.argmax(model_predictions[i])]
        y_true[i] = convert_to_label[np.argmax(y[i])]
        #if y_true[i] != y_pred[i]:
            #errors+=1.0
    return y_true,y_pred
    

# read the data
obama_data = pd.read_excel(TWITTER_FILE,names = ['date','time','text','sentiment'],parse_cols = 4,sheetname = 'Obama')
romney_data = pd.read_excel(TWITTER_FILE,names = ['date','time','text','sentiment'],parse_cols = 4,sheetname = 'Romney')

def get_data(data):
    """ get and clean the data """
    data = data.iloc[1:]
    data['text'] = data['text'].values.astype('unicode')
    data['date'] = data['date'].values.astype('str')
    data['time'] = data['time'].values.astype('unicode')
    # remove rows with mixed sentiment
    data = data[data['sentiment'] < 2]
    data.index = range(len(data))
    
    return data

obama_data = get_data(obama_data)
romney_data = get_data(romney_data)

obama_dataO = copy.deepcopy(obama_data)

print obama_data.head()
print romney_data.head()


emoticon_dictionary = {':)':' smileyface ','(:':' smileyface ','XD': ' happyface ',':D': ' smileyface ','>.<':' smileyface ',':-)':' smileyface ',';)':' winkface ',';D':' winkface ',':\'(':' cryingface '}

emoticons = [':\)','\(:','XD',':D','>\.<',':-\)',';\)',';D',':\'\(']

emoticon_pattern = re.compile(r'(' + '\s+|\s+'.join(emoticons) + r')')

# convert emoticons to words
def emoticon_converter(x):
    x = emoticon_pattern.sub(lambda i : emoticon_dictionary[i.group().replace(' ','')],x)   
    return x

obama_data['text'] = obama_data['text'].apply(emoticon_converter)
romney_data['text'] = romney_data['text'].apply(emoticon_converter)

# http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
# convert hashtags into words
def separate_hashtag(x):
    for i in range(0,len(x)):
        hashtags = re.findall(r"#(\w+)", x[i])
        for words in hashtags:
            x[i] = re.sub('#'+ words,split_hashtag(words.lower()),x[i])
    return x

#obama_data['text'] = separate_hashtag(obama_data['text'])
#romney_data['text'] = separate_hashtag(romney_data['text'])






# remove punctuations
punc = ['\:','\;','\?','\$','\.','\(','\)','\#',',','-']
cond_1 = re.compile('|'.join(punc))
# remove tags
tags = ['<a>','</a>','<e>','</e>']
cond_2 = re.compile("|".join(tags))

def preprocess(data):
    """ preprocess the data"""
    # remove punctuations
    data = data.apply(lambda x : re.sub(cond_1,'',x))
    # remove tags
    data = data.apply(lambda x : re.sub(cond_2,'',x))
    # remove users
    data = data.apply(lambda x : re.sub(r'\@\s?\w+','',x))
    # remove hypertext 
    data = data.apply(lambda x : re.sub(r"http\S+",'',x)) #edited 
    # remove digits
    data = data.apply(lambda x : re.sub(r'[0-9]+','',x))
    # convert to ascii
    data = data.apply(lambda x: x.encode('utf-8')) #sometimes doesn't work
    # in that case
    printable = set(string.printable)
    for i in range(len(data)):
        data[i] = filter(lambda x: x in printable, data[i])
    
    return data

obama_data['text'] = preprocess(obama_data['text'])
romney_data['text'] = preprocess(romney_data['text'])


def process_time(data):
    """ processes time """

    def extract_date(pattern,string):
        temp = re.match(pattern,string)
        if temp:
            return temp.group(1)
        else:
            return string
    # clean date
    date_format_1 = re.compile('\d+/(\d{2})/\d+')
    date_format_2 = re.compile('\d+\-\d+\-(\d{2})')
    date_format_3 = re.compile('(\d{2})\-[a-zA-Z]+\-\d+')
    date_format = [date_format_1] + [date_format_2] + [date_format_3]

    # remove whitespace
    data['date'] = data['date'].apply(lambda x : x.replace(' ',''))

    for i in date_format:
        data['date'] = data['date'].apply(lambda x: extract_date(i,x))

    def converter(first,second):
        if first == 'AM':
            return second
        else:
            val = re.findall('(\d{1,2})',second)[0]
            if int(val) > 12:
                val = str(int(val) + 12)
            return re.sub('\d{1,2}',val,second,1)

    def extract_time(pattern,string):

        temp = re.match(pattern,string)
        if temp:
            first = temp.group(1)
            second = temp.group(2)
            third = temp.group(3)

            if first is None and third is None:
                return second

            if first == 'AM' or first == 'PM':
                return converter(first,second)
            else:
                return converter(third,second)

    # clean time
    time_format_1 = re.compile('(AM|PM)?\s?(\d{1,2}:\d{1,2}:\d{1,2})\s?(AM|PM)?')

    # remove whitespace
    data['time'] = data['time'].apply(lambda x : x.replace(' ',''))

    data['time'] = data['time'].apply(lambda x : extract_time(time_format_1,x))
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    
    return data
    
## IMP - Process Emoticons, better stopwords list, clean hashtags
# Tweet NLP

manual_stopwords_list = ['rt']

# stopwords list based on pos tags

from nltk.tag import StanfordPOSTagger

jar = JAR_FILE
model = MODEL_FILE

st = StanfordPOSTagger(model, jar, encoding='utf8')

remove_tags_stanfordpos = ['IN','DT','PRP','PRP$','WDT','WP','WP$','CD','PDT']
remove_tags_tweetnlp = []


def tweet_tag_filter(x):
    pass


# obama_data['text'] = obama_data['text'].apply(tweet_tag_filter)
# romney_data['text'] = romney_data['text'].apply(tweet_tag_filter)


def pos_tag_filter(x):
    x = x.split()
    s = st.tag(x)
    for i,(_,tag) in enumerate(s):
        if tag in remove_tags_stanfordpos:
            x[i] = ''
    return ''.join(x)
    

# obama_data['text'] = obama_data['text'].apply(pos_tag_filter)
# romney_data['text'] = romney_data['text'].apply(pos_tag_filter)

# remove stopwords


stopwords_list = stopwords.words('english') + manual_stopwords_list

# stemming
class Tokenizer(object):
    def __init__(self,stemmer='porter'):
        self.stemmer = stemmer
        if stemmer == 'wordnet':
            self.wnl = WordNetLemmatizer()
        if stemmer == 'porter':
            self.wnl = PorterStemmer()
        if stemmer == 'snowball':
            self.wnl = SnowballStemmer('english')
    def __call__(self, doc):
        if self.stemmer == 'wordnet':
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        else:
            return [self.wnl.stem(t) for t in word_tokenize(doc)]
                
            
def get_X_y(data):
    return data['text'],data['sentiment'].astype(int)


texts = obama_data['text']
labels = np.array(obama_data['sentiment'])

tokenizer = keras.preprocessing.text.Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) #list of lists, basically replaces each word with number

tokens = []
myTokenizer = Tokenizer(STEMMER)
for i in range(0,len(texts)):
    try:
        tokens.append(myTokenizer.__call__(texts[i]))
    except UnicodeDecodeError:
        pass
word_dict = {}
winx = 1
mysequences = []
tsq = []
for i in range(0,len(tokens)):
    for token in tokens[i]:
        if token not in word_dict:
            word_dict[token] = winx
            winx += 1
        tsq.append(word_dict[token])
    mysequences.append(tsq)
    tsq = []


    
word_index = tokenizer.word_index #key = word, value = number
#word_index = word_dict
#sequences = mysequences
if TOKENIZER == 'nltk':
    word_index = word_dict
    sequences = mysequences
print('Found %s unique tokens.' % len(word_index))

#pad the data 
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print labels[0:4]
Y = labels
labels = keras.utils.np_utils.to_categorical(labels,nb_classes=3)
print labels[0:4]

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



embeddings_index = {}
f = open(GLOVE_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#prepare embedding matrix

#num_words = min(MAX_NB_WORDS, len(word_index))
num_words = len(word_index)+1
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


#create the embedding layer


# create the model
np.random.seed(1)

def build_model():
    np.random.seed(1)
    l2 = regularizers.l2(0.01)
    l22 = regularizers.l2(0.01)
    model = Sequential()
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=1)
    model.add(embedding_layer)
    #model.add(LSTM(10,return_sequences=True))
    model.add(LSTM(150,return_sequences=False,W_regularizer=l2))
    #model.add(LSTM(15,return_sequences=False,W_regularizer=l22))
    #model.add(Dropout(0.2))
    #model.add(Dense(10, activation='relu'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=False)
    model.add(Dense(len(labels[0]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


#k fold cross validaiton
avg_acc = []
avg_f1 = []

#data = data[0:500]
kf = KFold(n=len(data),n_folds=10)
#kf = StratifiedKFold(Y,n_folds=10)
for train,test in kf: #do the cross validation
    np.random.seed(1)
    x_train, x_val, y_train, y_val = data[train], data[test], labels[train], labels[test]
    
    model = build_model()
    model.fit(x_train, y_train, nb_epoch=10, batch_size=64,verbose=0)
    y_true,y_pred = get_Ytrue_Ypred(model,x_val,y_val)
    avg_acc.append(accuracy_score(y_true,y_pred))
    avg_f1.append(f1_score(y_true,y_pred,average='binary'))      
    print classification_report(y_true,y_pred)


#print classification_report(y_true,y_pred)
print 'Average f1-score = ', np.mean(np.array(avg_f1))
#print 'Overall Accuracy = ',100.0*np.mean(np.array(avg_acc)),'%'

#
#
#get = word_index.items()
#for i in range(0,len(get)):
#    if get[i][1] == 6760:
#        print get[i][0]
#
#np.random.seed(1)
#model = build_model()
#history = model.fit(data, labels, nb_epoch=70, validation_split=0.2, shuffle=True)
#train_loss = history.history['loss']
#val_loss = history.history['val_loss']
#plt.plot(range(0,len(train_loss)),train_loss,label='train')
#plt.plot(range(0,len(train_loss)),val_loss,label='val')
#plt.legend(loc='left')
