# This Python 3 environment comes with many helpful analytics libraries installed
#Fast_Text + Embedding
#Ensemble of the 5 Fold 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import string
import time
from collections import defaultdict
#from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import f1_score as F1_Metric_Score

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,Flatten,SpatialDropout1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Activation, CuDNNGRU, Conv1D,LSTM,MaxPooling1D
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
import math

import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K

#Preprocessing
from functools import partial
import re


from collections import Counter

from multiprocessing import Pool
from nltk.corpus import stopwords
import gc

stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#from sklearn.pipeline import make_pipeline
#from sklearn.base import BaseEstimator, TransformerMixin,clone
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import FunctionTransformer
#from sklearn.pipeline import FeatureUnion


#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import StandardScaler
#from scipy.sparse import csr_matrix, hstack,vstack
from tqdm import tqdm
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV

#from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score

#############################################################################
print("Start!!!!")
print('*'*50)

cores = 4
max_text_length=50

def Parallelize_function(df:pd.DataFrame,function)->pd.Series:
    return df.apply( lambda s : function(str(s)))

#Function 2 , it's better general
def parallelize_dataframe2(df,f_arg, func):
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(partial(func, function=f_arg), df_split))
    pool.close()
    pool.join()
    return df

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
    
contraction_patterns = { (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), 
                        (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),(r'(\w+)\'ve', '\g<1> have'),
                        (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'),
                        (r'dont', 'do not'), (r'wont', 'will not')}

patterns = {(re.compile(regex), repl) for (regex, repl) in contraction_patterns}
    
def Contraction(text):
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text
    
def clean_str(text):
    try:
        text=Contraction(text)
        text = text.lower()
        text=re.sub(r'\b(\d+)([a-z]+)\b', r'\1 \2',text)
        text= re.sub(r'[^\w\s]', '', text)
        text = ' '.join( [ wordnet_lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words ] )
        text=re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join(re.split('(\d+)',text))
        text = re.sub( "\s+", " ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text



chars = re.escape(string.punctuation)

def clean_str2(text):
    
    try:
        text = ' '.join( [w for w in text.split()[:max_text_length]])
        text=Contraction(text)
        text = text.lower()
        text=re.sub('['+chars+']',' ',text)
        text=re.sub(r'\b(\d+)([a-z]+)\b', r'\1 \2',text)
        text=re.sub('[0-9]+', ' ', text)
        text= re.sub(r'[^\w\s]', '', text)
        #text = ' '.join( [ w for w in text.split() if w not in stop_words ] )
        text=re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join(re.split('(\d+)',text))
        text = re.sub( "\s+", " ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text

def clean_str3(text):
    
    try:
        text = ' '.join( [w for w in text.split()])
        text=Contraction(text)
        text = text.lower()
        text=re.sub(r'\b(\d+)([a-z]+)\b', r'\1 \2',text)
        text=re.sub('['+chars+']',' ',text)
        text=re.sub('[0-9]+', ' #', text)
        #text= re.sub(r'[^\w\s]', '', text)
        #text = ' '.join( [ w for w in text.split() if w not in stop_words ] )
        #text=re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join(re.split('(\d+)',text))
        text = re.sub( "\s+", " ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text


#def Max_LenghtWords(String):
#    stats={k:len(k) for k in String.split(' ')}
#    return len(max(stats, key=stats.get))

#def Ratio_Words(df,colum='question_text'):
#    df=df.copy()
#    df[colum]=df[colum].astype('str')
#    df['num_words_description']=df[colum].apply(lambda x:len(x.split()))
#    df['num_unique_description']=df[colum].apply(lambda x: len(set(w for w in x.split())))
#    df['Ratio_Words_description']=(df['num_unique_description']/df['num_words_description'])* 100
#    return df.drop(labels=['num_words_description','num_unique_description'],axis=1)
#def Clean_String(df):
#    return df.apply(lambda x: clean_str(x))

#def Add_Clean_String(df):
#    df['Clean_String']=parallelize_dataframe(df['question_text'],Clean_String)
#    return df

#def Ratio_Words_PostCleaning(df,colum='question_text'):
#    df=df.copy()
#    df[colum]=df[colum].astype('str')
#    df=df.pipe(Add_Clean_String)
#    df['num_words_description']=df['Clean_String'].apply(lambda x:len(set(w for w in x.split()))).fillna(0)
#    df['num_unique_description']=df[colum].apply(lambda x: len(set(w for w in x.split()))).fillna(1)
#    df['Ratio_Words_PostCleaning']=(df['num_words_description']/df['num_unique_description'])* 100
#    return df.drop(labels=['num_words_description','num_unique_description','Clean_String'],axis=1)

#def Add_FirstWords(DF):
    #df=DF.copy()
#    DF['What']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='What' else 0)
#    DF['How']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='How' else 0)
#    DF['Why']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Why' else 0)
#    DF['Is']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Is' else 0)
#    DF['Can']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Can' else 0)
#    DF['Which']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Which' else 0)
#    DF['Do']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Do' else 0)
#    DF['If']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='If' else 0)
#    DF['Are']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Are' else 0)
#    DF['Who']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Who' else 0)
#    return DF

#def Reduction_String(String):
#    max_text_length=80
#    s=str(String)
#    s = ' '.join( [w for w in s.split()[:max_text_length]] )
#    return s
    

start_time = time.time()
print("Load the Data !!!!")
print('*'*50)
print()
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)


print('*'*50)
print("Processing Text Step 1")
train.loc[:,'question_text2']=parallelize_dataframe2(train.question_text,clean_str2,Parallelize_function)
test.loc[:,'question_text2']=parallelize_dataframe2(test.question_text,clean_str2,Parallelize_function)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train.question_text2)+list(test.question_text2))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Max_Emb=len(word_index) + 1
maxlen=45
X_train = tokenizer.texts_to_sequences(train.question_text2.values)
X_test = tokenizer.texts_to_sequences(test.question_text2.values)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen,padding="post",truncating="post")
x_test = sequence.pad_sequences(X_test, maxlen=maxlen,padding="post",truncating="post")



print('Shape of data tensor:', x_train.shape)
print('Shape of data tensor:', x_test.shape)

del X_train,X_test
del tokenizer,sequence
gc.collect()

#Validation
#question_text_train=parallelize_dataframe2(train.question_text,Reduction_String,Parallelize_function)
#question_text_Out=parallelize_dataframe2(question_text_train,clean_str3,Parallelize_function)
#question_text_Out.to_frame().to_csv(r'Coorpus_Question_Quora.txt', index=False, sep=' ',
 #                         header=False, escapechar=" ")

#del question_text_train,question_text_Out

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.9, random_state=0)
L1,L2=sss.split(train.question_text,train.target)

M_train=x_train[L1[1]]
M_Val=x_train[L1[0]]
DF_Train=train.iloc[L1[1],:].copy()
DF_Validation=train.iloc[L1[0],:].copy()
Target_Train=DF_Train.target.copy()
Target_Val=DF_Validation.target.copy()

print('Shape of data general tensor:', DF_Train.shape)
print('Shape of data general tensor:', DF_Validation.shape)
print('Shape of data general test:', test.shape)
print('Shape of data tensor Matrix Train:', M_train.shape)
print('Shape of data tensor Matrix Validation:', M_Val.shape)
print('Shape of data tensor Matrix Test:', x_test.shape)
print('Shape of data tensor Target:', Target_Train.shape)
print('Shape of data tensor Target Val:', Target_Val.shape)

print("Save DataFrame")
DF_Train.to_parquet("DF_Train.parquet")
DF_Validation.to_parquet("DF_Validation.parquet")
test.to_parquet("test.parquet")
train.to_parquet("train.parquet")
np.save('M_train.npy',M_train)
np.save('M_Val.npy',M_Val)


import pyarrow as pa
import pyarrow.parquet as pq

df_train = pd.DataFrame(x_train)
table = pa.Table.from_pandas(df_train)
pq.write_table(table, 'x_train.parquet')

df_test = pd.DataFrame(x_test)
table = pa.Table.from_pandas(df_test)
pq.write_table(table, 'x_test.parquet')

del table
del M_Val,M_train
del df_train,x_train
del df_test,x_test
del DF_Train,DF_Validation,test,train
gc.collect()
print('*'*50)
print()


print('*'*50)
print("Embedding 1 !!!!")
print()
EMBEDDING_FILE1 = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1,encoding='utf-8'))
print('Found %s word vectors.' % len(embeddings_index1))

Auxiliar_Word=embeddings_index1.get('something')
embedding_matrix1 = np.zeros((len(word_index) + 1, 300),dtype=np.float32)
for word, i in word_index.items():
    embedding_vector = embeddings_index1.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector
    else:
        embedding_matrix1[i] = Auxiliar_Word
        
del embeddings_index1,
del embedding_vector,Auxiliar_Word
gc.collect()        

print("Save embedding 1")
np.save('embedding_matrix1.npy',embedding_matrix1)
del embedding_matrix1
gc.collect()

print('*'*50)
print("Embedding 2 !!!!")
print()

EMBEDDING_FILE2 = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
embeddings_index2 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE2,encoding='utf-8'))
print('Found %s word vectors.' % len(embeddings_index2))

Auxiliar_Word=embeddings_index2.get('something')
embedding_matrix2 = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index2.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector
    else:
        embedding_matrix2[i] =Auxiliar_Word 
        
np.save('embedding_matrix2.npy',embedding_matrix2)        
del embeddings_index2
gc.collect()
del embedding_matrix2
gc.collect()


EMBEDDING_FILE3 = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
embeddings_index3 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE3, encoding="utf8", errors='ignore') if len(o)>100)
print('Found %s word vectors.' % len(embeddings_index3))

Auxiliar_Word=embeddings_index3.get('something')
embedding_matrix3 = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index3.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix3[i] = embedding_vector
    else:
        embedding_matrix3[i] =Auxiliar_Word 
        
np.save('embedding_matrix3.npy',embedding_matrix3)        
del embeddings_index3
gc.collect()

print('*'*50)
print()

#from fastText import train_unsupervised
#model_ft=train_unsupervised(input=os.path.join(os.getenv("DATADIR", ''), 'Coorpus_Question_Quora.txt'),model='skipgram',dim=400)
#model_ft.save_model("Quora_FastText.bin")


#embedding_matrix2 = np.zeros((len(word_index) + 1, 400))
#for word, i in word_index.items():
#    embedding_vector = model_ft.get_word_vector(re.sub('\\n','',word))
#    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
#        embedding_matrix2[i] = embedding_vector

        
#del embedding_vector
#del model_ft
#gc.collect()

embedding_matrix1=np.load('embedding_matrix1.npy')
embedding_matrix2=np.load('embedding_matrix2.npy')
print('Shape of Embedding1:', embedding_matrix1.shape)
print('Shape of Embedding 2:', embedding_matrix2.shape)
print('Shape of Embedding 2:', embedding_matrix3.shape)


#Embedding_General=np.concatenate([embedding_matrix1,embedding_matrix2],axis=1)
#print('Shape of Embedding General:', Embedding_General.shape)
print("Normalization Embeddings")
print('Shape of Embedding1:', embedding_matrix1.mean(0).reshape(-1,1).shape)
print('Shape of Embedding1:', embedding_matrix1.mean(1).reshape(-1,1).shape)

embedding_matrix1=(embedding_matrix1-embedding_matrix1.mean(1).reshape(-1,1))
embedding_matrix2=(embedding_matrix2-embedding_matrix2.mean(1).reshape(-1,1))
embedding_matrix3=(embedding_matrix3-embedding_matrix3.mean(1).reshape(-1,1))

print('Shape of Embedding1:', embedding_matrix1.shape)
print('Shape of Embedding 2:', embedding_matrix2.shape)
print('Shape of Embedding 2:', embedding_matrix3.shape)


#del embedding_matrix1,embedding_matrix2

M_train=np.load('M_train.npy')
M_Val=np.load('M_Val.npy')
print('Shape of data tensor Matrix Train:', M_train.shape)
print('Shape of data tensor Matrix Validation:', M_Val.shape)

print('[{}] Finished TRAIN DATA PREPARATION'.format(time.time() - start_time))


#print("Processing Question !!!!")
#print('*'*50)



#train_stage1=(train.pipe(Add_FirstWords)
#       .assign(Lenght_by_Words=lambda X:X['question_text'].apply(lambda x: len(str(x).split(" "))),
#             Lenght_by_Char=lambda X:X['question_text'].apply(lambda x: len(x)),
#             Max_Lenght_String=lambda X:X['question_text'].apply(lambda x: Max_LenghtWords(x) if len(x)>1  else 0))
#      .pipe(Ratio_Words)
#    .pipe(Add_Clean_String)
#    .assign(Clean_String_Length=lambda X:X['Clean_String'].astype('object').apply(lambda x: len(x.split(' '))))
#    .drop(labels=['Clean_String'],axis=1)
#    .pipe(Ratio_Words_PostCleaning).drop(labels=['qid'],axis=1)
#            .assign(target=lambda X:X['target'].astype(np.uint8),
#                    What=lambda X:X['What'].astype(np.uint8),
#                    How=lambda X:X['How'].astype(np.uint8),
#                    Why=lambda X:X['Why'].astype(np.uint8),
#                    Is=lambda X:X['Is'].astype(np.uint8),
#                    Can=lambda X:X['Can'].astype(np.uint8),
#                    Which=lambda X:X['Which'].astype(np.uint8),
#                    Do=lambda X:X['Do'].astype(np.uint8),
#                    If=lambda X:X['If'].astype(np.uint8),
#                    Are=lambda X:X['Are'].astype(np.uint8),
#                    Who=lambda X:X['Who'].astype(np.uint8),
#                    Lenght_by_Words=lambda X:X['Lenght_by_Words'].astype(np.int32),
#                    Lenght_by_Char=lambda X:X['Lenght_by_Char'].astype(np.int32),
#                    Max_Lenght_String=lambda X:X['Max_Lenght_String'].astype(np.int32),
#                    Ratio_Words_description=lambda X:X['Max_Lenght_String'].astype(np.float32),
 #                   Clean_String_Length=lambda X:X['Max_Lenght_String'].astype(np.int32),
 #                   Ratio_Words_PostCleaning=lambda X:X['Max_Lenght_String'].astype(np.float32)))
#del train
#gc.collect()


#test_stage1=(test.pipe(Add_FirstWords)
#       .assign(Lenght_by_Words=lambda X:X['question_text'].apply(lambda x: len(str(x).split(" "))),
#             Lenght_by_Char=lambda X:X['question_text'].apply(lambda x: len(x)),
#             Max_Lenght_String=lambda X:X['question_text'].apply(lambda x: Max_LenghtWords(x) if len(x)>1  else 0))
#      .pipe(Ratio_Words)
#    .pipe(Add_Clean_String)
#    .assign(Clean_String_Length=lambda X:X['Clean_String'].apply(lambda x: len(x.split(' '))))
#    .drop(labels=['Clean_String'],axis=1)
#    .pipe(Ratio_Words_PostCleaning).drop(labels=['qid'],axis=1)
#            .assign(What=lambda X:X['What'].astype(np.uint8),
#                    How=lambda X:X['How'].astype(np.uint8),
#                    Why=lambda X:X['Why'].astype(np.uint8),
#                    Is=lambda X:X['Is'].astype(np.uint8),
#                    Can=lambda X:X['Can'].astype(np.uint8),
#                    Which=lambda X:X['Which'].astype(np.uint8),
#                    Do=lambda X:X['Do'].astype(np.uint8),
#                    If=lambda X:X['If'].astype(np.uint8),
#                    Are=lambda X:X['Are'].astype(np.uint8),
#                    Who=lambda X:X['Who'].astype(np.uint8),
#                    Lenght_by_Words=lambda X:X['Lenght_by_Words'].astype(np.int32),
#                    Lenght_by_Char=lambda X:X['Lenght_by_Char'].astype(np.int32),
#                    Max_Lenght_String=lambda X:X['Max_Lenght_String'].astype(np.int32),
#                    Ratio_Words_description=lambda X:X['Max_Lenght_String'].astype(np.float32),
#                    Clean_String_Length=lambda X:X['Max_Lenght_String'].astype(np.int32),
#                    Ratio_Words_PostCleaning=lambda X:X['Max_Lenght_String'].astype(np.float32)))

#del test
#gc.collect()


#print("Train shape : ",train_stage1.shape)
#print("Test shape : ",test_stage1.shape)

###################################################################

print("Building the NN")
print('*'*50)

start_time = time.time()
batch_size = 128

def batch_train_gen(df_train,df_target):
    n_batches = math.ceil(df_train.shape[0] / batch_size)
    indices = np.arange(df_train.shape[0])
    np.random.shuffle(indices)

    while True: 
        #data = DF_Train[indices]
        MT =df_train[indices]
        labels =df_target.iloc[indices].values
        for i in range(n_batches):
            out_arr = MT[i*batch_size:(i+1)*batch_size,:]
            out_target=labels[i*batch_size:(i+1)*batch_size]
        
            yield out_arr, out_target

#####################################################################
from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

############################################################
def f1_b(y_true, y_pred):

    def recall(y_true, y_pred):
        """Recall metric.
  
        Only computes a batch-wise average of recall.
  
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        true_positives = K.sum(y_true * y_pred)
        possible_positives = K.sum(y_true)
        #possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

         Only computes a batch-wise average of precision.

       Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        #predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        true_positives = K.sum(y_true * y_pred)
        predicted_positives = K.sum(y_pred)
        
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        
    y_pred = K.tf.where(y_pred > 0.3,K.tf.ones_like(y_pred),K.tf.zeros_like(y_pred))    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

        
#############################################################
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=regularizers.l2(1e-8), b_regularizer= regularizers.l2(1e-8),
                 bias=False, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        #self.W_constraint = constraints.get(W_constraint)
        #self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='context',
                                 regularizer=self.W_regularizer,
                                 trainable=True)
        self.features_dim = input_shape[-1]
    	    

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='context_b',
                                     regularizer=self.b_regularizer)
        else:
            self.b = None

        self.built = True
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
        
#############################################################
import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K

class AttentionWithContext(Layer):

    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    refer https://github.com/fchollet/keras/issues/4962
    refer https://gist.github.com/rmdort/596e75e864295365798836d9e8636033
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight((input_shape[2], 1,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        # word context vector uw
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # in the paper refer equations (5) on page 3
        # (batch, time_steps, 40) x (40, 1)
        W_w_dot_h_it =  K.dot(x, self.kernel) # (batch, 40, 1)
        W_w_dot_h_it = K.squeeze(W_w_dot_h_it, -1) # (batch, 40)
        W_w_dot_h_it = W_w_dot_h_it + self.b # (batch, 40) + (40,)
        uit = K.tanh(W_w_dot_h_it) # (batch, 40)

        # in the paper refer equations (6) on page 3
        uit_dot_uw = uit * self.u # (batch, 40) * (40, 1) => (batch, 1)
        ait = K.exp(uit_dot_uw) # (batch, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(batch, 40)
            ait = mask*ait #(batch, 40) * (batch, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        # sentence vector si is returned
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[-1],)
############################################################
clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=500., mode='exp_range')


#K.clear_session()
#start_time = time.time()
print("Model 1")    
#model = Sequential()
#model.add(Embedding(Max_Emb, 700,weights=[Embedding_General],trainable=False,name='Embedding'))
#model.add(Dropout(0.5))
#model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True,input_shape=(45,700),name='Layer_1')))
#model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True),name='Layer_2'))
#model.add(Attention(maxlen))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation="sigmoid"))
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['acc',f1_b])


#model.fit_generator(batch_train_gen(M_train,Target_Train), epochs=20,
#                    steps_per_epoch=1000,
#                     validation_data=(M_Val,Target_Val),callbacks=[clr], verbose=2)


#y_pred=model.predict(M_Val,batch_size=512, verbose=1)
#y_Val_Pred = (np.array(y_pred) > 0.3).astype(np.int)
MAX_SENT_LEN=45
words_encoder_with_attention1 = Sequential([
    Embedding(Max_Emb, 300, weights=[embedding_matrix1], 
               input_length=MAX_SENT_LEN,trainable=False),
               Dropout(0.4),
    Bidirectional(CuDNNLSTM(40, return_sequences=True)),
    AttentionWithContext()])

words_encoder_with_attention2 = Sequential([
    Embedding(Max_Emb, 300, weights=[embedding_matrix2], 
               input_length=MAX_SENT_LEN,trainable=False),
               Dropout(0.4),
    Bidirectional(CuDNNLSTM(40, return_sequences=True)),
    AttentionWithContext()])

words_encoder_with_attention3 = Sequential([
    Embedding(Max_Emb, 300, weights=[embedding_matrix3], 
               input_length=MAX_SENT_LEN,trainable=False),
               Dropout(0.4),
    Bidirectional(CuDNNLSTM(40, return_sequences=True)),
    AttentionWithContext()])

#Change GRU --> LSTM
doc_input_layer = Input(shape=(1,MAX_SENT_LEN), dtype='int32')
#words_encoder_with_attention=concatenate( [words_encoder_with_attention1, words_encoder_with_attention2])
word_encoder_for_each_sentence1 = TimeDistributed(words_encoder_with_attention1)(doc_input_layer)
word_encoder_for_each_sentence2 = TimeDistributed(words_encoder_with_attention2)(doc_input_layer)
word_encoder_for_each_sentence3 = TimeDistributed(words_encoder_with_attention3)(doc_input_layer)

gru_layer1 = Bidirectional(CuDNNGRU(40, return_sequences=True))(word_encoder_for_each_sentence1)
sentence_attention_layer1 = AttentionWithContext()(gru_layer1)
gru_layer2 = Bidirectional(CuDNNGRU(40, return_sequences=True))(word_encoder_for_each_sentence2)
sentence_attention_layer2 = AttentionWithContext()(gru_layer2)
gru_layer3 = Bidirectional(CuDNNGRU(40, return_sequences=True))(word_encoder_for_each_sentence3)
sentence_attention_layer3 = AttentionWithContext()(gru_layer3)
sentence_attention_layer=concatenate([sentence_attention_layer1,sentence_attention_layer2,sentence_attention_layer3])
sentence_attention_layer = Dense(16, activation="relu")(sentence_attention_layer)
sentence_attention_layer=Dropout(0.5)(sentence_attention_layer)
doc_output_layer = Dense(1, activation='sigmoid')(sentence_attention_layer)
sentence_encoder_with_attention = Model(doc_input_layer, doc_output_layer)
sentence_encoder_with_attention.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])

M_train=M_train.reshape((1175510,1,45))
M_Val=M_Val.reshape((130612,1,45))

sentence_encoder_with_attention.fit(M_train,Target_Train ,validation_data=(M_Val, Target_Val),
                                    epochs=10, batch_size=256, shuffle=True,verbose=2)

y_pred=sentence_encoder_with_attention.predict(M_Val,batch_size=1024, verbose=0)
y_Val_Pred = (np.array(y_pred) > 0.3).astype(np.int)
print()
print("F1 Prediction ",F1_Metric_Score(Target_Val,y_Val_Pred))
#print("F1 stimaded on the epoch",np.mean(sentence_encoder_with_attention.history.history['f1_b']))
print(sentence_encoder_with_attention.summary())


x_test= pq.read_table('x_test.parquet')
x_test=x_test.to_pandas().values
#batch_size = 128
#def batch_gen(x_test):
#    n_batches = math.ceil(len(x_test) / batch_size)
#    for i in range(n_batches):
#        text_arr = x_test[i*batch_size:(i+1)*batch_size,:]
        #text_arr = Trans_Vector_Words.fit_transform(texts).reshape(texts.shape[0],1,300)
#        yield text_arr

#test_df = pd.read_csv("../input/test.csv")

#all_preds = []
#for x in tqdm(batch_gen(x_test)):
#    all_preds.extend(model.predict(x).flatten())

    
#y_te = (np.array(all_preds) > 0.3).astype(np.int)
Dim=x_test.shape[0]
x_test2=x_test.reshape((Dim,1,45))
print("Dim Text:",x_test2.shape)
y_pred=sentence_encoder_with_attention.predict(x_test2,batch_size=1024, verbose=0)
y_te = (np.array(y_pred) > 0.3).astype(np.int)


test= pq.read_table('test.parquet')
test=test.to_pandas()

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te.ravel()})
submit_df.to_csv("submission.csv", index=False) 
print('[{}] Finished Train NN'.format(time.time() - start_time))
