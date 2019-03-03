
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as F1_Metric_Score


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




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,Flatten,SpatialDropout1D,concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D,Activation, CuDNNGRU, Conv1D,LSTM,MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, ZeroPadding1D, GlobalMaxPool1D,BatchNormalization,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import Concatenate, Dot, Multiply, RepeatVector
from keras.layers import  TimeDistributed

from keras.layers import SimpleRNN, Lambda, Permute
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D

from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed

import math

import tensorflow as tf

from keras.engine import Layer, InputSpec,InputLayer
from keras import regularizers, initializers, constraints,optimizers, layers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import concatenate
import tensorflow as tf
from keras import backend as K

from keras.layers.core import Reshape, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.constraints import maxnorm
from keras.regularizers import l2


############# Global ##################
cores = 2
EMBEDDING_FILE1 = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
EMBEDDING_FILE2 = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
EMBEDDING_FILE3 = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

########################################


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

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def Get_FeaturesNames_coefficients(coefficients, feature_names, n_top_features=40):
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients,
                                          positive_coefficients])
    feature_names = np.array(feature_names)
    return pd.DataFrame({'Words':feature_names[interesting_coefficients],'Coeff':coef[interesting_coefficients]})

########## Text Processing ######################################
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

def clean_str2(text):
    try:
        text=Contraction(text)
        text = text.lower()
        text=re.sub(r'\b(\d+)([a-z]+)\b', r'\1 \2',text)
        text= re.sub(r'[^\w\s]', '', text)
        text = re.sub('[0-9]{5,}', '#####',text)
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
        text = ' '.join( [ w for w in text.split() if w not in stop_words ] )
        text=re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join(re.split('(\d+)',text))
        text = re.sub( "\s+", " ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text

######################### Extra  Features ######################
def Max_LenghtWords(String):
    stats={k:len(k) for k in String.split(' ')}
    return len(max(stats, key=stats.get))

def Ratio_Words(df,colum='question_text'):
    df=df.copy()
    df[colum]=df[colum].astype('str')
    df['num_words_description']=df[colum].apply(lambda x:len(x.split()))
    df['num_unique_description']=df[colum].apply(lambda x: len(set(w for w in x.split())))
    df['Ratio_Words_description']=(df['num_unique_description']/df['num_words_description'])* 100
    return df.drop(labels=['num_words_description','num_unique_description'],axis=1)
def Clean_String(df):
    return df.apply(lambda x: clean_str(x))

def Add_Clean_String(df):
    df['Clean_String']=parallelize_dataframe(df['question_text'],Clean_String)
    return df

def Ratio_Words_PostCleaning(df,colum='question_text'):
    df=df.copy()
    df[colum]=df[colum].astype('str')
    df=df.pipe(Add_Clean_String)
    df['num_words_description']=df['Clean_String'].apply(lambda x:len(set(w for w in x.split()))).fillna(0)
    df['num_unique_description']=df[colum].apply(lambda x: len(set(w for w in x.split()))).fillna(1)
    df['Ratio_Words_PostCleaning']=(df['num_words_description']/df['num_unique_description'])* 100
    return df.drop(labels=['num_words_description','num_unique_description','Clean_String'],axis=1)

def Add_FirstWords(DF):
    #df=DF.copy()
    DF['What']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='What' else 0)
    DF['How']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='How' else 0)
    DF['Why']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Why' else 0)
    DF['Is']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Is' else 0)
    DF['Can']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Can' else 0)
    DF['Which']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Which' else 0)
    DF['Do']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Do' else 0)
    DF['If']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='If' else 0)
    DF['Are']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Are' else 0)
    DF['Who']=DF['question_text'].apply(lambda x: 1 if str(x).split()[:1][0]=='Who' else 0)
    return DF

def Preprocessing_DF(DF):
    df=DF.copy()
    df=(df.pipe(Add_FirstWords)
       .assign(Lenght_by_Words=lambda X:X['question_text'].apply(lambda x: len(str(x).split(" "))),
             Lenght_by_Char=lambda X:X['question_text'].apply(lambda x: len(x)),
             Max_Lenght_String=lambda X:X['question_text'].apply(lambda x: Max_LenghtWords(x) if len(x)>1  else 0))
      .pipe(Ratio_Words)
    .pipe(Add_Clean_String)
    .assign(Clean_String_Length=lambda X:X['Clean_String'].apply(lambda x: len(x.split(' '))))
    .drop(labels=['Clean_String'],axis=1)
    .pipe(Ratio_Words_PostCleaning).drop(labels=['qid'],axis=1)
            .assign(What=lambda X:X['What'].astype(np.uint8),
                    How=lambda X:X['How'].astype(np.uint8),
                    Why=lambda X:X['Why'].astype(np.uint8),
                    Is=lambda X:X['Is'].astype(np.uint8),
                    Can=lambda X:X['Can'].astype(np.uint8),
                    Which=lambda X:X['Which'].astype(np.uint8),
                    Do=lambda X:X['Do'].astype(np.uint8),
                    If=lambda X:X['If'].astype(np.uint8),
                    Are=lambda X:X['Are'].astype(np.uint8),
                    Who=lambda X:X['Who'].astype(np.uint8),
                    Lenght_by_Words=lambda X:X['Lenght_by_Words'].astype(np.int32),
                    Lenght_by_Char=lambda X:X['Lenght_by_Char'].astype(np.int32),
                    Max_Lenght_String=lambda X:X['Max_Lenght_String'].astype(np.int32),
                    Ratio_Words_description=lambda X:X['Max_Lenght_String'].astype(np.float32),
                    Clean_String_Length=lambda X:X['Max_Lenght_String'].astype(np.int32),
                    Ratio_Words_PostCleaning=lambda X:X['Max_Lenght_String'].astype(np.float32)))
    return df

##################### Auxiliar Transformations ########################
class DataFrameSelect(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values


################# Rigde Classification ###############################

numeric_features1 = ['Lenght_by_Words','Lenght_by_Char',
                    'Max_Lenght_String','Clean_String_Length']
numeric_features2 = ['Ratio_Words_description','Ratio_Words_PostCleaning']

numeric_transformer1 = Pipeline(steps=[
    ('Log1p',FunctionTransformer(np.log1p, validate=True)),
    ('scaler', StandardScaler())])
numeric_transformer2 = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_features = ['What','How','Why','Is','Can','Which','Do','If','Are', 'Who']
categorical_transformer = Pipeline([
    ('Selection_Cat', DataFrameSelect(attribute_names=categorical_features))])

feat_text=['question_text']
feat_text_transform=Pipeline(steps=[
    ('Selection_Text', DataFrameSelect(attribute_names='question_text')),
    ('countVect',TfidfVectorizer(binary=True,max_features=200000,dtype=np.uint8,lowercase=False,ngram_range=(1,4),min_df=5,stop_words='english'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num1', numeric_transformer1, numeric_features1),
        ('num2', numeric_transformer2, numeric_features2),
        ('cat', categorical_transformer, categorical_features),
        ('Text',feat_text_transform,feat_text)])

### Fake Embedding

TV_Text = TfidfVectorizer(min_df=5,max_features=100000,ngram_range=(1,1))
clf = LogisticRegression(C=20,max_iter=150,class_weight='balanced',n_jobs=4,solver='saga')
Regression_CLF = Pipeline([('TextTfidfV', TV_Text), ('LogRregression', clf)])

############## NN ####################
batch_size = 128

def batch_train_gen(df_train,df_target,batch_size):
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



############# Load Data ##################
print("Load the Data !!!!")
print('*'*50)
print()
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)



Target=train.target.copy()
del train['target']


print('*'*50)
print("Processing Text Step 1")

print('*'*50)
print("Fake Embedding ")
print('*'*50)
Regression_CLF.fit(train.question_text,Target)
vect_Regression = Regression_CLF.named_steps['TextTfidfV']
feature_names_reg = np.array(vect_Regression.get_feature_names())
coef_Regression = Regression_CLF.named_steps['LogRregression'].coef_[0,:]

DF_SimpleWords=Get_FeaturesNames_coefficients(coef_Regression, feature_names_reg, n_top_features=10000)
Lista_Words=DF_SimpleWords.Words.drop_duplicates().tolist()

print('*'*50)
print("Question ")
print('*'*50)
question_text2_train=parallelize_dataframe2(train.question_text,clean_str2,Parallelize_function)
question_text2_test=parallelize_dataframe2(test.question_text,clean_str2,Parallelize_function)

#question_text_Out.to_frame().to_csv(r'Coorpus_Question_Quora.txt', index=False, sep=' ',
#                          header=False, escapechar=" ")
	
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(question_text2_train)+list(question_text2_test))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


Max_Emb=len(word_index) + 1
maxlen=40
X_train = tokenizer.texts_to_sequences(question_text2_train.values)
X_test = tokenizer.texts_to_sequences(question_text2_test.values)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen,padding="post",truncating="post")
x_test = sequence.pad_sequences(X_test, maxlen=maxlen,padding="post",truncating="post")


print('Shape of data tensor:', x_train.shape)
print('Shape of data tensor:', x_test.shape)

del X_train,X_test
del tokenizer,sequence
gc.collect()

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.9, random_state=0)
L1,L2=sss.split(train.question_text,Target)

M_train=x_train[L1[1]]
M_Val=x_train[L1[0]]
DF_Train=train.iloc[L1[1],:].copy()
DF_Validation=train.iloc[L1[0],:].copy()
Target_Train=Target[L1[1]].copy()
Target_Val=Target[L1[0]].copy()

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
test.to_parquet("DF_Test.parquet")
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
######################### Embeddings ##################################
print('*'*50)
print("Embedding FastText!!!!!!")

#from fastText import train_unsupervised
#model_ft=train_unsupervised(input=os.path.join(os.getenv("DATADIR", ''), 'Coorpus_Question_Quora.txt'),model='skipgram',dim=400)
#model_ft.save_model("Quora_FastText.bin")


#embedding_matrix2 = np.zeros((len(word_index) + 1, 400))
#for word, i in word_index.items():
#    embedding_vector = model_ft.get_word_vector(re.sub('\\n','',word))
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix2[i] = embedding_vector

        
#del embedding_vector
#del model_ft
#gc.collect()

print('*'*50)
print("Embedding 1 !!!!")
print()

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

Vec=np.zeros(300)
Fakeembedding_matrix1 = np.zeros((len(word_index) + 1, 300))
for i,word in enumerate(Lista_Words):
    embedding_vector = embeddings_index1.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        Fakeembedding_matrix1[i] = embedding_vector
    else:
        Fakeembedding_matrix1[i] = Vec

        
del embeddings_index1
del embedding_vector,Auxiliar_Word
gc.collect()        

print("Save embedding 1")
np.save('embedding_matrix1.npy',embedding_matrix1)
del embedding_matrix1
gc.collect()

print('*'*50)
print("Embedding 2 !!!!")
print()


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
        
Vec=np.zeros(300)
fakeembedding_matrix2 = np.zeros((len(word_index) + 1, 300))
for i,word in enumerate(Lista_Words):
    embedding_vector = embeddings_index2.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        fakeembedding_matrix2[i] = embedding_vector
    else:
        fakeembedding_matrix2[i] =Vec 

embedding_matrixG=(Fakeembedding_matrix1+fakeembedding_matrix2)/2

Dic_Embedding={k:embedding_matrixG[1,:] for i,k in enumerate(Lista_Words)}

del Fakeembedding_matrix1,fakeembedding_matrix2

np.save('embedding_matrixG.npy',embedding_matrixG)        
np.save('embedding_matrix2.npy',embedding_matrix2)        
del embeddings_index2,embedding_matrix2,embedding_matrixG
gc.collect()



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
del embeddings_index3,embedding_matrix3
gc.collect()

print('*'*50)
print("Finish preprocessing!!!")

del question_text2_test,question_text2_train,feature_names_reg,coef_Regression,Vec,Auxiliar_Word


########################### Model 1 ####################################
################################# Ridge Regression ######################
print("Ridge Regression")
print("Load Data")

DF_Train= pq.read_table('DF_Train.parquet')
DF_Train=DF_Train.to_pandas()

DF_Validation= pq.read_table('DF_Validation.parquet')
DF_Validation=DF_Validation.to_pandas()

DF_Test= pq.read_table('DF_Test.parquet')
DF_Test=DF_Test.to_pandas()

print('Shape of data general tensor:', DF_Train.shape)
print('Shape of data general tensor:', DF_Validation.shape)
print('Shape of data general test:', DF_Test.shape)
print('Shape of data tensor Target:', Target_Train.shape)
print('Shape of data tensor Target Val:', Target_Val.shape)

print("Preprocessing 1")
DF_Train=Preprocessing_DF(DF_Train)
DF_Validation=Preprocessing_DF(DF_Validation)
DF_Test=Preprocessing_DF(DF_Test)
print("Save Processing DF")

DF = pd.DataFrame(DF_Train)
table = pa.Table.from_pandas(DF)
pq.write_table(table, 'DF_Train_Processing.parquet')

DF = pd.DataFrame(DF_Validation)
table = pa.Table.from_pandas(DF)
pq.write_table(table, 'DF_Val_Processing.parquet')

DF = pd.DataFrame(DF_Test)
table = pa.Table.from_pandas(DF)
pq.write_table(table, 'DF_Test_Processing.parquet')

print("Preprocessing 2")
preprocessor.fit(DF_Train)

L_Train=preprocessor.transform(DF_Train)
L_Val=preprocessor.transform(DF_Validation)
L_Test=preprocessor.transform(DF_Test)

Clf=RidgeClassifier(fit_intercept=True,normalize=False,random_state=2019,tol=0.0025,alpha=20,max_iter=150,solver='sag',class_weight='balanced')

del DF_Train,DF_Validation,DF_Test
gc.collect()

print("Estimation 1")
Clf.fit(X=L_Train,y=Target_Train)

Pred_Val_Ridge=Clf.predict(L_Val)
print("F1 Prediction ",F1_Metric_Score(Target_Val,Pred_Val_Ridge))

Pred_Test=Clf.predict(L_Test)

test= pq.read_table('DF_Test.parquet')
test=test.to_pandas()

print("Save Prediction")
Submit_Out={}
Submit_Out['Ridge']=Pred_Val_Ridge

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": Pred_Test})
submit_df.to_csv("submission_Ridge.csv", index=False) 

del submit_df,Pred_Val_Ridge,Pred_Test,L_Train,L_Val,L_Test,test
################################# model 2 ###########################

print("ConvModel 1")

print("Load Embedding GloVe")
embedding_Glove=np.load('embedding_matrix2.npy')



def CNN_model():
    doc_input = Input(shape=(40,), dtype="int32")
    doc_embedding = Embedding(Max_Emb, 300, weights=[embedding_Glove], trainable=False)(doc_input)
    #doc_embedding = Dropout(0.3)(doc_embedding)
    convs = []
    ngram_filters = [15,25,35]
    KernSize=[3,2,1]
    n_filters = 40
    for n_gram,nfilt in zip(ngram_filters,KernSize):
        l_conv1 = Conv1D(filters =40 ,kernel_size =nfilt ,strides = 1,padding="valid",activation="relu")(doc_embedding)
        l_conv1 = GlobalMaxPool1D()(l_conv1)
        convs.append(l_conv1)
    l_concat = Concatenate(axis=1)(convs)
    l_BatNorm=BatchNormalization()(l_concat)
    #l_concat = Dense(40, activation='relu')(l_concat)
    l_OUT=Dropout(0.4)(l_BatNorm)
    #l_concat=Activation('relu')(l_concat)
    #l_blstm = Bidirectional(CuDNNLSTM(32, return_sequences=True))(l_concat)
    
    #l_dropout = Dropout(0.5)(l_BatNorm)
    l_fc = Dense(1, activation='sigmoid')(l_OUT)
    #l_softmax = Dense(1, activation='sigmoid')(l_fc)
    model = Model(inputs=[doc_input], outputs=[l_fc])
    model.compile(loss='binary_crossentropy', metrics=['acc'],optimizer='adam')
    return model 

M_Val=np.load('M_Val.npy')
M_train=np.load('M_train.npy')
M_test= pq.read_table('x_test.parquet')
M_test=M_test.to_pandas().values

print("Estimation Model")
modeloDC=CNN_model()
modeloDC.fit(M_train,Target_Train ,
          validation_data=(M_Val, Target_Val),
          epochs=1, batch_size=128, shuffle=True,verbose=2)

Pred_Val_Cov=modeloDC.predict(M_Val,batch_size=512, verbose=1)
Pred_Val_Cov= (np.array(Pred_Val_Cov) > 0.35).astype(np.int)

print()
print("F1 Prediction ",F1_Metric_Score(Target_Val,Pred_Val_Cov))

print("Save Prediction")
Submit_Out['Conv1']=Pred_Val_Cov

Pred_test_Cov=modeloDC.predict(M_test,batch_size=512, verbose=1)
Pred_test_Cov= (np.array(Pred_test_Cov) > 0.35).astype(np.int)


test= pq.read_table('DF_Test.parquet')
test=test.to_pandas()


submit_df = pd.DataFrame({"qid": test["qid"], "prediction": Pred_test_Cov.ravel()})
submit_df.to_csv("submission_conv1.csv", index=False) 
print("Finish Convolution Model")
del Pred_Val_Cov,Pred_test_Cov,M_Val,M_train,M_test,submit_df
gc.collect()

del DF,DF_SimpleWords,test,table,modeloDC
gc.collect()
########################## model 3 ##############################
print("Model 3: Lstm1")
print("Definition Process")
preprocessor_num1 = ColumnTransformer(transformers=[('num1', numeric_transformer1, numeric_features1)])
preprocessor_num2 = ColumnTransformer(transformers=[('num2', numeric_transformer2, numeric_features1)])
preprocessor_cat2 = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])

print("Load Data")
DF_Train= pq.read_table('DF_Train_Processing.parquet')
DF_Train=DF_Train.to_pandas()

print("Fit the process")
preprocessor_num1.fit(DF_Train) 
preprocessor_num2.fit(DF_Train) 
preprocessor_cat2.fit(DF_Train) 


M_train=np.load('M_train.npy')

def get_keras_Lstm1(df,M):
    X = {'Text':M,
        'lenghts': preprocessor_num1.transform(df),
        'rations': preprocessor_num2.transform(df),
        'cats': preprocessor_cat2.transform(df)}
    return X

train_keras = get_keras_Lstm1(DF_Train,M_train)

print("Embedding")
embedding_Glove=np.load('embedding_matrix2.npy')
print("Size Embedding:",embedding_Glove.shape)


def get_model():
    text = Input(shape=(40,),dtype = 'float32',name='Text')
    lenghts = Input(shape=[train_keras["lenghts"].shape[1]], name="lenghts")
    rations = Input(shape=[train_keras["rations"].shape[1]], name="rations")
    cats = Input(shape=[train_keras["cats"].shape[1]], name="cats")
    x = Embedding(Max_Emb, 300, weights=[embedding_Glove], trainable=False)(text)
    x=Dropout(0.4)(x)
    x= Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    att=Attention(40)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([att,avg_pool, max_pool,lenghts,rations,cats])
    x=Dropout(0.3)(x)
    outp = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[text,lenghts,rations,cats], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

lstm1_nn=get_model()
print("Fitting Lstm1 model ...")
BATCH_SIZE=128
#epochs=10
M_Val=np.load('M_Val.npy')
DF_Validation= pq.read_table('DF_Val_Processing.parquet')
DF_Validation=DF_Validation.to_pandas()


Val_keras=get_keras_Lstm1(DF_Validation,M_Val)
lstm1_nn.fit(train_keras, Target_Train, 
                      batch_size=BATCH_SIZE, epochs=1, verbose=2,validation_data=(Val_keras,Target_Val) )

del train_keras,DF_Train,M_train
gc.collect()

Pre_val_lstm1=lstm1_nn.predict(Val_keras,batch_size=512, verbose=1)
Pre_val_lstm1 = (np.array(Pre_val_lstm1) > 0.3).astype(np.int)
print()
print("F1 Prediction ",F1_Metric_Score(Target_Val,Pre_val_lstm1))

del Val_keras,DF_Validation,M_Val
gc.collect()

print("Save Prediction")
Submit_Out['Lstm1']=Pre_val_lstm1

DF_Test= pq.read_table('DF_Test_Processing.parquet')
DF_Test=DF_Test.to_pandas()
M_Test= pq.read_table('x_test.parquet')
M_Test=M_Test.to_pandas().values


Test_keras=get_keras_Lstm1(DF_Test,M_Test)

Pred_test_lstm1=lstm1_nn.predict(Test_keras,batch_size=512, verbose=1)
Pred_test_lstm1= (np.array(Pred_test_lstm1) > 0.35).astype(np.int)


test= pq.read_table('DF_Test.parquet')
test=test.to_pandas()

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": Pred_test_lstm1.ravel()})
submit_df.to_csv("submission_lstm1.csv", index=False) 
print("Finish")
del Pre_val_lstm1,Pred_test_lstm1,M_test,submit_df,Test_keras,lstm1_nn,embedding_Glove,DF_Test,M_Test
gc.collect()
########################################### model 4 ###############################

print("Model Ht-att")
print("Load Embeddings")
embedding_matrix1=np.load('embedding_matrix1.npy')
embedding_matrix2=np.load('embedding_matrix2.npy')
print('Shape of Embedding1:', embedding_matrix1.shape)
print('Shape of Embedding 2:', embedding_matrix2.shape)

print("Normalization Embeddings")
embedding_matrix1=(embedding_matrix1-embedding_matrix1.mean(1).reshape(-1,1))
embedding_matrix2=(embedding_matrix2-embedding_matrix2.mean(1).reshape(-1,1))

print('Shape of Embedding1:', embedding_matrix1.shape)
print('Shape of Embedding 2:', embedding_matrix2.shape)


M_train=np.load('M_train.npy')
M_Val=np.load('M_Val.npy')
print('Shape of data tensor Matrix Train:', M_train.shape)
print('Shape of data tensor Matrix Validation:', M_Val.shape)

MAX_SENT_LEN=40

words_encoder_with_attention1 = Sequential([
    Embedding(Max_Emb, 300, weights=[embedding_matrix1], 
               input_length=MAX_SENT_LEN,trainable=False),
               Dropout(0.4),
    Bidirectional(CuDNNLSTM(64, return_sequences=True)),
    AttentionWithContext()])

words_encoder_with_attention2 = Sequential([
    Embedding(Max_Emb, 300, weights=[embedding_matrix2], 
               input_length=MAX_SENT_LEN,trainable=False),
               Dropout(0.4),
    Bidirectional(CuDNNLSTM(40, return_sequences=True)),
    AttentionWithContext()])


#Change GRU --> LSTM
doc_input_layer = Input(shape=(1,MAX_SENT_LEN), dtype='int32')
word_encoder_for_each_sentence1 = TimeDistributed(words_encoder_with_attention1)(doc_input_layer)
word_encoder_for_each_sentence2 = TimeDistributed(words_encoder_with_attention2)(doc_input_layer)

gru_layer1 = Bidirectional(CuDNNLSTM(40, return_sequences=True))(word_encoder_for_each_sentence1)
sentence_attention_layer1 = AttentionWithContext()(gru_layer1)
gru_layer2 = Bidirectional(CuDNNLSTM(40, return_sequences=True))(word_encoder_for_each_sentence2)
sentence_attention_layer2 = AttentionWithContext()(gru_layer2)
sentence_attention_layer=concatenate([sentence_attention_layer1,sentence_attention_layer2])
sentence_attention_layer=Dropout(0.5)(sentence_attention_layer)
doc_output_layer = Dense(1, activation='sigmoid')(sentence_attention_layer)
sentence_encoder_with_attention = Model(doc_input_layer, doc_output_layer)
sentence_encoder_with_attention.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['acc'])



dmtrain=M_train.shape[0]
dmval=M_Val.shape[0]

M_train=M_train.reshape((dmtrain,1,40))
M_Val=M_Val.reshape((dmval,1,40))

print("Estimation Model Ht att")
sentence_encoder_with_attention.fit(M_train,Target_Train ,validation_data=(M_Val, Target_Val),
                                    epochs=1, batch_size=256, shuffle=True,verbose=2)

y_pred_val=sentence_encoder_with_attention.predict(M_Val,batch_size=1024, verbose=0)
y_pred_val = (np.array(y_pred_val) > 0.3).astype(np.int)
print()
print("F1 Prediction ",F1_Metric_Score(Target_Val,y_pred_val))
#print(sentence_encoder_with_attention.summary())

del M_train,M_Val
gc.collect()

Submit_Out['htatt']=y_pred_val.ravel()

M_test= pq.read_table('x_test.parquet')
M_test=M_test.to_pandas().values



Dim=M_test.shape[0]
x_test2=M_test.reshape((Dim,1,40))
print("Dim Text:",x_test2.shape)
print("Predicciont Model ht-att")
y_pred=sentence_encoder_with_attention.predict(x_test2,batch_size=1024, verbose=0)
y_pred = (np.array(y_pred) > 0.3).astype(np.int)


test= pq.read_table('DF_Test.parquet')
test=test.to_pandas()

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_pred.ravel()})
submit_df.to_csv("submission_hattt.csv", index=False) 

del test,y_pred,y_pred_val,submit_df,embedding_matrix1,embedding_matrix2
del x_test2,M_test
gc.collect()
##############################

print("Last Model LSTM 2")
print("Embedding")
#print("Size Embedding:",embedding_Glove)
#embedding_matrix1=np.load('embedding_matrix1.npy')
#embedding_matrix3=np.load('embedding_matrix3.npy')
#embedding_matrix4=np.load('embedding_matrixG.npy')
#embedding_matrix2=np.load('embedding_matrix2.npy')

#embedding_matrix1=(embedding_matrix1-embedding_matrix1.mean(1).reshape(-1,1))
#embedding_matrix3=(embedding_matrix3-embedding_matrix3.mean(1).reshape(-1,1))
#embedding_matrix4=(embedding_matrix4-embedding_matrix4.mean(1).reshape(-1,1))
#embedding_matrix2=(embedding_matrix2-embedding_matrix2.mean(1).reshape(-1,1))

#print('Shape of Embedding1:', embedding_matrix1.shape)
#print('Shape of Embedding 2:', embedding_matrix2.shape)
#print('Shape of Embedding 3:', embedding_matrix4.shape)
#print("Size Embedding:",embedding_Glove)
#Embedding_General=np.mean([embedding_matrix2,embedding_matrix4],axis=1)

#del embedding_matrix2,embedding_matrix4
#gc.collect()

#M_train=np.load('M_train.npy')
#M_Val=np.load('M_Val.npy')
#print('Shape of data tensor Matrix Train:', M_train.shape)
#print('Shape of data tensor Matrix Validation:', M_Val.shape)


#batch_size = 128

#def batch_train_gen(df_train,df_target):
#    n_batches = math.ceil(df_train.shape[0] / batch_size)
#    indices = np.arange(df_train.shape[0])
#    np.random.shuffle(indices)

#    while True: 
        #data = DF_Train[indices]
#        MT =df_train[indices]
#        labels =df_target.iloc[indices].values
#        for i in range(n_batches):
#            out_arr = MT[i*batch_size:(i+1)*batch_size,:]
#            out_target=labels[i*batch_size:(i+1)*batch_size]
        
#            yield out_arr, out_target
            
#clr = CyclicLR(base_lr=0.001, max_lr=0.002,
#               step_size=500., mode='exp_range')


#K.clear_session()
#start_time = time.time()
#print("Model 1")    
#model = Sequential()
#model.add(Embedding(Max_Emb, 600,weights=[Embedding_General],trainable=False,name='Embedding'))
#model.add(Dropout(0.5))
#model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True,input_shape=(40,600),name='Layer_1')))
#model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True),name='Layer_2'))
#model.add(Attention(maxlen))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation="sigmoid"))
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['acc'])


#model.fit_generator(batch_train_gen(M_train,Target_Train), epochs=1,
#                    steps_per_epoch=1000,
#                     validation_data=(M_Val,Target_Val),callbacks=[clr], verbose=2)


#y_pred_val=model.predict(M_Val,batch_size=512, verbose=1)
#y_pred_val = (np.array(y_pred_val) > 0.3).astype(np.int)

#print()
#print("F1 Prediction ",F1_Metric_Score(Target_Val,y_pred_val))

#Submit_Out['lstm2']=y_pred_val.ravel()

#x_test= pq.read_table('x_test.parquet')
#x_test=x_test.to_pandas().values


#M_test= pq.read_table('x_test.parquet')
#M_test=M_test.to_pandas().values


#y_pred_test=model.predict(M_test,batch_size=512, verbose=1)
#y_pred_test = (np.array(y_pred_test) > 0.3).astype(np.int)

#test= pq.read_table('test.parquet')
#test=test.to_pandas()

#submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_pred_test.ravel()})
#submit_df.to_csv("submission_lstm2.csv", index=False) 

################### Ensemble #####################
subPrediction=pd.DataFrame(Submit_Out)

Output1=pd.read_csv("submission_Ridge.csv",usecols=["prediction"]) 
Output1['predict_Conv']=pd.read_csv("submission_conv1.csv",usecols=["prediction"]) 
Output1['predict_Lstm1']=pd.read_csv("submission_lstm1.csv", usecols=["prediction"]) 
Output1['predict_hattt']=pd.read_csv("submission_hattt.csv", usecols=["prediction"]) 
#Output1['predict_Lstm2']=pd.read_csv("submission_lstm2.csv",usecols=["prediction"] ) 

y_pred=subPrediction.mean(axis=1)
y_pred = (np.array(y_pred) > 0.5).astype(np.int)

print ('ENSEMBLE MEAN SCORE :',F1_Metric_Score(Target_Val,y_pred))

clf=LogisticRegression()
clf.fit(subPrediction.values,Target_Val)
Out_Pred=clf.predict(Output1.values)
Out_Pred = (np.array(Out_Pred) > 0.5).astype(np.int)


y_pred=clf.predict(subPrediction.values)
y_pred = (np.array(y_pred) > 0.5).astype(np.int)

print ('ENSEMBLE MEAN SCORE Regression :',F1_Metric_Score(Target_Val,y_pred))

test= pq.read_table('DF_Test.parquet')
test=test.to_pandas()

submit_df = pd.DataFrame({"qid": test["qid"], "prediction": Out_Pred})
submit_df.to_csv("submission.csv", index=False) 


