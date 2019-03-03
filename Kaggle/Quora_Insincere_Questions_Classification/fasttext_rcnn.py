import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import string

from collections import defaultdict
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack,vstack
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


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
#print re.sub(r'['+chars+']', '',my_str)
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


def Sample_2(DF):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
    L1,_=sss.split(DF.question_text,DF.target)
    return DF.iloc[L1[0],:]   
    

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
    
class DataFrameSelect(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]

print("Load the Data !!!!")
print('*'*50)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

print("Processing Question !!!!")
print('*'*50)

train.loc[:,'question_text2']=parallelize_dataframe2(train.question_text,clean_str2,Parallelize_function)
test.loc[:,'question_text2']=parallelize_dataframe2(test.question_text,clean_str2,Parallelize_function)

        
train_stage1=(train.pipe(Add_FirstWords)
       .assign(Lenght_by_Words=lambda X:X['question_text'].apply(lambda x: len(str(x).split(" "))),
             Lenght_by_Char=lambda X:X['question_text'].apply(lambda x: len(x)),
             Max_Lenght_String=lambda X:X['question_text'].apply(lambda x: Max_LenghtWords(x) if len(x)>1  else 0))
      .pipe(Ratio_Words)
    .pipe(Add_Clean_String)
    .assign(Clean_String_Length=lambda X:X['Clean_String'].apply(lambda x: len(x.split(' '))))
    .drop(labels=['Clean_String'],axis=1)
    .pipe(Ratio_Words_PostCleaning).drop(labels=['qid'],axis=1)
            .assign(target=lambda X:X['target'].astype(np.uint8),
                    What=lambda X:X['What'].astype(np.uint8),
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
del train
gc.collect()    
    
test_stage1=(test.pipe(Add_FirstWords)
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

del test
gc.collect()

print("Train shape : ",train_stage1.shape)
print("Test shape : ",test_stage1.shape)

print("Words Processing")
print('*'*50)

def word_count(text, dc):
    text = set( text.split(' ') ) 
    for w in text:
        dc[w]+=1
def remove_low_freq(text, dc):
    return ' '.join( [w for w in text.split() if w in dc] )


labels_dict = dict()
#df_train = load_data()
print (train_stage1.shape)

min_df_one=5

word_count_dict_one = defaultdict(np.uint32)
for feat in ['question_text2']:
    train_stage1[feat].apply(lambda x : word_count(x, word_count_dict_one) )
rare_words = [key for key in word_count_dict_one if  word_count_dict_one[key]<min_df_one ]

for key in rare_words :
    word_count_dict_one.pop(key, None)

for feat in ['question_text2']:
    train_stage1[feat] = train_stage1[feat].apply( lambda x : remove_low_freq(x, word_count_dict_one) )
word_count_dict_one=dict(word_count_dict_one)
vocabulary_one = word_count_dict_one.copy()
print("Ready!!!")
vocabulary_one = word_count_dict_one.copy()

for dc in [vocabulary_one]:
    cpt=0
    for key in dc:
        dc[key]=cpt
        cpt+=1

MAX_ITEM_DESC_SEQ=40

def preprocess_keras(text):
    return [ vocabulary_one[w] for w in (text.split())[:MAX_ITEM_DESC_SEQ] ]
def preprocess_keras_df(df):
    return df.apply( preprocess_keras )
print("Sequens...!!!")
print('*'*50)

train_stage1['seq_name'] = parallelize_dataframe(train_stage1['question_text2'], preprocess_keras_df)
test_stage1['question_text2']= test_stage1['question_text2'].apply( lambda x : remove_low_freq(x, vocabulary_one) )
test_stage1['seq_name'] = parallelize_dataframe(test_stage1['question_text2'], preprocess_keras_df)


#ngram_range = 2
#max_features = 40000
#maxlenE = 400
MAX_TEXT = len(vocabulary_one)+ 1
batch_size = 128
embedding_dims = 200
epochs = 5


numeric_features1 = ['Lenght_by_Words','Lenght_by_Char',
                    'Max_Lenght_String','Clean_String_Length']
numeric_features2 = ['Ratio_Words_description','Ratio_Words_PostCleaning']

categorical_features = ['What','How','Why','Is','Can','Which','Do','If','Are', 'Who']

numeric_transformer1 = Pipeline(steps=[
    ('Log1p',FunctionTransformer(np.log1p, validate=True)),
    ('scaler', StandardScaler())])

numeric_transformer2 = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline([
    ('Selection_Cat', DataFrameSelect(attribute_names=categorical_features))])



preprocessor_num1 = ColumnTransformer(transformers=[('num1', numeric_transformer1, numeric_features1)])
preprocessor_num2 = ColumnTransformer(transformers=[('num2', numeric_transformer2, numeric_features1)])
preprocessor_cat2 = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])

preprocessor_num1.fit(train_stage1) 
preprocessor_num2.fit(train_stage1) 
preprocessor_cat2.fit(train_stage1) 


def get_keras_sparse(df):
    X = {'text_data': sequence.pad_sequences(df['seq_name'], maxlen=30),
        'lenghts': preprocessor_num1.transform(df),
        'rations': preprocessor_num2.transform(df),
        'cats': preprocessor_cat2.transform(df)}
    return X
    

train_keras = get_keras_sparse(train_stage1)

from keras.models import Sequential
from keras.layers import Dense,PReLU
from keras.layers import Input, Dropout, concatenate, Activation, BatchNormalization
from keras.initializers import he_uniform
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Flatten



def fasttext_model():                                             
    text_data = Input( shape=[train_keras["text_data"].shape[1]],dtype = 'float32',name='text_data')  
    lenghts = Input(shape=[train_keras["lenghts"].shape[1]], name="lenghts")
    rations = Input(shape=[train_keras["rations"].shape[1]], name="rations")
    cats = Input(shape=[train_keras["cats"].shape[1]], name="cats")
    embedding = Embedding(MAX_TEXT,embedding_dims,input_length=30)
    emb_question = embedding (text_data)
    val=10
    emb_category1 = Flatten() ( Embedding(3, val)(cats) )
    emb_question = GlobalAveragePooling1D( name='output_question_max' )(emb_question)
    x = concatenate([emb_question,emb_category1,lenghts,rations])
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)

    x = Dense(1, activation="sigmoid") (x)
    model = Model([text_data,lenghts,rations,cats],x)
    #optimizer = Adam(.0011)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model

        
    

fasttext_model = fasttext_model()


print("Fitting FASTTEXT NN model ...")

fasttext_model.fit(train_keras, train_stage1.target,
          batch_size=batch_size,
          epochs=epochs,verbose=True)
          
#for ep in range(epochs):
#    BATCH_SIZE = int(BATCH_SIZE*2)
#    fasttext_model.fit(  train_keras, (df_train.price.values-mean_price), 
#                      batch_size=BATCH_SIZE, epochs=1, verbose=10 )


    
    
    
test_keras=get_keras_sparse(test_stage1)
all_preds=fasttext_model.predict(test_keras,batch_size=500)

test_df = pd.read_csv("../input/test.csv")

y_te = (np.array(all_preds) > 0.3).astype(np.int)


print('Finish!')
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te.ravel()})
submit_df.to_csv("submission.csv", index=False)
