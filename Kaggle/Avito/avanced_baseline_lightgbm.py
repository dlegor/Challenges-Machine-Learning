import numpy as np 
import pandas as pd 

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

from nltk.corpus import stopwords
import lightgbm as gbm
import gc
import time
import nltk
import re,string
import operator
import functools
#from pymystem3 import Mystem
from multiprocessing import Pool
from functools import partial
from nltk.stem.snowball import SnowballStemmer

#Create lemmatizer and stopwords list
#mystem = Mystem() 
russian_stopwords = stopwords.words("russian")
cores=4
max_text_length=100
tr =pd.read_csv("../input/avito-demand-prediction/train.csv",parse_dates=['activation_date']) 
te =pd.read_csv("../input/avito-demand-prediction/test.csv",parse_dates=['activation_date'])
Extra_Features=pd.read_pickle("../input/end-to-end-lightgbm-2/aggregated_features.pkl")

print("Preprocessing")

tri=tr.shape[0]
y = tr.deal_probability.copy()

lb=LabelEncoder()

def Concat_Text(df,Columns,Name):
    df=df.copy()
    df.loc[:,Columns].fillna(" ",inplace=True)
    df[Name]=df[Columns[0]].astype('str')
    for col in Columns[1:]:
        df[Name]=df[Name]+' '+df[col].astype('str')
    return df
    
def Count_Missing(df,Columns):
    df=df.copy()
    df['no_na']=df.isnull().sum(axis=1)
    for col in Columns:
        df['no_na_'+col]=df[col].isnull().apply( lambda x: 1 if x==True else 0)
    return df
    
def Ratio_Words(df):
    df=df.copy()
    df['description']=df['description'].astype('str')
    df['num_words_description']=df['description'].apply(lambda x:len(x.split()))
    df['num_unique_description']=df['description'].apply(lambda x: len(set(w for w in x.split())))
    df['Ratio_Words_description']=(df['num_unique_description']/df['num_words_description'])* 100
    df['title']=df['title'].astype('str')
    df['num_words_title']=df['title'].apply(lambda x:len(x.split()))
    df['num_unique_title']=df['title'].apply(lambda x: len(set( w for w in x.split())))
    df['Ratio_Words_title']=(df['num_unique_title']/df['num_words_title'])*100
    return df
    
def Lenght_Columns(df,Columns):
    df=df.copy()
    Columns_Len=['len_'+s for s in Columns]
    for col in Columns:
        df[col]=df[col].astype('str')
    for x,y in zip(Columns,Columns_Len):
        df[y]=df[x].apply(len)
    return df    
    
def Mean_Category(df,Cat_Features):
    df=df.copy()
    df['deal_probability']=y
    lb=LabelEncoder()
    Cat_dict={}
    df.loc[:,Cat_Features].fillna(0.0,inplace=True)
    for col in Cat_Features:
        Cat_dict[col]=df[col].unique()
    
    for col in Cat_Features:
        df[col]=lb.fit_transform(df[col])
    mean_dc=dict()
    for feat in Cat_Features:
        mean_dc[feat] = df.groupby(feat)['deal_probability'].mean().astype(np.float32)
        mean_dc[feat] /= np.max(mean_dc[feat])
        df['mean_deal_'+feat] = df[feat].map(mean_dc[feat]).astype(np.float32)
        df['mean_deal_'+feat].fillna( mean_dc[feat].mean(), inplace=True  )
    return df

def Trun_Lop1(x):
    return np.min([np.log1p(x),17.5])    

def number_count(s):
    return sum(1 for c in re.findall('\d',s))


def contains_number(s):
    if len(re.findall('\d',s))>1:
        return 1
    return     0

def uppercase_count(s):
    return sum(1 for c in s if c.isupper())

def contain_special_char(s):
    special_chars = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ '''
    if sum(1 for c in s if c in special_chars) >= 1:
        return 1
    return 0

def Cap_TextE(s):
    s=str(s)
    return len(re.findall(r"[A-ZА-Я]",s))

def Cap_TextR(s):
    s=str(s)
    return len(re.findall(r"[А-Я]",s))

def Cap_TextW(s):
    s=str(s)
    return len(re.findall(r"\W",s))

def Cap_Textw(s):
    s=str(s)
    return len(re.findall(r"\w",s))

def Len_Character(String):
    S=str(String)
    return len(list(S))

p=re.compile('\s')
def Count_NewLine(String):
    S=str(String)
    return len([s for s in p.findall(S) if s!=' '])

def punctuation_count(s):
    return sum(1 for c in s if c in string.punctuation)

def digits_count(s):
    return sum(1 for c in s if c in string.digits)

def Started_Error(s):
    if len(re.findall('r[^a-я0-9]',s))>2:
        return 1
    return     0

def Has_Word_not_in_Russian(s):
    if len(set(re.findall(r'[a-zA-Z]{2,15}',s)))>0:
        return 1
    return     0

def Word_not_in_Russian(s):
    return " ".join(re.findall(r'[a-zA-Z0-9]{2,15}',s))
#Function for Parallelization

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

#Main Function
def Clean_Basic(text):
    try:
        pattern1 = re.compile(r'\n\r\f\v')
        pattern2 = re.compile(r'\s+')
        punt = re.compile('[%s]' % re.escape(string.punctuation))
        #text=str(text).lower()
        text = ' '.join( [w for w in str(text).lower().split()[:max_text_length]] )
        text=re.sub(u"\xa0",u"",text)
        text = text.replace(' -', '').replace('- ', '').replace(' - ', '').replace('-','')
        text = text.replace('—','')
        text = text.replace(':)', 'улыбка').replace('(:', 'улыбка').replace(':-)', 'улыбка')
        text = text.replace('⇘', '').replace('⇓', '').replace('⇙', '').replace('⇒','')
        text = text.replace('ﻩ', '').replace('º','')
        text = text.replace('ᗒᗣᗕ', '').replace('ღ','').replace('ლ','').replace('ஜ','')
        text=re.sub(u"\/",u"",text)
        text=re.sub(u"\\n",u" ",text)
        text=re.sub(pattern1, '', text)
        text=re.sub(punt,'',text)
        text=re.sub(u"[a-z0-9]",u"",text) 
        text = re.sub(pattern2,' ',text)
        text=re.sub(r'\b(\d+)([a-z]+)\b', r'\1 \2',text)
        text=re.sub(r'\b([a-z]+)(\d+)\b', r'\1 \2',text)
        #text=''.join(text)
    except:
        text = np.NaN    
    
    return text


#Paralellization
def clean_str_df(df):
    return df.apply( lambda s : Clean_Basic(str(s)))

#Lemmatization/stem

stemmer = SnowballStemmer("russian") 

def preprocess_text(text):
    try:
        
        text = text.lower()
        tokens = [stemmer.stem(token) for token in text.split() if token not in russian_stopwords and token !=' ']
        #tokens = [token for token in tokens if token not in russian_stopwords and token !=' ']
        L = ' '.join(tokens)
        
    except:
        np.NaN
    return L                

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


cat_features =['region','city','param_1','param_2','param_3','user_id','user_type','mday','mon','week','wday','Region_City','Region_User_type','Parent_User_Type']
List_Var=['item_id', 'activation_date', 'image',]
#'deal_probability']

tr_te=(tr.drop(labels=["deal_probability"],axis=1).append(te)
     .merge(Extra_Features,on='user_id',how='left')
     .pipe(Count_Missing,['image','image_top_1','description','param_1','param_2','param_3','price'])
     .pipe(Concat_Text,['region','city'],'Region_City')
     .pipe(Concat_Text,['param_1','param_2','param_3'],'Params')
     .pipe(Concat_Text,['parent_category_name','category_name'],'Parent_Category_Type')
     .pipe(Concat_Text,['image_top_1','city'],'Image_city')
     .pipe(Concat_Text,['user_id','image_top_1'],'User_Image_top')
     .pipe(Ratio_Words)
     .pipe(Lenght_Columns,['title','description'])
     .assign(titl_numb = lambda x: x.title.apply(number_count),
             desc_numb = lambda x: x.description.apply(number_count),
             titl_withNum = lambda x: x.title.apply(contains_number),
             desc_withNum = lambda x: x.description.apply(contains_number),
             titl_uppercase = lambda x: x.title.apply(uppercase_count),
             desc_uppercase = lambda x: x.description.apply(uppercase_count),
             titl_spec_char = lambda x: x.title.apply(contain_special_char),
             desc_spec_char = lambda x: x.description.apply(contain_special_char),
             titl_CapE = lambda x: x.title.apply(Cap_TextE),
             desc_CapE = lambda x: x.description.apply(Cap_TextE),
             titl_CapR = lambda x: x.title.apply(Cap_TextR),
             desc_CapR = lambda x: x.description.apply(Cap_TextR),
             titl_CapW = lambda x: x.title.apply(Cap_TextW),
             desc_CapW = lambda x: x.description.apply(Cap_TextW),
             titl_Capw = lambda x: x.title.apply(Cap_Textw),
             desc_Capw = lambda x: x.description.apply(Cap_Textw),
             titl_LCap = lambda x: x.title.apply(Len_Character),
             desc_LCap = lambda x: x.description.apply(Len_Character),
             desc_nline = lambda x: x.description.apply(Count_NewLine),
             titl_puntuation = lambda x: x.title.apply(punctuation_count),
             desc_puntuation = lambda x: x.description.apply(punctuation_count),
             titl_digits = lambda x: x.title.apply(digits_count),
             desc_digits = lambda x: x.description.apply(digits_count),
             desc_Started_Error = lambda x: x.description.apply(Started_Error),
             titl_has_Word_not_in_Russian = lambda x: x.title.apply(Has_Word_not_in_Russian),
             desc_has_Word_not_in_Russian = lambda x: x.description.apply(Has_Word_not_in_Russian),
             category_name=lambda x: pd.Categorical(x['category_name']).codes,
             parent_category_name=lambda x:pd.Categorical(x['parent_category_name']).codes,
             region=lambda x:pd.Categorical(x['region']).codes,
             city=lambda x:lb.fit_transform(x['city'].astype('str')),
             user_type=lambda x:pd.Categorical(x['user_type']).codes,
             param_1=lambda x:lb.fit_transform(x['param_1'].fillna('-1').astype('str')),
             param_2=lambda x:lb.fit_transform(x['param_2'].fillna('-1').astype('str')),
             param_3=lambda x:lb.fit_transform(x['param_3'].fillna('-1').astype('str')),
             user_id=lambda x:lb.fit_transform(x['user_id'].astype('str')),
             Log_price=lambda x: x['price'].fillna(0).apply(Trun_Lop1),
             Tail_price=lambda x:x['price'].fillna(0).apply(lambda z: 0 if np.log1p(z)<17.5 else 1),
             mon=lambda x: pd.to_datetime(x['activation_date']).dt.month,
             mday=lambda x: pd.to_datetime(x['activation_date']).dt.day,
             week=lambda x: pd.to_datetime(x['activation_date']).dt.week,
             wday=lambda x:pd.to_datetime(x['activation_date']).dt.dayofweek,
             title=lambda x: x['title'].fillna('unknown_title').astype('str'),
             description=lambda x: x['description'].fillna('unknown_description').astype('str'),
             image_top_1=lambda x:x['image_top_1'].fillna(0),
             Region_City=lambda x:lb.fit_transform(x['Region_City'].fillna('-1').astype('str')),
             Parent_Category_Type=lambda x:lb.fit_transform(x['Parent_Category_Type'].fillna('-1').astype('str')),
             Image_city=lambda x:lb.fit_transform(x['Image_city'].fillna('-1').astype('str')),
             User_Image_top=lambda x:lb.fit_transform(x['User_Image_top'].fillna('-1').astype('str')))
            #.pipe(Mean_Category,cat_features)             
            .drop(labels=List_Var,axis=1))
            
            
tr_te['avg_days_up_user'].fillna(0,inplace=True) 
tr_te['avg_times_up_user'].fillna(0,inplace=True)
tr_te['Sum_D1'].fillna(0,inplace=True)
tr_te['N_Item_id'].fillna(0,inplace=True)

Cat_Features_Large=['user_id','city','Region_City','image_top_1','Region_City','Image_city','User_Image_top','param_3']
Cat_Features_ohe=['region','parent_category_name','category_name','param_1','param_2','mon','mday','week','wday']

print("Aggregation User_id")

L1=tr_te[['user_id','price']].groupby('user_id',as_index=False).agg(['mean','max','min','sem','std']).reset_index()
Columns=['user_id']+[a[0]+'_'+a[1] for a in L1.columns if a[1] !='']
L1.columns=Columns

L2=tr_te[['user_id','image_top_1']].groupby('user_id',as_index=False).agg(['max','min','size']).reset_index()
Columns=['user_id']+[a[0]+'_'+a[1] for a in L2.columns if a[1] !='']
L2.columns=Columns

L3=tr_te[['user_id','item_seq_number']].groupby('user_id',as_index=False).agg(['max','min','std']).reset_index()
Columns=['user_id']+[a[0]+'_'+a[1] for a in L3.columns if a[1] !='']
L3.columns=Columns

L_users_id=L1.merge(L2,on='user_id',how='left').merge(L3,on='user_id',how='left')
L_users_id.fillna(0,inplace=True)

tr_te=tr_te.merge(L_users_id,on='user_id',how='left')

del L1,L2,L3,tr,te
gc.collect()        

print("Texts")

tr_te.loc[:,'description']=parallelize_dataframe(tr_te['description'],clean_str_df)
#L2=parallelize_dataframe2(L,preprocess_text,Parallelize_function)
tr_te.loc[:,'description']=parallelize_dataframe2(tr_te.description,preprocess_text,Parallelize_function)


print("Processing Text")

def get_col(col_name): return lambda x: x[col_name]

vectorizer = FeatureUnion([
        ('title',CountVectorizer(
            lowercase=True,
            stop_words=russian_stopwords,
            dtype=np.uint8,
            min_df=10,
            max_features=2500,
            binary=True,
            preprocessor=get_col('title'))),
        ('description',TfidfVectorizer(
            ngram_range=(1,2),
            stop_words=russian_stopwords,
            min_df=10,
            token_pattern= r'\w{1,}',
            max_features=15000,
            dtype=np.uint8,
            preprocessor=get_col('description'))),
            ('Params',CountVectorizer(
            ngram_range=(2,3),    
            lowercase=True,
            dtype=np.uint8,
            min_df=25, 
            binary=True,
            max_features=30,
            preprocessor=get_col('Params')))])


Sparse=vectorizer.fit_transform(tr_te[['title','description','Params']].to_dict('records'))
tr_te.drop(labels=['title','description','Params'],axis=1,inplace=True)
gc.collect()

print("Categories Features")

def Get_Hash(df,ndim=100):
    df=df.copy()
    for i in range(df.shape[1]):
        df.iloc[:,i]=df.iloc[:,i].astype('str')
    h = FeatureHasher(n_features=ndim,input_type="string")
    return  h.transform(df.values)

Features_Hash=Get_Hash(tr_te[Cat_Features_Large],ndim=3000)
Features_OHE=OneHotEncoder().fit_transform(tr_te[Cat_Features_ohe])

tr_te.drop(labels=Cat_Features_Large,inplace=True,axis=1)
tr_te.drop(labels=Cat_Features_ohe,inplace=True,axis=1)
print(tr_te.shape)

print("General Data")
data  = hstack((tr_te.values,Features_Hash,Features_OHE,Sparse)).tocsr()

print(data.shape)
del tr_te,Sparse
gc.collect()

dtest=data[tri:]
X=data[:tri]
print(X.shape)
del data
gc.collect()

Dparam = {'objective' : 'huber',
          'boosting_type': 'gbdt',
          'metric' : 'rmse',
          'nthread' : 8,
          #'max_bin':350,
          'shrinkage_rate':0.05,
          'max_depth':18,
          'min_child_weight': 11,
          'bagging_fraction':0.75,
          'feature_fraction':0.6,
          'lambda_l1':2,
          'lambda_l2':1}
          #'num_leaves':31}        

print("Training Model")

def RMSE(L,L1):
    return np.sqrt(mse(L,L1))
    
folds = KFold(n_splits=5, shuffle=True, random_state=50001)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(dtest.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    dtrain =gbm.Dataset(X[trn_idx], y.iloc[trn_idx])
    dval =gbm.Dataset(X[val_idx], y.iloc[val_idx])
    m_gbm=gbm.train(params=Dparam,train_set=dtrain,num_boost_round=6000,verbose_eval=1000,valid_sets=[dtrain,dval],valid_names=['train','valid'])
    oof_preds[val_idx] = m_gbm.predict(X[val_idx])
    sub_preds += m_gbm.predict(dtest) / folds.n_splits
    print('Fold %2d rmse : %.6f' % (n_fold + 1, RMSE(y.iloc[val_idx],oof_preds[val_idx])))
    del dtrain,dval
    gc.collect()
    
print('Full RMSE score %.6f' % RMSE(y, oof_preds))   

oof_Train = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols=['item_id','deal_probability'])
oof_Train.loc[:,'deal_prob_gbm_rmse']=oof_preds
oof_Train.to_csv("Mod_1_gbm_cdesc_rmse_oof_train.csv", index=False)

sub_preds[sub_preds<0]=0
sub_preds[sub_preds>1]=1

print("Output Model")

Submission=pd.read_csv("../input/avito-demand-prediction/sample_submission.csv")
Submission.loc[:,'deal_probability']=sub_preds
Submission[['item_id','deal_probability']].to_csv("Mod_1_gbm_cdesc_rmse.csv", index=False)
