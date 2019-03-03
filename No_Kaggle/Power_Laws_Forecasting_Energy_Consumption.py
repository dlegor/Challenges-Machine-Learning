
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gc

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


from dateutil.parser import parse
from datetime import date, timedelta


# In[4]:


plt.rcParams['figure.figsize'] = 20, 10


# In[5]:


#df_train15=pd.read_csv("Data_processing/Train_all_data_freq_15.csv.gz",compression='gzip',parse_dates=['Timestamp'])
#df_train15.shape


# In[7]:


#Variables=['obs_id', 'SiteId', 'Timestamp', 'ForecastId', 'Value','Minute','Hour', 'Day', 'DayofWeek','Holiday_Type_1', 'Holiday_Type_2']


# In[8]:


#Aux_Data=df_train15[Variables]
#L=Aux_Data[['SiteId','Timestamp','ForecastId']].groupby(['SiteId','ForecastId']).agg({'Timestamp':[max]}).reset_index()
#L.columns=['SiteId','ForecastId','Timestamp_Max']
#Aux_Data=Aux_Data.merge(L,on=['SiteId','ForecastId'],how='left')
#del(L,df_train15)


# In[9]:


#Aux_Data.head()


# In[12]:


def get_Index_Bases(n_momentos):
    L1=Aux_Data.Timestamp_Max-((n_momentos+1)*16)*timedelta(minutes=15)
    L2=Aux_Data.Timestamp_Max-((n_momentos)*16)*timedelta(minutes=15)
    Indices_Base=Aux_Data[(Aux_Data.Timestamp>=L1) & (Aux_Data.Timestamp<L2)]['obs_id']
    Indices_GrBy=Aux_Data[Aux_Data.Timestamp<L1]['obs_id']
    return Indices_Base,Indices_GrBy,L2


# In[79]:


def get_Site_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby('SiteId',as_index=False)['Value']                                                                     .agg({'SiteId_min{}'.format(n_momentos): 'min',
                                                                           'SiteId_mean{}'.format(n_momentos): 'mean',
                                                                           'SiteId_median{}'.format(n_momentos): 'median',
                                                                           'SiteId_max{}'.format(n_momentos): 'max',
                                                                           'SiteId_count{}'.format(n_momentos): 'count',
                                                                           'SiteId_std{}'.format(n_momentos): 'std',
                                                                           'SiteId_skew{}'.format(n_momentos): 'skew'}) 
    COLUMNAS=['obs_id']+result.columns.tolist()[1:]   
    return temp1.merge(result,on=['SiteId'],how='left').fillna(0)[COLUMNAS]
    


# In[80]:


def get_ForecastId_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby('ForecastId',as_index=False)['Value']                                                                     .agg({'ForecastId_min{}'.format(n_momentos): 'min',
                                                                           'ForecastId_mean{}'.format(n_momentos): 'mean',
                                                                           'ForecastId_median{}'.format(n_momentos): 'median',
                                                                           'ForecastId_max{}'.format(n_momentos): 'max',
                                                                           'ForecastId_count{}'.format(n_momentos): 'count',
                                                                           'ForecastId_std{}'.format(n_momentos): 'std',
                                                                           'ForecastId_skew{}'.format(n_momentos): 'skew'}) 
    COLUMNAS=['obs_id']+result.columns.tolist()[1:]    
    return temp1.merge(result,on=['ForecastId'],how='left').fillna(0)[COLUMNAS]


# In[82]:


def get_Site_ForecastId_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','ForecastId'],as_index=False)['Value']                                                                     .agg({'Site_ForecastId_min{}'.format(n_momentos): 'min',
                                                                           'Site_ForecastId_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_ForecastId_median{}'.format(n_momentos): 'median',
                                                                           'Site_ForecastId_max{}'.format(n_momentos): 'max',
                                                                           'Site_ForecastId_count{}'.format(n_momentos): 'count',
                                                                           'Site_ForecastId_std{}'.format(n_momentos): 'std',
                                                                           'Site_ForecastId_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[2:]    
    return temp1.merge(result,on=['SiteId','ForecastId'],how='left').fillna(0)[COLUMNAS]


# In[85]:


def get_Site_Minute_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','Minute','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','Minute','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','Minute'],as_index=False)['Value']                                                                     .agg({'Site_Minutes_min{}'.format(n_momentos): 'min',
                                                                           'Site_Minutes_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_Minutes_median{}'.format(n_momentos): 'median',
                                                                           'Site_Minutes_max{}'.format(n_momentos): 'max',
                                                                           'Site_Minutes_count{}'.format(n_momentos): 'count',
                                                                           'Site_Minutes_std{}'.format(n_momentos): 'std',
                                                                           'Site_Minutes_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[2:]    
    return temp1.merge(result,on=['SiteId','Minute'],how='left').fillna(0)[COLUMNAS]


# In[87]:


def get_Site_Hour_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','Hour','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','Hour','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','Hour'],as_index=False)['Value']                                                                     .agg({'Site_Hour_min{}'.format(n_momentos): 'min',
                                                                           'Site_Hour_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_Hour_median{}'.format(n_momentos): 'median',
                                                                           'Site_Hour_max{}'.format(n_momentos): 'max',
                                                                           'Site_Hour_count{}'.format(n_momentos): 'count',
                                                                           'Site_Hour_std{}'.format(n_momentos): 'std',
                                                                           'Site_Hour_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[2:]       
    return temp1.merge(result,on=['SiteId','Hour'],how='left').fillna(0)[COLUMNAS]


# In[89]:


def get_Site_Day_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','Day','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','Day','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','Day'],as_index=False)['Value']                                                                     .agg({'Site_Day_min{}'.format(n_momentos): 'min',
                                                                           'Site_Day_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_Day_median{}'.format(n_momentos): 'median',
                                                                           'Site_Day_max{}'.format(n_momentos): 'max',
                                                                           'Site_Day_count{}'.format(n_momentos): 'count',
                                                                           'Site_Day_std{}'.format(n_momentos): 'std',
                                                                           'Site_Day_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[2:]       
    return temp1.merge(result,on=['SiteId','Day'],how='left').fillna(0)[COLUMNAS]


# In[91]:


def get_Site_DayofWeek_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','DayofWeek','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','DayofWeek','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','DayofWeek'],as_index=False)['Value']                                                                     .agg({'Site_DayofWeek_min{}'.format(n_momentos): 'min',
                                                                           'Site_DayofWeek_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_DayofWeek_median{}'.format(n_momentos): 'median',
                                                                           'Site_DayofWeek_max{}'.format(n_momentos): 'max',
                                                                           'Site_DayofWeek_count{}'.format(n_momentos): 'count',
                                                                           'Site_DayofWeek_std{}'.format(n_momentos): 'std',
                                                                           'Site_DayofWeek_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[2:]    
    return temp1.merge(result,on=['SiteId','DayofWeek'],how='left').fillna(0)[COLUMNAS]


# In[93]:


def get_Site_ForecastId_Minute_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Minute','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Minute','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','ForecastId','Minute'],as_index=False)['Value']                                                                     .agg({'Site_ForecastId_Minute_min{}'.format(n_momentos): 'min',
                                                                           'Site_ForecastId_Minute_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_ForecastId_Minute_median{}'.format(n_momentos): 'median',
                                                                           'Site_ForecastId_Minute_max{}'.format(n_momentos): 'max',
                                                                           'Site_ForecastId_Minute_count{}'.format(n_momentos): 'count',
                                                                           'Site_ForecastId_Minute_std{}'.format(n_momentos): 'std',
                                                                           'Site_ForecastId_Minute_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[3:]    
    return temp1.merge(result,on=['SiteId','ForecastId','Minute'],how='left').fillna(0)[COLUMNAS]


# In[95]:


def get_Site_ForecastId_Hour_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Hour','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Hour','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','ForecastId','Hour'],as_index=False)['Value']                                                                     .agg({'Site_ForecastId_Hour_min{}'.format(n_momentos): 'min',
                                                                           'Site_ForecastId_Hour_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_ForecastId_Hour_median{}'.format(n_momentos): 'median',
                                                                           'Site_ForecastId_Hour_max{}'.format(n_momentos): 'max',
                                                                           'Site_ForecastId_Hour_count{}'.format(n_momentos): 'count',
                                                                           'Site_ForecastId_Hour_std{}'.format(n_momentos): 'std',
                                                                           'Site_ForecastId_Hour_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[3:]    
    return temp1.merge(result,on=['SiteId','ForecastId','Hour'],how='left').fillna(0)[COLUMNAS]


# In[97]:


def get_Site_ForecastId_Day_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Day','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Day','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','ForecastId','Day'],as_index=False)['Value']                                                                     .agg({'Site_ForecastId_Day_min{}'.format(n_momentos): 'min',
                                                                           'Site_ForecastId_Day_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_ForecastId_Day_median{}'.format(n_momentos): 'median',
                                                                           'Site_ForecastId_Day_max{}'.format(n_momentos): 'max',
                                                                           'Site_ForecastId_Day_count{}'.format(n_momentos): 'count',
                                                                           'Site_ForecastId_Day_std{}'.format(n_momentos): 'std',
                                                                           'Site_ForecastId_Day_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[3:]    
    return temp1.merge(result,on=['SiteId','ForecastId','Day'],how='left').fillna(0)[COLUMNAS]


# In[99]:


def get_Site_ForecastId_DayofWeek_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','DayofWeek','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','DayofWeek','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].groupby(['SiteId','ForecastId','DayofWeek'],as_index=False)['Value']                                                                     .agg({'Site_ForecastId_DayofWeek_min{}'.format(n_momentos): 'min',
                                                                           'Site_ForecastId_DayofWeek_mean{}'.format(n_momentos): 'mean',
                                                                           'Site_ForecastId_DayofWeek_median{}'.format(n_momentos): 'median',
                                                                           'Site_ForecastId_DayofWeek_max{}'.format(n_momentos): 'max',
                                                                           'Site_ForecastId_DayofWeek_count{}'.format(n_momentos): 'count',
                                                                           'Site_ForecastId_DayofWeek_std{}'.format(n_momentos): 'std',
                                                                           'Site_ForecastId_DayofWeek_skew{}'.format(n_momentos): 'skew'})
    COLUMNAS=['obs_id']+result.columns.tolist()[3:]    
    return temp1.merge(result,on=['SiteId','ForecastId','DayofWeek'],how='left').fillna(0)[COLUMNAS]


# In[101]:


def get_Site_Exp_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].copy()
    del(temp2)
    result.loc[:,'Diff_Momentos']=(Time_Temp-result.Timestamp).dt.total_seconds()/900
    result.loc[:,'weight'] = result['Diff_Momentos'].apply(lambda x: 0.985**x)
    result['Value']=result['Value']*result['weight']
    
    Temp3=result[['SiteId','Value','weight']].groupby('SiteId',as_index=False).agg({'Value':sum,'weight':sum})
    Temp3.columns=['SiteId','Value_sum','weight_sum']
    Temp3.loc[:,'Site_Exp_mean']=Temp3['Value_sum']/Temp3['weight_sum']
    del(result)
    return temp1.merge(Temp3[['SiteId','Site_Exp_mean']],on=['SiteId'],how='left').fillna(0)[['obs_id','Site_Exp_mean']]


# In[106]:


def get_Site_ForecastId_Exp_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].copy()
    del(temp2)
    result.loc[:,'Diff_Momentos']=(Time_Temp-result.Timestamp).dt.total_seconds()/900
    result.loc[:,'weight'] = result['Diff_Momentos'].apply(lambda x: 0.985**x)
    result['Value']=result['Value']*result['weight']
    
    Temp3=result[['SiteId','ForecastId','Value','weight']].groupby(['SiteId','ForecastId'],as_index=False).agg({'Value':sum,'weight':sum})
    Temp3.columns=['SiteId','ForecastId','Value_sum','weight_sum']
    Temp3.loc[:,'Site_ForecastId_Exp_mean']=Temp3['Value_sum']/Temp3['weight_sum']
    del(result)
    return temp1.merge(Temp3[['SiteId','ForecastId','Site_ForecastId_Exp_mean']],on=['SiteId','ForecastId'],how='left').fillna(0)[['obs_id','Site_ForecastId_Exp_mean']]


# In[109]:


def get_Site_ForecastId_Minute_Exp_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Minute','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Minute','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].copy()
    del(temp2)
    result.loc[:,'Diff_Momentos']=(Time_Temp-result.Timestamp).dt.total_seconds()/900
    result.loc[:,'weight'] = result['Diff_Momentos'].apply(lambda x: 0.985**x)
    result['Value']=result['Value']*result['weight']
    
    Temp3=result[['SiteId','ForecastId','Minute','Value','weight']].groupby(['SiteId','ForecastId','Minute'],as_index=False).agg({'Value':sum,'weight':sum})
    Temp3.columns=['SiteId','ForecastId','Minute','Value_sum','weight_sum']
    Temp3.loc[:,'Site_ForecastId_Minute_Exp_mean']=Temp3['Value_sum']/Temp3['weight_sum']
    del(result)
    return temp1.merge(Temp3[['SiteId','ForecastId','Minute','Site_ForecastId_Minute_Exp_mean']],on=['SiteId','ForecastId','Minute'],how='left').fillna(0)[['obs_id','Site_ForecastId_Minute_Exp_mean']]


# In[114]:


def get_Site_ForecastId_Hour_Exp_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Hour','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Hour','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].copy()
    del(temp2)
    result.loc[:,'Diff_Momentos']=(Time_Temp-result.Timestamp).dt.total_seconds()/900
    result.loc[:,'weight'] = result['Diff_Momentos'].apply(lambda x: 0.985**x)
    result['Value']=result['Value']*result['weight']
    
    Temp3=result[['SiteId','ForecastId','Hour','Value','weight']].groupby(['SiteId','ForecastId','Hour'],as_index=False).agg({'Value':sum,'weight':sum})
    Temp3.columns=['SiteId','ForecastId','Hour','Value_sum','weight_sum']
    Temp3.loc[:,'Site_ForecastId_Hour_Exp_mean']=Temp3['Value_sum']/Temp3['weight_sum']
    del(result)
    return temp1.merge(Temp3[['SiteId','ForecastId','Hour','Site_ForecastId_Hour_Exp_mean']],on=['SiteId','ForecastId','Hour'],how='left').fillna(0)[['obs_id','Site_ForecastId_Hour_Exp_mean']]


# In[121]:


def get_Site_ForecastId_Day_Exp_Feat(Indices_L1,Indices_L2,Time,n_momentos=1):
    temp1=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Day','Value']].iloc[Indices_L1.index,:].copy()
    temp2=Aux_Data[['obs_id','SiteId','Timestamp','ForecastId','Day','Value']].iloc[Indices_L2.index,:].copy()
    Time_Temp=Time[Indices_L2.index].copy()
    L_Time_Moments=Time_Temp-((n_momentos)*16)*timedelta(minutes=15)
    result=temp2[(temp2.Timestamp<Time_Temp)& (temp2.Timestamp>=L_Time_Moments)].copy()
    del(temp2)
    result.loc[:,'Diff_Momentos']=(Time_Temp-result.Timestamp).dt.total_seconds()/900
    result.loc[:,'weight'] = result['Diff_Momentos'].apply(lambda x: 0.985**x)
    result['Value']=result['Value']*result['weight']
    
    Temp3=result[['SiteId','ForecastId','Day','Value','weight']].groupby(['SiteId','ForecastId','Day'],as_index=False).agg({'Value':sum,'weight':sum})
    Temp3.columns=['SiteId','ForecastId','Day','Value_sum','weight_sum']
    Temp3.loc[:,'Site_ForecastId_Day_Exp_mean']=Temp3['Value_sum']/Temp3['weight_sum']
    del(result)
    return temp1.merge(Temp3[['SiteId','ForecastId','Day','Site_ForecastId_Day_Exp_mean']],on=['SiteId','ForecastId','Day'],how='left').fillna(0)[['obs_id','Site_ForecastId_Day_Exp_mean']]

def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result

import time

def Make_New_Data(L1,L2,Tiempo_Max):
    t0 = time.time()
    Salidas=[]
    Salidas.append(get_Site_Feat(L1,L2,Tiempo_Max,2))
    Salidas.append(get_Site_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_Feat(L1,L2,Tiempo_Max,12))
    Salidas.append(get_Site_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_ForecastId_Feat(L1,L2,Tiempo_Max,2))
    Salidas.append(get_ForecastId_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_ForecastId_Feat(L1,L2,Tiempo_Max,12))
    Salidas.append(get_ForecastId_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_ForecastId_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_ForecastId_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_ForecastId_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_Minute_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_Minute_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_Minute_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_Minute_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_Hour_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_Hour_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_Hour_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_Hour_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_Day_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_Day_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_DayofWeek_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_DayofWeek_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Minute_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_ForecastId_Minute_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Minute_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_ForecastId_Minute_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Hour_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Hour_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_ForecastId_Day_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Day_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_DayofWeek_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_DayofWeek_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_Exp_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_Exp_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_Exp_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_Exp_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Exp_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_ForecastId_Exp_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Exp_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_ForecastId_Exp_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Minute_Exp_Feat(L1,L2,Tiempo_Max,3))
    Salidas.append(get_Site_ForecastId_Minute_Exp_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Minute_Exp_Feat(L1,L2,Tiempo_Max,14))
    Salidas.append(get_Site_ForecastId_Minute_Exp_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Hour_Exp_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Hour_Exp_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_Day_Exp_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_Day_Exp_Feat(L1,L2,Tiempo_Max,17))
    Salidas.append(get_Site_ForecastId_DayofWeek_Feat(L1,L2,Tiempo_Max,6))
    Salidas.append(get_Site_ForecastId_DayofWeek_Feat(L1,L2,Tiempo_Max,17))
    print('merge...')
    result = concat(Salidas)
    del(Salidas)
    print('data shape：{}'.format(result.shape))
    print('spending {}s'.format(time.time() - t0))
    return result



