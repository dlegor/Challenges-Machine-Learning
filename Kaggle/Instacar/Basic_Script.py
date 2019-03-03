###########################################################################################################
#
# Kaggle Instacart competition
# Similary to Fabien Vavrand's script , Ago 2017
# Origina code: https://www.kaggle.com/fabienvs/instacart-xgboost-starter-lb-0-3791
# Simple xgboost starter, score 0.3791 on LB
# Products selection is based on product by product binary classification, with a global threshold (0.21)
# 
# @Daniel Legorreta 
# 
###########################################################################################################

import numpy 
import pandas
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load Data ---------------------------------------------------------------
path="../input/"

aisles=pd.read_csv(path+"aisles.csv")
departments=pd.read_csv(path+"departments.csv")
orderp=pd.read_csv(path+"order_products__prior.csv")
ordert= pd.read_csv(path+"order_products__train.csv")
orders= pd.read_csv(path+"orders.csv")
products= pd.read_csv(path+"products.csv")

# Reshape data ------------------------------------------------------------

Factor=LabelEncoder()
Factor.fit(aisles.aisle)
aisles['aisle']=Factor.transform(aisles.aisle)
Factor.fit(departments.department)
departments['department']=Factor.transform(departments.department)
Factor.fit(products.product_name)
products['product_name']=Factor.transform(products.product_name)

products=departments.join(aisles\
	                .join(products.set_index("aisle_id"),how="inner",on='aisle_id')\
	                .set_index("department_id"),how="inner",on='department_id')

del(products['aisle_id'])
del(products['department_id'])
del(aisles,departments)

ordert=pd.merge(ordert,orders[orders.eval_set=='train'][['order_id','user_id']],how='left',on='order_id')
orders_products=pd.merge(orders,orderp, how='inner',on = 'order_id')

del(orderp)

# Products ----------------------------------------------------------------

Aux_4=orders_products[['user_id','order_number','product_id','reordered']]\
                     .assign(product_time=orders_products\
                     	.sort_values(['user_id','order_number','product_id'])\
                     	.groupby(['user_id','product_id'])\
                     	.cumcount() + 1)

prd1=Aux_4.groupby('product_id')\
          .apply(lambda x:pd.Series(dict(prod_orders=x.user_id.count(),prod_reorders=x.reordered\
          	.sum(),prod_first_orders=(x.product_time == 1).sum(),prod_second_orders=(x.product_time == 2).sum())))

prd1.loc[:,'prod_reorder_probability']=prd1.prod_second_orders/prd1.prod_first_orders
prd1.loc[:,'prod_reorder_times']=1+prd1.prod_reorders/prd1.prod_first_orders
prd1.loc[:,'prod_reorder_ratio']=prd1.prod_reorders/prd1.prod_orders

prd=prd1.drop(['prod_reorders', 'prod_first_orders', 'prod_second_orders'],axis=1)

del(Aux_4,prd1)


# Users -------------------------------------------------------------------


users=orders[orders['eval_set'] == "prior"]\
                   .groupby('user_id')\
                   .agg({'order_number':'max','days_since_prior_order':['sum','mean']})

users.columns = ["_".join(x) for x in users.columns.ravel()]
users.columns=["user_orders","user_period","user_mean_days_since_prior"]

us=orders_products[['user_id','reordered','order_number','product_id']]\
                  .groupby('user_id')\
                  .apply(lambda x:pd.Series(dict(
    user_total_products=np.size(x.product_id),user_reorder_ratio = np.divide((x.reordered == 1).sum(),(x.order_number > 1).sum()),user_distinct_products =np.size(x.product_id.unique()))))

users1=pd.merge(users,us, left_index=True, right_index=True,how='inner')
users1.loc[:,'user_average_basket']=users1.user_total_products / users1.user_orders

del(us)

us=orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set','days_since_prior_order']]\
                         .rename(index=str, columns={"user_id": "user_id", "order_id": "order_id","eval_set":"eval_set","time_since_last_order":"days_since_prior_order"})      
users=pd.merge(users1,us,left_index=True,right_on="user_id")

del(us,users1)


# Database ----------------------------------------------------------------

data=orders_products[["user_id", "product_id","order_number","add_to_cart_order"]]\
                    .groupby(["user_id", "product_id"])\
                    .agg({'order_number':['min','max','size'],'add_to_cart_order':['mean']})

data.columns = ["_".join(x) for x in data.columns.ravel()]
data.reset_index(level=[0,1], inplace=True)
data.columns =["user_id","product_id","up_first_order","up_last_order","up_orders","up_average_cart_position"]

prd.reset_index(level=0, inplace=True)

data=users.join(prd\
	      .join(data.set_index("product_id"),on='product_id',how="inner")\
	      .set_index("user_id"),on="user_id",how="inner")

data.loc[:,"up_order_rate"]=data.up_orders / data.user_orders
data.loc[:,"up_orders_since_last_order"]=data.user_orders - data.up_last_order
data.loc[:,"up_order_rate_since_first_order"]=data.up_orders / (data.user_orders - data.up_first_order + 1)

data=pd.merge(data,ordert[["user_id", "product_id", "reordered"]],how='left',on=["user_id", "product_id"])

del(ordert,prd,users)


# Train / Test datasets ---------------------------------------------------

train=data[data.eval_set == "train"]
train=train.drop(labels=['eval_set','user_id','product_id','order_id'], axis=1)
train.reordered=train.reordered.fillna(0)

test=data[data.eval_set == "test"]
test=test.drop(labels=['eval_set','user_id','reordered'], axis=1)

del(data)

# Model -------------------------------------------------------------------
import xgboost as xgb


subtrain=train.sample(frac=0.5)

param={'objective':'reg:logistic','eval_metric':'logloss','eta':0.1,'max_depth':6,'min_child_weight':10,
'gamma':0.7,'subsample':0.76,'colsample_bytree':0.95,'alpha':2e-05,'lambda':10}

X_train =xgb.DMatrix(subtrain.drop( "reordered", axis=1), label = subtrain.loc[:,"reordered"])


num_round = 80
model = xgb.train(param, X_train, num_round)

X_test =xgb.DMatrix(test.drop(labels=['order_id','product_id'],axis=1))

test.loc[:,'reordered']=model.predict(X_test)
test.loc[:,'reordered']=np.where(test.reordered>.21,1,0)


test.loc[:,'product_id']=test.product_id.apply(lambda x: str(x))
Submission=test[test.reordered==1][['order_id','product_id']].groupby('order_id')['product_id']\
                                .agg(lambda x: ' '.join(x)).reset_index()

missing=pd.DataFrame()
missing.loc[:,'order_id']=np.unique(test.order_id[~test.order_id.isin(Submission.order_id)])
missing.loc[:,'product_id']=None

test_salida = pd.concat([Submission, missing], axis=0)
test_salida.columns=['order_id','products']

#Out writing
test_salida.to_csv("submit/submit4.csv", index=False)
