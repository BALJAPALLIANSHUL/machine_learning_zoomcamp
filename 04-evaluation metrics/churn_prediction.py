#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install kagglehub[pandas-datasets]


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # 3.1 Download and overview of data
import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)
# In[3]:


#!mv "/home/codespace/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1/WA_Fn-UseC_-Telco-Customer-Churn.csv" .


# In[4]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[5]:


df.head()


# # 3.2 Data preparation

# In[6]:


df.head().T


# ### Normalize column names

# In[7]:


df.columns = df.columns.str.lower().str.replace(' ','_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.head().T


# In[8]:


df.dtypes

#totalcharges is of object: should have been a float
#seniorcitizen is of int64: should have been a object 'yes' or 'no'  but works good with model
# In[9]:


df.isnull().sum()


# In[10]:


# there was '_' for this column if there is no data. So we converted it to nan
df.totalcharges = pd.to_numeric(df.totalcharges,errors = 'coerce')


# In[11]:


df.isnull().sum()


# In[12]:


df.totalcharges = df.totalcharges.fillna(0)


# In[13]:


df.churn.unique()


# In[14]:


# For now we will only modify one categorical variable 'churn'
df.churn = (df.churn == 'yes').astype('int64')


# In[15]:


df.churn.unique()


# # 3.3 Setting up the validation framework

# In[16]:


#!pip install scikit-learn


# In[17]:


#!python3 -m pip install --upgrade pip


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


df.columns


# In[20]:


features = [ 'gender', 'seniorcitizen', 'partner', 'dependents',
       'tenure', 'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod', 'monthlycharges', 'totalcharges']
target = ['churn']


# In[21]:


X = df[features]
y = df[target]
X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size = 0.2,random_state=1)


# In[22]:


len(X_train_val),len(y_train_val)


# In[23]:


X_train,X_val,y_train,y_val = train_test_split(X_train_val,y_train_val,test_size = (20/80),random_state=1)


# In[24]:


len(X_train),len(X_val),len(X_test)


# # 3.4 EDA

# In[25]:


df.isnull().sum()


# In[26]:


y_train.churn.value_counts(normalize = True)


# In[27]:


df.dtypes


# In[28]:


numerical = ['tenure','monthlycharges','totalcharges']
categorical = [ 'gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
Id = ['customerid']


# In[29]:


df[categorical].nunique() # Check on original data set


# In[30]:


X_train[categorical].nunique() # check on training set


# In[31]:


# churn rate: percentage of customers opted out from services
# churn is a categorical variable {0,1} 0=no churn, 1 = churn. So mean will be churn rate mean = sum/no of records
churn_rate_train = y_train.mean()
churn_rate_train


# In[32]:


churn_rate_val = y_val.mean()
churn_rate_val


# In[33]:


churn_rate_test = y_test.mean()
churn_rate_test


# # 3.5 Feature importance: Churn rate and risk ratio

# Feature importance analysis (part of EDA) - identifying which features affect our target variable
# 
# - Churn rate
# - Risk ratio
# - Mutual information - later

# In[34]:


churn_female = df[df.gender == 'female'].churn.mean()
churn_female


# In[35]:


churn_male = df[df.gender == 'male'].churn.mean()
churn_male


# In[36]:


global_churn = df.churn.mean()


# In[37]:


global_churn-churn_male


# In[38]:


global_churn/churn_male # risk ratio


# In[39]:


global_churn-churn_female 


# In[40]:


global_churn/churn_female # risk ratio

'''SELECT
    gender,
    AVG(churn),
    AVG(churn) - global_churn AS diff,
    AVG(churn) / global_churn AS risk
FROM
    data
GROUP BY
    gender;
'''
# In[41]:


# pandas way of doing
df_groupby_gender = df.groupby('gender').churn.agg(['mean','count'])
df_groupby_gender.columns = df_groupby_gender.columns.str.replace('mean','churn_rate')
df_groupby_gender


# In[42]:


from IPython.display import display
for c in categorical:
        print(c)
        df_groupby = df.groupby(c).churn.agg(['mean','count'])
        df_groupby.columns = df_groupby.columns.str.replace('mean','churn_rate')
        df_groupby['diff with global churn rate'] = df_groupby['churn_rate']-global_churn
        df_groupby['risk_ratio'] = df_groupby['churn_rate']/global_churn
        display(df_groupby)
        print('\n\n')


# ## 3.6 - Feature Importance: Mutual Information

# In[43]:


from sklearn.metrics import mutual_info_score


# In[44]:


mutual_info_score(X_train.contract,y_train.churn)


# In[45]:


mutual_info_score(X_train.gender,y_train.churn)


# In[46]:


mutual_info_score(df.dependents,df.churn)


# In[47]:


mutual_info_score(df.partner,df.churn)


# In[48]:


def mututal_info_score_func(series):
    return mutual_info_score(series,y_train_val.churn)


# In[49]:


scores = X_train_val[categorical].apply(mututal_info_score_func)


# In[50]:


scores.sort_values(ascending=False)


# ## 3.7 - Feature Importance: Correlation

# In[51]:


X_train_val[numerical]


# In[52]:


y_train_val


# In[53]:


X_train_val[numerical].corrwith(y_train_val.reset_index(drop = True)['churn'])


# ## 3.8 One-hot encoding

# In[54]:


from sklearn.feature_extraction import DictVectorizer


# In[55]:


dv = DictVectorizer(sparse=False)


# In[56]:


dicts = X_train[categorical+numerical].to_dict(orient='records')


# In[57]:


X_train_ohe = dv.fit_transform(dicts)
X_train_ohe


# In[58]:


val_dicts = X_val[categorical+numerical].to_dict(orient='records')


# In[59]:


X_val_ohe = dv.transform(val_dicts)
X_val_ohe


# ## 3.9 Logistic regression

# - Binary classification
# - Linear vs logistic regression

# In[60]:


def sigmoid(z):
    return 1/(1+np.exp(-1*z))


# In[61]:


z = np.linspace(-10,10,100)


# In[62]:


plt.plot(z,sigmoid(z))


# # 3.10 Training logistic regression with Scikit-Learn

# - Train a model with Scikit-Learn
# - Apply it to the validation dataset
# - Calculate the accuracy

# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


model = LogisticRegression()


# In[65]:


model.fit(X_train_ohe,y_train)


# In[66]:


model.coef_[0].round(3)


# In[67]:


model.intercept_[0].round(3)


# In[68]:


y_pred = model.predict(X_val_ohe).reshape(-1,1)
y_pred


# In[69]:


y_pred_prob = model.predict_proba(X_val_ohe)[:,1]


# In[70]:


y_val.shape,y_pred.shape


# In[71]:


correct_pred = (y_val == y_pred)


# In[72]:


correct_df = pd.DataFrame()


# In[73]:


correct_df['actual'] = y_val
correct_df['prediction'] = y_pred
correct_df['probability'] = y_pred_prob
correct_df['correct_prediction'] = correct_pred


# In[74]:


correct_df


# In[75]:


accuracy = correct_df.correct_prediction.mean()
accuracy*100


# # 3.11 - Model Interpretation
# - Look at the coefficients
# - Train a smaller model with fewer features

# In[76]:


dict(zip(dv.get_feature_names_out(),model.coef_[0]))


# In[77]:


small_feat_set = ['contract','tenure','monthlycharges'] 


# In[78]:


X_small_train = X_train[small_feat_set].reset_index(drop=True)
X_small_train.head()


# In[79]:


X_small_val = X_val[small_feat_set].reset_index(drop=True)
X_small_val.head()


# In[80]:


dv1 = DictVectorizer(sparse = False)


# In[81]:


dicts_train = X_small_train.to_dict(orient='records')
dicts_val = X_small_val.to_dict(orient='records')


# In[82]:


dv1.fit(dicts_train)


# In[83]:


dv1.get_feature_names_out()


# In[84]:


X_train_small_ohe = dv1.transform(dicts_train)
X_val_small_ohe = dv1.transform(dicts_val)


# In[85]:


model_small = LogisticRegression()


# In[103]:


model_small.fit(X_train_small_ohe,y_train.churn.values)


# In[104]:


coef,bias = model_small.coef_[0], model_small.intercept_[0]


# - As the duration of contract increases, probability of churn decreased

# In[105]:


# month to month contract
churn_prob = sigmoid(1*coef[0]+1*coef[3]+50*coef[-1]+bias)
churn_prob


# In[106]:


# one_year contract
churn_prob = sigmoid(1*coef[1]+1*coef[3]+50*coef[-1]+bias)
churn_prob


# In[107]:


# two_year contract
churn_prob = sigmoid(1*coef[2]+1*coef[3]+50*coef[-1]+bias)
churn_prob


# - As monthly charges increase, churn probability decreased, which is counter intuitive as per general 
#   customer behaviour

# In[108]:


# month to month contract
churn_prob = sigmoid(1*coef[0]+1*coef[3]+10*coef[-1]+bias)
churn_prob


# In[109]:


# month to month contract
churn_prob = sigmoid(1*coef[0]+1*coef[3]+20*coef[-1]+bias)
churn_prob


# In[110]:


# month to month contract
churn_prob = sigmoid(1*coef[0]+1*coef[3]+30*coef[-1]+bias)
churn_prob


# - accuracy less compared to model trained on all features

# In[111]:


y_pred_small = model_small.predict(X_val_small_ohe).reshape(-1,1)
accuracy = (y_pred_small == y_val).churn.mean()
accuracy*100


# # 3.12 - Testing the model

# In[112]:


dicts_full_train = X_train_val.to_dict(orient = 'records')
dicts_full_train[:1]


# In[113]:


X_full_train_ohe = dv.transform(dicts_full_train)
X_full_train_ohe


# In[121]:


model_full = LogisticRegression(solver='lbfgs',max_iter=100)


# In[122]:


model_full.fit(X_full_train_ohe,y_train_val.churn.values)


# In[123]:


dicts_test = X_test.to_dict(orient = 'records')
dicts_test[:1]


# In[124]:


X_test_ohe =dv.transform(dicts_test)
X_test_ohe


# In[125]:


y_pred_test = model_full.predict(X_test_ohe).reshape(-1,1)
y_pred_test


# In[126]:


accuracy = (y_pred_test == y_test).churn.mean()
accuracy*100


# In[ ]:




