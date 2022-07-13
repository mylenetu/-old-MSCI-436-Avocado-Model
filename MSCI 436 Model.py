#!/usr/bin/env python
# coding: utf-8

# ### Install & import libraries 

# In[29]:


#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing package pandas (For Panel Data Analysis)

#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt                                     # Importing pyplot interface to use matplotlib
import seaborn as sns                                               # Importing seaborn library for interactive visualization
get_ipython().run_line_magic('matplotlib', 'inline')
#-------------------------------------------------------------------------------------------------------------------------------
import scipy as sp                                                  # Importing library for scientific calculations
#-------------------------------------------------------------------------------------------------------------------------------


# In[30]:


import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None


# In[31]:


# for data pipeline --------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

# for prediction (machine learning models) ------------------------

from sklearn.linear_model import*
from sklearn.preprocessing import*
from sklearn.ensemble import*
from sklearn.neighbors import*
from sklearn import svm
from sklearn.naive_bayes import*
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ### Data acquisition & description

# In[32]:


data = pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv')


# In[33]:


print('Data Shape:', data.shape)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[34]:


data.head()


# In[35]:


data.describe()


# In[36]:


data.dtypes


# ### Data Pre-processing 

# In[37]:


print('total number of duplicate values : ',sum(data.duplicated()))


# In[38]:


sns.heatmap(data.isnull());


# We can see that there are no duplicate values and no null values. 

# In[39]:


data=data.drop(['Unnamed: 0'], axis=1)


# In[40]:


data.head()


# In[41]:


datam=pd.read_csv(filepath_or_buffer = 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv') # Archieving main dataset


# In[42]:


data.select_dtypes('object').columns


# #### Outlier Detection

# In[43]:


sns.set_style("white")

plt.figure(figsize=(12,12))
sns.distplot(data.AveragePrice)
plt.title("Distribution of Average Price",fontsize=12);


# In[44]:


mean = data.AveragePrice.mean()
std = data.AveragePrice.std()
lower, upper = mean-std*2,mean+std*2 # Use 2*std and it will exclude data that is not included in 95% of data
print("Lower Limit : {} Upper Limit : {}".format(lower,upper))


# In[45]:


outliers = [x for x in data.AveragePrice if x < lower or x > upper]


# There is some data that is not included within 95% of data

# In[46]:


df_exclude = data[(data.AveragePrice < upper) | (data.AveragePrice > lower)]


# In[47]:


df_exclude.shape


# In[48]:


data.shape


# In[49]:


quantile = np.quantile(data.AveragePrice,[0.25,0.5,0.75,1]) # Use numpy quantile
IQR = quantile[2] - quantile[0] # Calculate IQR through third quantile - first quantile
upper = 1.5*IQR + quantile[2]
lower = quantile[0] - 1.5*IQR

print("Upper bound : {} Lower bound : {}".format(upper,lower))

outlier = [x for x in data.AveragePrice if x < lower or x>upper]


# In[50]:


df_exclude2 = data[(data.AveragePrice > lower) | (data.AveragePrice < upper)]
df_exclude2


# #### Data Normalization 

# In[51]:


log_data = np.log(data.AveragePrice+1)
sns.set_style("white")
plt.figure(figsize=(8,8))
sns.distplot(log_data);


# ### EDA & Class Imbalance Check

# In[53]:


len(data.region.unique())


# In[54]:


data.groupby('region').size() 


# #### Dates & Seasonality Check
# 
# We have two columns which are 'Date' and 'year', being year the extracted year of date. To make the analysis easier, let's extract day and month out of 'Date' and see each value separately. That way, we are also going to have two more potentially usefull columns: day and month

# In[57]:


from datetime import datetime
data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
# monday = 0
data['day of week'] = data['Date'].dt.dayofweek
dates = ['year', 'month', 'day', 'day of week']
data[dates]


# #### Day & Day of Week
# 
# We can see that the day chart has a repeating trend, and this is because of the day that the data was always recorded: day 6 (Sunday).
# The data was, therefore, recorded weekly, 'day of week' becomes redundant and we can eliminate it.

# data.drop('day of week', axis=1, inplace=True)

# In[60]:


data.head()


# In[61]:


print(len(data.type.unique()))

data.groupby('type').size()


# The types of avocados are also balanced since the ratio is close to 0.5 each.

# In[62]:


# Specifying dependent and independent variables

X = data[['4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region']]
Y = data['AveragePrice']
y=np.log1p(Y)


# In[63]:


X.head()


# In[64]:


Y.head()


# #### Label categorical variables

# In[66]:


# X_labelled = pd.get_dummies(X[["type","region"]], drop_first = True)
# X_labelled.head()

X = pd.get_dummies(X, prefix=["type","region"], columns=["type","region"], drop_first = True)
X.head()


# In[67]:


print(X.columns)


# ### Split into Train & Valid Set

# In[68]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[69]:


X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.3, random_state = 99)


# In[70]:


X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# ### Training the Model

# #### Multiple Linear Regression

# In[71]:


lr = LinearRegression()
lr.fit(X_train,y_train)

print("R2 of Linear Regresson:", lr.score(X_train,y_train) )
print("----- Prediction Accuracy-----")
print('MAE: ',metrics.mean_absolute_error(y_valid, lr.predict(X_valid)))
print('MSE: ',metrics.mean_squared_error(y_valid, lr.predict(X_valid)))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_valid, lr.predict(X_valid))))


# ### Support Vector Regression 

# In[79]:


from sklearn.svm import SVR


# #### Parameter Tuning or Hyper Parameter 

# In[80]:


svr = SVR(kernel='rbf', C=1, gamma= 0.5)   # Parameter Tuning to get the best accuracy

svr.fit(X_train,y_train)
print(svr.score(X_train,y_train))


# In[81]:


from math import sqrt 


# In[83]:


# calculate RMSE
error = sqrt(metrics.mean_squared_error(y_valid,svr.predict(X_valid))) 
print('RMSE value of the SVR Model is:', error)


# #### Train & Validation

# In[84]:


X=datam.drop('AveragePrice',1)
y=datam['AveragePrice']


# In[85]:


print('shape of X and y respectively :',X.shape,y.shape)


# In[86]:


X.head()


# #### *performing a 80-20 train test split over the dataset.*

# In[87]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[88]:


print('shape of X and y respectively(train) :',X_train.shape,y_train.shape)
print('shape of X and y respectively(test) :',X_test.shape,y_test.shape)


# In[89]:


cols=X_train.columns


# #### Preprocessing

# #### *Encoding all the categorical columns to dig deep into the data.*

# In[90]:


scaler=LabelEncoder()


# In[91]:


for col in X_train.columns:
    if datam[col].dtype=='object':
        X_train[col]=scaler.fit_transform(X_train[col])
        X_test[col]=scaler.transform(X_test[col])


# In[92]:


X_train.head()


# #### Variance Thresholding

# In[93]:


scaler=VarianceThreshold(0.1)


# In[94]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# #### Scaling

# In[96]:


scaler=StandardScaler()


# In[97]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[98]:


print("Type of X_train :",type(X_train))
print("Type of X_test :",type(X_test))


# In[99]:


X_train=pd.DataFrame(X_train,columns=cols)
X_train.head()


# In[100]:


X_test=pd.DataFrame(X_test,columns=cols)
X_test.head()


# In[101]:


print('Type of X_train and X_test :',type(X_train),type(X_test))


# ### Pipeline

# In[102]:


actr=[]
acts=[]
lstr=[]
lsts=[]


# #### Random Forest Regression

# In[103]:


clf=RandomForestRegressor(random_state=0)


# In[104]:


clf.fit(X_train,y_train)
y_tr1=clf.predict(X_train)
y_pr=clf.predict(X_test)


# In[105]:


print('train data accuracy :',clf.score(X_train,y_train))
print('test data accuracy :',clf.score(X_test,y_test))
print('loss of train data :',mean_squared_error(y_train,y_tr1))
print('loss of test data :',mean_squared_error(y_test,y_pr))


# So we can see the RFR really predicts the model very well and gives a quite accurate prediction.

# In[106]:


actr.append(clf.score(X_train,y_train))
acts.append(clf.score(X_test,y_test))
lstr.append(mean_squared_error(y_train,y_tr1))
lsts.append(mean_squared_error(y_test,y_pr))


# ### 70:30 Split

# In[107]:


from sklearn.model_selection import train_test_split

trainflights, testflights, ytrain, ytest = train_test_split(data, y, train_size=0.7,test_size=0.3, random_state=0)


# In[108]:


s = (trainflights.dtypes == 'object')
object_cols = list(s[s].index)

n = (trainflights.dtypes == ('float64','int64'))
numerical_cols = list(n[n].index)


# In[109]:


#checking the columns containing categorical columns:
print(object_cols)


# In[110]:


#using One Hot Encoder to make the categorical columns usable

oneHot = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
oneHottrain = pd.DataFrame(oneHot.fit_transform(trainflights[object_cols]))
oneHottest = pd.DataFrame(oneHot.transform(testflights[object_cols]))

#reattaching index since OneHotEncoder removes them:
oneHottrain.index = trainflights.index
oneHottest.index = testflights.index 

#dropping the old categorical columns:
cattraincol = trainflights.drop(object_cols, axis=1)
cattestcol = testflights.drop(object_cols, axis=1)

#concatenating the new columns:
trainflights = pd.concat([cattraincol, oneHottrain], axis=1)
testflights = pd.concat([cattestcol, oneHottest], axis=1)


# In[111]:


#scaling the values

trainf = trainflights.values
testf = testflights.values

minmax = MinMaxScaler()

trainflights = minmax.fit_transform(trainf)
testflights = minmax.transform(testf)

#defining a way to find Mean Absolute Percentage Error:
def PercentError(preds, ytest):
  error = abs(preds - ytest)

  errorp = np.mean(100 - 100*(error/ytest))

  print('the accuracy is:', errorp)


# In[ ]:




