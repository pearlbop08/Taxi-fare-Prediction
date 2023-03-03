#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries and algorithms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
print("Libraries Import Successfully")
from sklearn.pipeline import make_pipeline


# In[2]:


#importing the data
data=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/TaxiFare.csv")


# ## We need to predict the dollar amount of the cost of the taxi ride.
# ### hence it is a Regession problem cause it is the continous data
# 

# # Visualizing the data

# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


sns.pairplot(data,diag_kind='kde')


# In[7]:


data.boxplot()


# In[8]:


plt.figure(figsize=(10,10))
sns.heatmap(data.isnull())


# In[9]:


data.hist(figsize=(20,15))


# In[10]:


duplicate=data.duplicated()
print(duplicate.sum())


# In[11]:


data.drop_duplicates(inplace=True)
data.duplicated().sum()


# ### Data Cleaning

# In[12]:


data.boxplot()


# In[13]:


data.boxplot(column=["amount"])


# In[14]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-1.5*IQR
    upper_range=Q3+1.5*IQR
    return lower_range,upper_range


# In[15]:


low,high=remove_outlier(data["amount"])


# In[16]:


data["amount"]=np.where(data["amount"]>high,high,data["amount"])


# In[17]:


data["amount"]=np.where(data["amount"]>low,low,data["amount"])


# In[18]:


data.boxplot(column="amount")


# In[19]:


data['date_time_of_pickup'] = pd.to_datetime(data.date_time_of_pickup)


# In[20]:


# import module

data['date_time_of_pickup'] = pd.to_datetime(data["date_time_of_pickup"])
data["pickup_date"]=data["date_time_of_pickup"].dt.date
data["pickup_day"]=data["date_time_of_pickup"].dt.day
data["pickup_hour"]=data["date_time_of_pickup"].dt.hour
data["pickup_day_of_week"]=data["date_time_of_pickup"].dt.dayofweek
data["pickup_month"]=data["date_time_of_pickup"].dt.month
data["pickup_year"]=data["date_time_of_pickup"].dt.year
data.drop('date_time_of_pickup',axis = 1,inplace = True)
data.drop("unique_id",axis = 1,inplace = True)
   


# def baseFare(x):
#     if x in range(16,20):
#         base_fare = 3.50
#     elif x in range(20,24):
#         base_fare = 3
#     else:
#         base_fare = 2.50
#     return base_fare
# 
# data['base_fare'] = data['pickup_hour'].apply(baseFare)
# 
# data['base_fare'], data['pickup_hour']

# data['fare'] = data['amount'] - data['base_fare']

# ### Again visualizing the data

# In[21]:


# Datetime features
plt.figure(figsize=(22, 6))

# Hour of day
plt.subplot(221)
sns.countplot(data['pickup_hour'])
plt.xlabel('Hour of Day')
plt.ylabel('Total number of pickups')
plt.title('Hourly Variation of Total number of pickups')

# Date
plt.subplot(223)
sns.countplot(data['pickup_date'])
plt.xlabel('Date')
plt.ylabel('Total number of pickups')
plt.title('Daily Variation of Total number of pickups')

# Day of week
plt.subplot(222)
sns.countplot(data['pickup_day_of_week'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Week Day')
plt.ylabel('Total Number of pickups')
plt.title('Weekly Variation of Total number of pickups')

# Month
plt.subplot(224)
sns.countplot(data['pickup_month'])
plt.xlabel('Month')
plt.ylabel('Total number of pickups')
plt.title('Monthly Variation of Total number of pickups');


# In[22]:


data.drop("pickup_date", axis=1, inplace=True)


# In[23]:


data.info()


# In[24]:


data.describe()


# In[25]:


data.head()


# In[26]:


data.isnull().sum()


# In[27]:


data.dtypes


# In[28]:


pip install folium


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
import folium

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# In[30]:


map_nyc = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

pickups = []
dropoffs = []
for index, row in data.iterrows():
    pickup = [row["latitude_of_pickup"], row["longitude_of_pickup"]]
    dropoff = [row["latitude_of_dropoff"], row["longitude_of_dropoff"]]
    pickups.append(pickup)
    dropoffs.append(dropoff)

#Plotting pickups and dropoff    
for index, (pickup, dropoff) in enumerate(zip(pickups[:10], dropoffs[:10])):
    folium.Marker(
        location=pickup,
        popup="Pickup",
        icon=folium.Icon(color="blue"),
        ).add_to(map_nyc)
    folium.Marker(
        location=dropoff,
        popup="Dropoff",
        icon=folium.Icon(color="red"),
        ).add_to(map_nyc)
map_nyc


# ## Modelling

# In[31]:


X = data.drop(["amount"], axis=1)
#dependent variable
Y= data["amount"]


# In[32]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y ,test_size=0.30, random_state=1)


# In[33]:


from sklearn.linear_model import LinearRegression


# ### Linear Regression
# 

# In[34]:


lr = LinearRegression()


# In[35]:


lr.fit(X_train,Y_train)


# In[36]:


lr.score(X_train, Y_train)


# In[37]:


lr.score(X_test, Y_test)


# ### Linear Reagression with Standardization
# 

# In[38]:


from sklearn.preprocessing import StandardScaler


# In[39]:


scaler= StandardScaler()


# In[40]:


X_train=scaler.fit_transform(X_train)


# In[41]:


X_test=scaler.fit_transform(X_test)


# In[42]:


lr.fit(X_train,Y_train)


# In[43]:


lr.score(X_train, Y_train)


# In[44]:


lr.score(X_test, Y_test)


# ### KNN Regression
# 

# In[45]:


knn = KNeighborsRegressor(n_neighbors=5,p=1)


# In[46]:


knn.fit(X_train,Y_train)


# In[47]:


knn.score(X_train, Y_train)


# In[48]:


knn.score(X_test, Y_test)


# ### SVR

# In[49]:


svr = SVR()


# In[50]:


svr.fit(X_train,Y_train)


# In[51]:


svr.score(X_train, Y_train)


# In[52]:


svr.score(X_test, Y_test)


# ### DecisionTreeRegressor

# In[53]:


dt = DecisionTreeRegressor(random_state=0)


# In[54]:


dt.fit(X_train,Y_train)


# In[55]:


dt.score(X_train, Y_train)


# In[56]:


dt.score(X_test, Y_test)


# ### BaggingRegressor

# In[57]:


bg = BaggingRegressor(n_estimators=1, random_state=2) 


# In[58]:


bg.fit(X_train,Y_train)


# In[59]:


bg.score(X_train, Y_train)


# In[60]:


bg.score(X_test, Y_test)


# ### AdaBoostRegressor

# In[61]:


abr = AdaBoostRegressor(random_state=1)


# In[62]:


abr.fit(X_train,Y_train)


# In[63]:


abr.score(X_train, Y_train)


# In[64]:


abr.score(X_test, Y_test)


# ### GradientBoostingRegressor

# In[65]:


gbr = GradientBoostingRegressor(max_depth=1, random_state=1,max_features='sqrt')


# In[66]:


gbr.fit(X_train,Y_train)


# In[67]:


gbr.score(X_train, Y_train)


# In[68]:


gbr.score(X_test, Y_test)


# ### RandomForestRegressor

# In[69]:


rf = RandomForestRegressor(max_depth=1, random_state=2,max_features='sqrt')


# In[70]:


rf.fit(X_train,Y_train)


# In[71]:


rf.score(X_train, Y_train)


# In[72]:


rf.score(X_test, Y_test)


# # Hence BaggingRegressor and DecisionTreeRegressor are best algorithms
