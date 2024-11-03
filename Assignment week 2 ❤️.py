#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df.duplicated().sum()


# In[9]:


df.dropna(inplace=True)


# In[10]:


df.info()


# In[12]:


df.shape


# In[13]:


df


# In[16]:


df['Year'] = df['Year'].str.replace(r'\(|\)', '',regex=True).astype(int)
df['Duration'] = pd.to_numeric(df['Duration'].str.replace(' min', ''))
df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''))


# In[18]:


df.dtypes


# In[19]:


df


# In[20]:


df.info()


# In[21]:


df


# In[22]:


df.describe(include='all')


# In[24]:


plt.style.use('seaborn-v0_8-white')
df[['Year', 'Duration', 'Votes']].hist(bins=30, edgecolor='black', figsize=(10,5))
plt.suptitle('Histogram of Numeric Features')
plt.show()


# In[25]:


df['Rating'].hist(bins=30, edgecolor='black', figsize=(10,5))
plt.suptitle('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('frequency')
plt.show()


# In[27]:


top_10_directors =df['Director'].value_counts(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_10_directors.values, y=top_10_directors.index, palette='Dark2')
plt.title('Top 10 directors with most movie involvements')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()


# In[28]:


top_10_genres = df['Genre'].value_counts(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_10_genres.values, y=top_10_genres.index, palette='muted')
plt.title('top 10 movie genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()


# In[29]:


combined_actors = pd.concat([df['Actor 1'], df['Actor 2'],  df['Actor 3']])


top_10_actors = combined_actors.value_counts().head(10)
top_10_actors


# In[30]:


plt.figure(figsize=(10,5))
sns.barplot(x=top_10_actors.values, y=top_10_actors.index, palette='Dark2')
plt.title('Top 10 actors with most movie involvements')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.show()


# In[34]:


average_rating_per_year = df.groupby('Year')['Rating'].mean().reset_index()
average_rating_per_year.columns = ['Year', 'Average Rating']

plt.figure(figsize=(8,4))
plt.plot(average_rating_per_year['Year'], average_rating_per_year['Average Rating'], linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.title('average movie rating over the years')
plt.grid(True)
plt.show()


# In[35]:


high_rating_movies = df[df['Rating'] > 8.5]
high_rating_movies


# In[36]:


rating_counts = high_rating_movies.groupby('Rating')['Name'].count().reset_index()
rating_counts.columns = ['Rating', 'Number of Movies']
rating_counts = rating_counts.sort_values(by='Rating', ascending=False)
rating_counts


# In[38]:


new_df = df.drop(columns=['Name', 'Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'])
corr=new_df.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidth=0.5)


# In[39]:


sns.pairplot(new_df, diag_kind='kde')
plt.suptitle('pair plot of features against rating ', y=1.02)
plt.show()


# In[41]:


df.drop('Name', axis=1)


# In[42]:


Genre_Average_Rating = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_Average_Rating'] = Genre_Average_Rating

Director_Average_Rating =df.groupby('Director')['Rating'].transform('mean')
df['Director_Average_Rating'] = Director_Average_Rating

Actor1_Average_Rating =df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor1_Average_Rating']=Actor1_Average_Rating

Actor2_Average_Rating =df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor2_Average_Rating']=Actor2_Average_Rating

Actor3_Average_Rating =df.groupby('Actor 3')['Rating'].transform('mean')
df['Actor3_Average_Rating']=Actor3_Average_Rating

df


# In[43]:


from sklearn.model_selection import train_test_split


x= df[['Year', 'Votes', 'Duration', 'Genre_Average_Rating', 'Director_Average_Rating', 'Actor1_Average_Rating', "Actor2_Average_Rating", 'Actor3_Average_Rating']]
y=df['Rating']


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[45]:


train_data= x_train, x_train.join(y_train)
train_data


# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

x_train, y_train = train_data[['Year', 'Votes', 'Duration', 'Genre_Average_Rating', 'Director_Average_Rating', 'Actor1_Average_Rating', 'Actor2_Average_Rating', 'Actor3_Average_Rating']], train_data['Rating']

fitted_model_lr = LinearRegression()
fitted_model_lr.fit(x_train, y_train)

y_pred_lr = fitted_model_lr.predict(x_test)



fitted_model_rf = RandomForestRegressor()
fitted_model_rf.fit(x_train, y_train)
y_pred_rf = fitted_model_rf.predict(x_test)


# In[48]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MSE_lr = mean_squared_error(y_test, y_pred_lr)
MAE_lr = mean_absolute_error(y_test, y_pred_lr)
R2_Score_lr = r2_score(y_test, y_pred_lr)


print('   Performance Evaluation for Linear Regression Model:  ')

print('Mean squared error is: ', MSE_lr)
print('Mean absolute error value is: ', MAE_lr)
print('R2 score value is: ', R2_Score_lr)



MSE_lr = mean_squared_error(y_test, y_pred_rf)
MAE_lr = mean_absolute_error(y_test, y_pred_rf)
R2_Score_rf = r2_score(y_test, y_pred_rf)



print('\n  Performance Evaluation for Random Forest Model: ')

print('Mean squared error is: ', MSE_lr)
print('Mean absolute error value is: ', MAE_lr)
print('R2 score value is: ', R2_Score_lr)


# In[49]:


plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.title('Linear Regression Model: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()




plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.title('Random Forest Model: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()


# In[50]:


x.head(10)


# In[51]:


y.head(10)


# In[55]:


data = {'Year': [2018], 'Votes': [100], 'Duration': [1
30], 'Genre Average Rating': [5,5],'Actor1_Average_Ra
ting': [5.5], 'Actor2_Aver
age_Rating': [5.8], 'Actor3_A
verage_Rating': [5.3]}
trail_data = pd.DataFrame(data)


# In[56]:


predict_rating = fitted_model_rf.predict(trail_data)
print('Predicted Rating for trial data: ', predict_rating[0])


# In[ ]:




