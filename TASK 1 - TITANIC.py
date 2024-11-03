#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df= pd.read_csv('Titanic-Dataset.csv')


# In[5]:


sns.set_style(style='ticks')


# In[10]:


shape = df.shape
features = df.columns

print(f'The data has following shape {shape}')
print(f'The features present in the data are:', features.to_list(), sep='\n')


# In[11]:


df.head()


# In[12]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.dtypes


# In[16]:


numerical_data = df.select_dtypes(include = ['float64', 'int64']).columns.to_list()
categorical_data = df.select_dtypes(exclude = ['float64', 'int64']).columns.to_list()


# In[23]:


plt.pie(
    df['Survived'].value_counts().sort_values(ascending=False),
    labels = ['Died', 'Survived'],
    colors = ['Pink', 'Yellow'],
    autopct='%1.1f%%',

)

plt.legend(['Died', 'Lived'], loc='upper left')

plt.title('Survival State')
plt.show()


# In[26]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
sns.histplot(data=df, x='Age', ax= axes[0,0]).set_title('Age')
sns.histplot(data=df, x='SibSp', ax= axes[0,1]).set_title('SibSp')
sns.histplot(data=df, x='Parch', ax= axes[1,0]).set_title('Parch')
sns.histplot(data=df, x='Fare', ax= axes[1,1]).set_title('Fare')
plt.tight_layout()
plt.suptitle('Visualizing Numerical Data for Distribution Analysis', y=1.02)
plt.show()


# In[33]:


fig, axes = plt.subplots(1, 2, figsize=(12,3), sharey=True)

ax= sns.kdeplot(
    data=df,
    x='Age',
    hue='Survived',
    palette=['Red', 'Green'],
    legend=True,
    linewidth=0.5,
    ax=axes[0],
)


ax.set_xlim([0,80])

ax.set_title('Age')

ax.legend(['Dead', 'Alive'], title='Survived')


ax= sns.kdeplot(
    data=df,
    x='Fare',
    hue='Survived',
    palette=['Pink', 'Yellow'],
    legend=True,
    linewidth=0.5,
    ax=axes[1],
)

ax.set_xlim([10, 200])

ax.set_title('Fare')

ax.legend(['Dead', 'Alive'], title="Survived")

plt.subplots_adjust(hspace=0.6)
plt.suptitle('Age & Fare')
plt.show()


# In[35]:


correlation_matrix = df[numerical_data].corr()
plt.figure(figsize=(6,6))
sns.heatmap(correlation_matrix, annot=True).set_title('Correlation among numerical features')
plt.show()


# In[37]:


pivot_table_df = pd.pivot_table(data = df, index = 'Survived',
                               values = ['Age', 'SibSp', 'Parch', 'Fare'],
                                aggfunc='mean').round(2) 
pivot_table_df
                               


# In[39]:


plt.figure(figsize=(10,10))
sns.pairplot(data=df, vars=numerical_data, kind='scatter', hue='Survived', palette=['Black', 'Pink'])
plt.suptitle('Pair Plot of Titanic Data = Numerical Data', y=1.02)
plt.show()


# In[40]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

for i, feature in enumerate(categorical_data[:4]):
    row = i // 2
    col = i % 2
    sns.barplot(x=df[feature].value_counts().index, y=df[feature].value_counts(),
ax=axes[row, col])
    axes[row, col].set_title(feature)


# In[42]:


print(pd.pivot_table(df, index= 'Survived', columns = 'Pclass', values= 'Ticket', aggfunc='count'))
print()
print(pd.pivot_table(df, index ='Survived', columns= 'Sex', values= 'Ticket', aggfunc= 'count'))
print()
print(pd.pivot_table(df, index ='Survived', columns= 'Embarked', values= 'Ticket', aggfunc= 'count'))


# In[43]:


fig, axes = plt.subplots(1,3, figsize=(12,3))
plt.suptitle('Comparing survival status over categorical classes', y =1, fontsize=12)

sns.heatmap(pd.pivot_table(df, index= 'Survived', columns= 'Pclass', values= 'Ticket', aggfunc='count'),
           ax= axes[0], cbar = False,
           annot=True, fmt='')
sns.heatmap(pd.pivot_table(df, index= 'Survived', columns= 'Sex', values= 'Ticket', aggfunc='count'),
           ax= axes[1], cbar = False,
           annot=True, fmt='')
sns.heatmap(pd.pivot_table(df, index= 'Survived', columns= 'Sex', values= 'Ticket', aggfunc='count'),
           ax= axes[2], cbar = False,
           annot=True, fmt='')

plt.tight_layout()
plt.show()


# In[45]:


df.isna().sum().sort_values(ascending=False)


# In[46]:


import missingno as msno
msno.matrix(df)


# In[47]:


pip install missingno


# In[48]:


import missingno as msno
msno.matrix(df)


# In[50]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import FunctionTransformer

numerical_transformer = Pipeline( steps= [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
    
])

categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

def family_size(data):
    data['Family_Size'] = data ['SibSp'] + data['Parch'] + 1
    data['Family_Size'] = data ['Family_Size'].apply(lambda x: 1 if  x == 1 else (2 if x <= 4 else 3))
    
    return data

preprocessor = ColumnTransformer(transformers=[
    ('numerical_transformer', numerical_transformer, ['Age', 'Fare']),
    ('categorical_transformer', categorical_transformer, ['Sex', 'Embarked']),
    
])


# In[52]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

lr= LogisticRegression(random_state=42, penalty='12', C=0.2)

base_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('lr', lr)
])


# In[54]:


X = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis=1)
y= df.Survived
len(y)


# In[58]:


get_ipython().run_cell_magic('time', '', '\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)\n\nbase_pipeline.fit(X_train, y_train)\nbase_pipeline.score(X_test, y_test)\ny_pred = base_pipeline.predict(X_test)\n')


# In[59]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.savefig('conf.jpg')


# In[65]:


transformed_feature_names = []
numerical_feature_names = preprocessor.named_transformers_['numerical_transformer'].named_steps['scaler'].get_feature_names_out()
transformed_feature_names.extend(numerical_feature_names)
                                                          
categorical_feature_names = preprocessor.named_transformers_['categorical_transformer'].get_feature_names_out(input_features=['Sex', 'Embarked'])
transformed_feature_names.extend(categorical_feature_names)
family_size_feature_names = ['Family_Size']
transformed_feature_names.extend(family_size_feature_names)
                                                                                                
print(transformed_feature_names)
                                                                                                


# In[66]:


cv_scores = cross_val_score(base_pipeline,  X_train, y_train, cv=10)
cv_scores.mean()


# In[67]:


from sklearn.metrics import RocCurveDisplay

models = [LogisticRegression(random_state=42, penalty='12', C=0.8), RandomForestClassifier(), GradientBoostingClassifier()]

fig, ax = plt.subplots()

for model in models:
    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                             estimator_name=model)
    display.plot(ax=ax)
    plt.safefig('ROC.jpg')
    
    
    ax.set_title('ROC curves for different models')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    plt.show()


# In[68]:


base_pipeline.fit(X,y)
test = pd.read_csv('Titanic-Dataset.csv')
predictions = base_pipeline.predict(test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})
output.to_csv('submission1.csv', index=False)
print('saved!')


# In[ ]:




