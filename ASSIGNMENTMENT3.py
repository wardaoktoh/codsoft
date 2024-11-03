#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from collections import defaultdict
import matplotlib.pyplot as plt


# In[4]:


import numpy as np
from collections import defaultdict

class KNN:
    def __init__(self, k):
        self.l = k
        
        def fit(self, train):
            self.train_points = train.drop('species', axis=1).values
            self.classes = train['species'].values
        def distance(self, point1, point2):
            
            return np.sqrt(np.sum((point1 - point2) ** 2))
        
        def predict(self, test):
            predictions = []
            test_points = test.values
            
            for point2 in test_points:
                
                dists = np.array([self.distance(point, point2) for point in self.train_points])
                
                
                nearest_neighbors_idx = np.argsort(dists)[:self.k]
                
                
                votes = defaultdict(int)
                for idx in nearest_neighbors_idx:
                    votes[self.classes[idx]] += 1
                    
                    predicted_classes = max(votes, key=votes.get)
                    predictions.append(predicted_class)
                    
                return predictions


# In[12]:


import matplotlib.pyplot as plt

def confusion_matrix(actual, predicted):
    mat = [[0]*3 for _ in range(3)]
    for i in range(len(actual)):
        if(actual[i] == 'Iris-setosa'):
            if(predicted[i] == actual[i]):
                mat[0][0] += 1
            else:
                if(predicted[i] == 'Iris-versicolor'):
                    mat[0][1] += 1
                else:
                    mat[0][2] += 1
        elif(actual[i] == 'Iris-versicolor'):
            if(predicted[i] == actual[i]):
                mat[1][1] += 1
            else:
                if(predicted[i] == 'Iris-setosa'):
                    mat[1][0] += 1
                else:
                    mat[1][2] += 1
        else:
            if(predicted[i] == actual[i]):
                mat[2][2] += 1
            else:
                if(predicted[i] == 'Iris-setosa'):
                    mat[2][0] += 1
                else:
                    mat[2][1] += 1

    precs = mat[0][0]/(mat[0][0]+mat[1][0]+mat[2][0])
    precvs = mat[1][1]/(mat[0][1]+mat[1][1]+mat[2][1])
    precvg = mat[2][2]/(mat[0][2]+mat[1][2]+mat[2][2])
    precision = (precs+precvs+precvg)/3
    recs = mat[0][0]/(mat[0][0]+mat[0][1]+mat[0][2])
    recvs = mat[1][1]/(mat[1][0]+mat[1][1]+mat[1][2])
    recvg = mat[2][2]/(mat[2][0]+mat[2][1]+mat[2][2])
    recall = (recs+recvs+recvg)/3
    f1s = 2*precs*recs/(precs+recs)
    f1vs = 2*precvs*recvs/(precvs+recvs)
    f1vg = 2*precvg*recvg/(precvg+recvg)
    f1 = (f1s+f1vs+f1vg)/3
    accuracy = (mat[0][0]+mat[1][1]+mat[2][2])/len(actual)
    
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    print(f'Accuracy = {accuracy}')
    print(f'F1-Score = {f1}')
    
    # Customize color scheme but keep default labels
    plt.matshow(mat, cmap='coolwarm')  # Colormap changed to 'coolwarm'
    
    for i in range(3):
        for j in range(3):
            plt.text(j, i, mat[i][j], ha="center", va="center", color="black")  # Text in the center of the cells
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    # Keep default labels for the classes
    x = [0, 1, 2]
    y = [0, 1, 2]
    plt.xticks(x, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], fontsize=10)
    plt.yticks(y, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], fontsize=10)
    
    plt.colorbar()  # Adding a color bar for better visual representation
    plt.show()


# In[13]:


df = pd.read_csv('IRIS.csv')
df = df.sample(frac=1)
df.head()


# In[14]:


train_size = round(len(df)*0.8)
train = df[0:train_size]
test = df[train_size:].drop('species', axis=1)
test


# In[15]:


actual = df[train_size:]['species'].tolist()


# In[37]:


df['species'] = encoder.fit_transform(df['species'])


# In[38]:


knnC1f3cv = KNN(3)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train = train.iloc[i*val:(i+1)*val].drop('species', axis=1)
    
    knnC1f3cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['species'].tolist()
    predcsv = knnC1f3cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[20]:


knnClf3cv = KNN(3)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('species', axis=1)
    knnClf3cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['species'].tolist()
    predscv = knnClf3cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knnC1f3cv = KNeighborsClassifier(n_neighbors=3)  # or any other value for n_neighbors


# In[23]:


knnClf3cv = KNN(3)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('sepal_length', axis=1)
    knnClf3cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['sepal_length'].tolist()
    predscv = knnClf3cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[22]:


print(train.columns)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
knnC1f3cv = KNeighborsClassifier(n_neighbors=3)  # or any other value for n_neighbors


# In[25]:


knnClf3cv = KNN(3)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('sepal_length', axis=1)
    knnClf3cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['sepal_length'].tolist()
    predscv = knnClf3cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[27]:


from sklearn.neighbors import KNeighborsClassifier

# Instantiate the classifier
knnClf3cv = KNeighborsClassifier(n_neighbors=3)

# Run the cross-validation loop
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('sepal_length', axis=1)  # Adjust as needed
    knnClf3cv.fit(traincv.drop('sepal_width', axis=1), traincv['sepal_width'])  # Fit with features and target
    actualcv = train.iloc[i*val:(i+1)*val]['species'].tolist()
    predscv = knnClf3cv.predict(testcv)


# In[28]:


knnClf3cv = KNN(3)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('species', axis=1)
    knnClf3cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['species'].tolist()
    predscv = knnClf3cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[29]:


knnClf5 = KNN(5)
knnClf5.fit(train)
predictions = knnClf5.predict(test)
confusion_matrix(actual, predictions)


# In[30]:


knnClf5cv = KNN(5)
val = len(train)//10
for i in range(10):
    traincv = train.drop(train.index[i*val:(i+1)*val])
    testcv = train.iloc[i*val:(i+1)*val].drop('species', axis=1)
    knnClf5cv.fit(traincv)
    actualcv = train.iloc[i*val:(i+1)*val]['species'].tolist()
    predscv = knnClf5cv.predict(testcv)
    print(f'Fold {i+1}:')
    confusion_matrix(actualcv, predscv)


# In[31]:


This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[32]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[34]:


df = pd.read_csv('IRIS.csv')
df = df.sample(frac=1)
df.head()


# In[35]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[36]:


df['species'] = encoder.fit_transform(df['species'])


# In[39]:


df['species'] = encoder.fit_transform(df['species'])


# In[41]:


df.head()


# In[42]:


df = df[df['species'] != 0][['sepal_width','petal_length', 'species' ]]
df.head()


# In[43]:


#visualize the data
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


plt.scatter(df['sepal_width'], df['petal_length'], c = df['species'])


# In[46]:


# Taking only 10 random rows for training
df = df.sample(100)        #shuffle all the rows
df.head()


# In[47]:


df_train = df.iloc[:60, :].sample(10)
df_val = df.iloc[60:80, :].sample(5)
df_test = df.iloc[80: , :].sample(5)


# In[ ]:





# In[ ]:





# In[48]:


df_train


# In[49]:


X_test = df_val.iloc[:,0:2].values
y_test = df_val.iloc[:,-1].values


# In[50]:


df_bag = df_train.sample(8, replace = True)
X = df_bag.iloc[:, 0:2]
y = df_bag.iloc[:, -1]

df_bag


# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# In[ ]:





# In[52]:


get_ipython().system('pip install mlxtend')


# In[53]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# In[54]:


dt_bag1 = DecisionTreeClassifier()


# In[55]:


def evaluate(clf, X,y):  # takes input as model, X train and y train data
    clf.fit(X,y)         # now train the model
    plot_tree(clf)
    plt.show()
    plot_decision_regions(X.values, y.values, clf= clf, legend = 2)
    y_pred = clf.predict(X_test)
    print("Model Accuracy : ", accuracy_score(y_test, y_pred))


# In[56]:


evaluate(dt_bag1, X, y)


# In[57]:


df_bag = df_train.sample(8, replace = True)
X = df_bag.iloc[:, 0:2]
y = df_bag.iloc[:, -1]

df_bag


# In[91]:


dt_bag2 = DecisionTreeClassifier()
evaluate(dt_bag2, X,y)


# In[92]:


dt_bag3 = DecisionTreeClassifier()
evaluate(dt_bag3, X, y)


# In[59]:


df_test


# In[60]:


df_test.iloc[0, 0:2].values


# In[63]:


print('Predictor1 : ', dt_bag1.predict(df_test.iloc[1, 0:2].values.reshape(1,2)))
print('Predictor2 : ', dt_bag2.predict(df_test.iloc[1, 0:2].values.reshape(1,2)))
print('Predictor3 : ', dt_bag3.predict(df_test.iloc[1, 0:2].values.reshape(1,2)))


# In[97]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
        


# In[98]:


df = pd.read_csv('IRIS.csv')
df.head()


# In[99]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[100]:


df['species'] = encoder.fit_transform(df['species'])


# In[101]:


df.head()


# In[102]:


df = df[df['species'] !=0][['sepal_width', 'petal_length', 'species']]
        
df.head()


# In[103]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[104]:


plt.scatter(df['sepal_width'], df['petal_length'], c = df['species'])


# In[105]:


df = df.sample(100)
df.head()


# In[106]:


df_train = df.iloc[:60, :].sample(10)
df_val = df.iloc[60:80, :].sample(5)
df_test = df.iloc[80:, :].sample(5)


# In[107]:


df_train


# In[108]:


X_test = df_val.iloc[:, 0:2].values
y_test = df_val.iloc[:, -1].values


# In[109]:


df_bag = df_train.sample(8, replace = True)
X = df_bag.iloc[:, 0:2]
y= df_bag.iloc[:, -1]
df_bag


# In[110]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# In[111]:


dt_bag1 = DecisionTreeClassifier()


# In[112]:


def evaluate(clf, X,y): 
    clf.fit(X,y)
    plot_tree(clf)
    plt.show()
    plot_decision_regions(X.values, y.values, clf=clf, legend=2)
    y_pred = clf.predict(X_test)
    print('Model Accuracy: ', accuracy_score(y_test, y_pred))


# In[113]:


print(X.shape)
print(y.shape)


# In[114]:


def evaluate(clf, X, y):
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Error: Empty dataset provided.")
        return
    clf.fit(X, y)
    plot_tree(clf)
    plt.show()


# In[115]:


evaluate(dt_bag1, X,y)


# In[116]:


df_bag = df_train.sample(8, replace=True)
X = df_bag.iloc[: 0:2]
y = df_bag.iloc[:, -1]

df_bag


# In[129]:


dt_bag2 = DecisionTreeClassifier()
evaluate(dt_bag2, X,y)


# In[130]:


df_bag = df_train.sample(8, replace =True)
X = df_bag.iloc[:, 0:2]
y=df_bag.iloc[:, -1]
df_bag


# In[131]:


dt_bag3 = DecisionTreeClassifier()
evaluate(dt_bag3, X, y)


# In[132]:


df_test.iloc[0, 0:2].values


# In[133]:


print('Predictor1 : ', dt_bag1.predict(df_test.iloc[1, 0:2].values.reshape(1,2)))


# In[134]:


from sklearn.exceptions import NotFittedError


# In[ ]:




