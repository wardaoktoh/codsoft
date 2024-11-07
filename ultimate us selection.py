#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd #this pandas bestie helps us store, analyze,clean up rows and columns
import numpy as np #this is math whiz or math expert
import matplotlib.pyplot as plt #they make beautiful chart and plot
import plotly.express as px #fancy artist, making it eazy to engaging and visualize our data
import seaborn as sns 
import matplotlib.pyplot as plt #its like a vandalism with words
from wordcloud import WordCloud, STOPWORDS
from collections import Counter #it will count most common word
import pandas as pd
import warnings
warnings.filterwarnings('ignore')# this is for asking to skip unnecessary alerts, to keep it nice and tidy


# In[3]:


get_ipython().system('pip install wordcloud')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('US_Election_dataset_v1.csv')
df.head()


# In[6]:


df.shape #dataset outfit, rows(height), column(width/chuubiness)


# In[7]:


df.tail()


# In[8]:


df.info() # There are 3143 rows has 35 columns like Country,State,Votes,Education levels, Income, population



# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df.describe().T.plot(kind='bar')


# In[12]:


df.columns.to_list()


# In[13]:


for col in df:
    sns.histplot(x=col, data=df, kde=True)
    plt.show()


# In[14]:


df


# In[18]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import pandas as pd


stop_words_list = set(STOPWORDS)

COUNTS = Counter(df['state'].dropna().apply(lambda x:str(x)))



wcc = WordCloud(
    background_color ='black',
    width=1600, height =800,
    max_words=2000,
    stopwords=stop_words_list
)
wcc.generate_from_frequencies(COUNTS)



plt.figure(figsize=(10,5), facecolor='k')
plt.imshow(wcc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




