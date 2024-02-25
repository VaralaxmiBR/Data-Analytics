#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


a={
    "Name" :['sam' , 'amit'],
    'Grade' :['A' , 'B']
}


# In[6]:


df = pd.DataFrame(a)
df


# In[7]:


df.to_csv("C:\\Users\\pkabb\\OneDrive\\Desktop\\Prep\\Student_data.csv" , index = 'False')


# In[8]:


#Visualization EDA-> Exploratory data Analysis


# In[9]:


#EDA 
#1 Univariate anlayis-> Analysis on single columns 
#2 Bivariate analysis-> Analysis on 2 columns
#3 multivariate analysis-> on more than 2 columns 


# In[10]:


import numpy as np
import pandas as pd


# In[12]:


import matplotlib.pyplot as plt #visulaization library
import seaborn as sns #matplotlibs updated version or library


# In[13]:


df = pd.read_csv("C:\\Users\\pkabb\\OneDrive\\Desktop\\Prep\\titanic.csv")


# In[14]:


df.head()


# In[15]:


df.columns


# In[16]:


sns.countplot(x=df['Survived'])


# In[17]:


df['Survived'].value_counts()


# In[18]:


sns.countplot(x=df['Survived'])


# In[19]:


df['Survived'].value_counts()


# In[20]:


df['Pclass'].value_counts()


# In[22]:


sns.countplot(x=df['Pclass'])


# In[23]:


#If we want to find out percentage then we can use Pie-Chart .


# In[25]:


df['Survived'].value_counts().plot(kind='pie' , autopct= "%.2f")


# In[26]:


df['Survived'].value_counts().plot(kind ='pie' , autopct ="%.2f")


# In[29]:


df['Pclass'].value_counts().plot(kind= 'pie' , autopct = "%.2f")


# In[30]:


#1 #If we have numerical data then we can use Histogram because it finds the distroibution .


# In[32]:


plt.hist(x= df['Age'])
plt.show()


# In[35]:


#2 Distplot
#Curve-> KDE(Kernel Density Extraction)
#Use to find probability


# In[37]:


sns.distplot(x=df['Age'])
plt.show()


# In[41]:


sns.distplot(x= df['Age'] , hist= False)
plt.show()


# In[42]:


#3 Boxplot -> To find the outliers
#1. Lower fence 
#2. 25% of data
#3. IQR(Inter Quartile Range) 50% 


# In[43]:


sns.boxplot(x= df['Age'])


# In[ ]:




