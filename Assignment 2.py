#!/usr/bin/env python
# coding: utf-8

# In[8]:


#CASE STUDY 1
Problem -Definition
Case Study 1: Sales Data Analysis
Scenario: You have a dataset containing sales data with columns for date, product ID, quantity sold, and total revenue. Perform the following tasks:

Load the dataset into a Pandas DataFrame.
Calculate the total revenue and quantity sold for each product.
Identify the product with the highest total revenue.
Plot a bar chart showing the total revenue for each product.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms


# In[24]:


Sales_Data = {
    "DATE":['2024-02-01','2024-02-02','2024-02-03','2024-02-04','2024-02-05'],   
    "Product_ID":['P001','P002','P003','P004','P005'],
    "Quantity_Sold":[ 10,15,20,25,30],
    "Total_Revenue":[100,200,300,400,500]                 
}


# In[26]:


df=pd.DataFrame(Sales_Data)
df


# In[30]:


Quantity_Sold= df["Quantity_Sold"].sum()
print("Total Quantity Sold:" , Quantity_Sold)


# In[31]:


Total_Revenue=df["Total_Revenue"].sum()
print("Total Revenue in Rs. :" , Total_Revenue)


# In[35]:


product_sales= df.groupby('Product_ID')['Quantity_Sold'].sum().reset_index(name='Total_Revenue')
top_selling_product = product_sales.loc[product_sales['Total_Revenue'].idxmax(),:]['Product_ID']
print("Product with the highest total revenue:" , top_selling_product)


# In[42]:


df=pd.DataFrame(Sales_Data)
Product_Revenue= df.groupby('DATE')['Total_Revenue'].sum()
plt.bar(Product_Revenue.index , Product_Revenue.values)


# In[46]:


#CASE STUDY 2

#Case Study 2: Student Performance Analysis
#Scenario: You have a dataset containing student information, including names, grades, and attendance. Perform the following tasks:

#Load the dataset into a Pandas DataFrame.
#Calculate the average grade for each student.
#Identify the student with the highest average grade.
#Plot a histogram showing the distribution of grades.

Student_Data={
    "Name":["Pavan","Varu","Abhi","Adi","Nitin","Neeraj"],
    "Maths":[80,70,50,50,40,40],
    "Science":[90,80,70,60,60,60],
    "SocialStudies":[80,90,70,60,50,50],
    "English":[100,90,80,80,80,80],
    "Attendance":['Present','Present','Present','Absent','Absent','Present'],
}
df=pd.DataFrame(Student_Data)
df


# In[49]:


df['Average_Grade']=df[['Science','Maths','SocialStudies','English']].mean(axis=1)
df['Average_Grade']


# In[51]:


average_grade=df.groupby('Name')['Average_Grade'].mean()
highest_avg_student=average_grade.idxmax()
print("The highest average grade among these students is:",highest_avg_student)


# In[52]:


grade=df['Average_Grade']
plt.hist(x=df['Average_Grade'])
plt.ylabel('Average_Grade')
plt.title("Histogram distribution of grades")


# In[56]:


#CASE 3

Employee_Information={
    "Name":['Pavan','Varu','Abhi','Adi','Nitin','Neeraj'],
    "Departments":['IT','IT','Pharmacy','Pharmacy','Finance','Finance'],
    "Salaries":[40000,60000,50000,50000,20000,20000]
}
df=pd.DataFrame(Employee_Information)
df


# In[59]:


avg_salary=df.groupby('Departments')['Salaries'].mean()
avg_salary


# In[65]:


avg_salary=df.groupby('Departments')['Salaries'].mean().reset_index(name='avg_salary')
highest_avg_salary=avg_salary.loc[avg_salary['avg_salary'].idxmax(),'Departments']
print("Department with the highest averge salary is :",highest_avg_salary)


# In[72]:


avg_salary.plot(kind='barh',color="b"),
plt.title("Average Salary by Department"),
plt.xlabel("Average Salary"),
plt.ylabel("Department")


# In[2]:


#Q.1 How is NumPy better than List ?
#There are two main reasons why we would use NumPy array instead of lists in Python. These reasons are:

#Less memory usage: The Python NumPy array consumes less memory than lists.
#Less execution time: The NumPy array is pretty fast in terms of execution, as compared to lists in Python.

#Q.2 Difference between zeros() , ones() , eye() , diag() , randomint() , rand() ?Please provide definition with examples ?
#The numpy. zeros() function provide a new array of given shape and type, which is filled with zeros.
#Python numpy.ones() function returns a new array of given shape and data type, where the element’s value is set to 1. This function is very similar to numpy zeros() function.
#eye() function in Python is used to return a two-dimensional array with ones (1) on the diagonal and zeros (0) elsewhere.
#diag() Extract a diagonal or construct a diagonal array.
#randomint() randint() is one of the function for doing random sampling in numpy. It returns an array of specified shape and fills it with random integers from low (inclusive) to high (exclusive), i.e. in the interval [low, high).
#RAND returns an evenly distributed random real number greater than or equal to 0 and less than 1.



# In[78]:


import numpy as np
np.zeros(5)


# In[83]:


np.zeros((2,1))


# In[84]:


np.zeros((4,5))


# In[85]:


np.ones(5)


# In[86]:


np.ones((1,5))


# In[87]:


np.eye(2)


# In[90]:


np.eye(3)


# In[94]:


np.diag((1,2,3,4,5))


# In[107]:


import random
print(random.randint(1,9))


# In[118]:


import random
print(random.random())


# In[119]:


#Q.3 Difference between linspace() and unique() with example?
#The numpy.linspace() fucntion creates an array with equally spaced values in a specified interval . The linspace() uses parameters such as start stop and num to return equal spaced values between start and stop .


# In[122]:


import numpy as np
np.linspace(1,2, num=5)


# In[127]:


np.linspace(1,5 , num= 5 , endpoint=False)


# In[133]:


# numpy.unique() returns only the unique values in the list. 

import numpy as np
list=[10,20,30,40,50,60,10,20,60,30]
np.unique(list)


# In[134]:


#Q.4 Difference between hstack() and vstack() ?

#HStack positions views in a horizontal line, VStack positions them in a vertical line, and ZStack overlays views on top of one another.

import numpy as np


# In[143]:


a=np.ones((3,3))
np.vstack((a,np.array((2,2,2))))


# In[147]:


a=np.zeros((4,4))
np.vstack((a , np.array((2,2,2,2))))


# In[149]:


a = np.ones((3, 3))
b = np.array((2,2,2)).reshape(3,1)
b
np.hstack((a, b))


# In[7]:


# difference between loc() and iloc() ?
#loc is typically used for label indexing and can access multiple columns, while . iloc is used for integer indexing.

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
Data = pd.DataFrame({'Brand': ['Maruti', 'Hyundai', 'Tata',
                               'Mahindra', 'Maruti', 'Hyundai',
                               'Renault', 'Tata', 'Maruti'],
                     'Year': [2012, 2014, 2011, 2015, 2012,
                              2016, 2014, 2018, 2019],
                     'Kms Driven': [50000, 30000, 60000,
                                    25000, 10000, 46000,
                                    31000, 15000, 12000],
                     'City': ['Gurgaon', 'Delhi', 'Mumbai',
                              'Delhi', 'Mumbai', 'Delhi',
                              'Mumbai', 'Chennai',  'Ghaziabad'],
                     'Mileage':  [28, 27, 25, 26, 28,
                                  29, 24, 21, 24]})
print(data)


# In[8]:


print(data.loc[(data.Brand == 'Maruti') & (data.Mileage > 25)]) #loc function example


# In[9]:


# selecting 0th, 2nd, 4th, and 7th index rows
print(data.iloc[[0, 2, 4, 7]])


# In[11]:


#Q.6 Difference between Series and dataframe ?

#One of the main differences between DataFrame and Series is that a DataFrame can have multiple columns, while a Series can only have one.


# In[14]:


#Q.7 How do you check missing values using pandas ?
#In order to check missing values in Pandas DataFrame, we use a function isnull() and notnull()

dict = {'First Score':[100, 90, np.nan, 95], 
        'Second Score': [30, 45, 56, np.nan], 
        'Third Score':[np.nan, 40, 80, 98]}  
df = pd.DataFrame(dict)    
df.isnull()


# In[15]:


#Q.8 How do you  drop missing values using pandas ?
#In order to drop a null values from a dataframe, we used dropna() function this function drop Rows/Columns of datasets with Null values in different ways.

dict = {'First Score':[100, 90, np.nan, 95], 
        'Second Score': [30, 45, 56, np.nan], 
        'Third Score':[np.nan, 40, 80, 98]}  
df = pd.DataFrame(dict)    
df.isnull()


# In[16]:


df.dropna()


# In[22]:


dict = {'First Score':[100, 90, np.nan, 95], 
        'Second Score': [30, 45, 56, np.nan], 
        'Third Score':[np.nan, 40, 80, 98]}  
df = pd.DataFrame(dict)    
df.isnull()


# In[23]:


df.dropna(how = 'all')


# In[26]:


dict = {'First Score':[100, 90, np.nan, 95], 
        'Second Score': [30, 45, 56, np.nan], 
        'Third Score':[np.nan, 40, 80, 98]}  
df = pd.DataFrame(dict)    
df


# In[27]:


df.dropna(axis=1)


# In[28]:


#Q.9 Prove that dataframe is mutable ?

#Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)
#lists , sets and dictionaries are mutable .


# In[30]:


import pandas as pd
import numpy as np


# In[40]:


data=np.arange(1,11,2)
a=pd.Series(data , index=[1,2,3,4,5])
print(a)


# In[41]:


data=(11,22,33,44,55)
a=pd.Series(data, index=[1,2,3,4,5])
print(a)


# In[42]:


#Q.10 How  can you rename a column using pandas ? 
#Using rename() function
#One way of renaming the columns in a Pandas Dataframe is by using the rename() function. This method is quite useful when we need to rename some selected columns because we need to specify information only for the columns which are to be renamed. 
import pandas as pd
import numpy as np


# In[49]:


Preferable_Food={
    "Name":['Pavan','Varu','Aadi','Abhi'],
    "Favourite_Food":['Pav_Bhaji', 'Idli','Dosa','Khushka'],
    "Restaurant":['Sagar_Cafe','Asha_Tiffins','Laxmi_Hotel','Niyaaz']
}
df =pd.DataFrame(Preferable_Food)
df


# In[58]:


df.rename(columns={'Name' : 'Customers'} , inplace=True)
print("Modifying first column", df.columns)


# In[ ]:




