#!/usr/bin/env python
# coding: utf-8

# # 1. Import Dataset and Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('C:/Users/Vivek Jagwani/Downloads/heart.csv')


# # 2. Applying basic pandas functions  

# In[3]:


data.columns


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data_dup = data.duplicated().any()


# In[9]:


data_dup


# In[10]:


data = data.drop_duplicates()


# In[11]:


data_dup = data.duplicated().any()


# In[12]:


data_dup


# # Data Preprocessing

# In[13]:


cate_val = []
cont_val = []
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[14]:


cate_val


# In[15]:


cont_val


# # 6 Encoding Categorical Data

# In[16]:


cate_val


# In[17]:


data['cp'].unique()


# In[18]:


cate_val.remove('sex')
cate_val.remove('target')
data=pd.get_dummies(data,columns=cate_val,drop_first=True)


# In[19]:


data.head()


# #  Feature Scaling

# In[20]:


data.head()


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[23]:


data.head()


# # Splitting the dataset into training and test

# In[24]:


X = data.drop('target',axis=1)


# In[25]:


y = data['target']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=42)


# In[28]:


y_test


# # Support Vector Machine

# In[29]:


from sklearn import svm


# In[30]:


svm = svm.SVC()


# In[31]:


svm.fit(X_train,y_train)


# In[32]:


y_pred2 = svm.predict(X_test)


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


accuracy_score(y_test,y_pred2)


# # KNeighbors Classifer

# In[43]:


from sklearn.neighbors import KNeighborsClassifier


# In[44]:


knn = KNeighborsClassifier(n_neighbors=4)


# In[45]:


knn.fit(X_train,y_train)


# In[48]:


print("X_train type:", type(X_train))
print("X_train shape:", X_train.shape)
print("y_train type:", type(y_train))
print("y_train shape:", y_train.shape)
y_pred3 = knn.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred3)


# In[47]:


score = []

for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))


# In[ ]:


score


# In[ ]:


plt.plot(score)
plt.xlabel("K Value")
plt.ylabel("Acc")
plt.show()


# # Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


log = LogisticRegression()
log.fit(X_train,y_train)


# In[41]:


y_pred1 = log.predict(X_test)


# In[42]:


accuracy_score(y_test,y_pred1)


# # Random Forest Classifeir

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_pred5= rf.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred5)


# In[ ]:





# In[ ]:





# In[ ]:





# # Accuracy of models

# In[ ]:


final_data = pd.DataFrame({'Models':['SVM','LR','RF'],
                          'ACC':[
                                accuracy_score(y_test,y_pred2)*100,
#                                 accuracy_score(y_test,y_pred3)*100,
                                accuracy_score(y_test,y_pred1)*100,
                                   accuracy_score(y_test,y_pred5)*100]})


# In[ ]:


final_data


# In[ ]:





# In[ ]:


# sns.barplot(final_data['Models'],final_data['ACC'])


# In[ ]:


data=pd.read_csv('C:/Users/Vivek Jagwani/Downloads/heart.csv')


# In[ ]:


data = data.drop_duplicates()


# In[ ]:





# In[ ]:


X=data.drop('target',axis=1)
y=data['target']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf1 = RandomForestClassifier()
rf1.fit(X,y)


# # Prediction on new Data

# In[ ]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])


# In[ ]:


new_data


# In[ ]:


p = rf1.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")


# # 16. Save Model using Joblib

# In[ ]:


import joblib


# In[ ]:


joblib.dump(rf1,'model_joblib_heart')


# In[ ]:


model = joblib.load('model_joblib_heart')


# In[ ]:


model.predict(new_data)


# In[ ]:


data.tail()


# # GUI

# In[ ]:


from tkinter import *
import joblib


# In[ ]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:





# In[ ]:




