#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import joblib


# In[11]:


def main():
    html_temp="""
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black";text-align:center">Heart Disease Prediction</h2>
    </div>"""
    
    st.markdown(html_temp,unsafe_allow_html=True)
    model = joblib.load('model_joblib_heart')
    p1 = st.slider("Enter Your Age",18,100)
    s1=st.selectbox("Sex",("Male","Female"))
    if s1=="Male":
        p2=1
    else:
        p2=0
    p3 =st.number_input("Enter Value of CP",step=1)
    p4 =st.number_input("Enter Value of trestbps",step=1)
    p5 =st.number_input("Enter Value of chol",step=1)
    p6 =st.number_input("Enter Value of fbs",step=1)
    p7 =st.number_input("Enter Value of restecg",step=1)
    p8 =st.number_input("Enter Value of thalach",step=1)
    p9 =st.number_input("Enter Value of exang",step=1)
    p10 =st.number_input("Enter Value of oldpeak")
    p11 =st.number_input("Enter Value of slope",step=1)
    p12 =st.number_input("Enter Value of ca",step=1)
    p13=st.number_input("Enter Value of thal",step=1)
    
    if st.button('Predict'):
        prediction = model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        prob=model.predict_proba([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])[:,1]
        percentage_of_heart_disease = prob * 100
        st.success(f"{int(percentage_of_heart_disease[0])}%")


#if __name__ =='_main_':
main()
    
        


# In[ ]:




