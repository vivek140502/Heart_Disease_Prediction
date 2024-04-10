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
    p3 =st.number_input("Enter Value of CP(chest pain type (4 values))",step=1)
    p4 =st.number_input("Enter Value of trestbps(resting blood pressure)",step=1)
    p5 =st.number_input("Enter Value of chol(serum cholestoral in mg/dl)",step=1)
    p6 =st.number_input("Enter Value of fbs(fasting blood sugar > 120 mg/dl)",step=1)
    p7 =st.number_input("Enter Value of restecg(resting electrocardiographic results (values 0,1,2))",step=1)
    p8 =st.number_input("Enter Value of maximum heart rate achieved ",step=1)
    p9 =st.number_input("Enter Value of exang(exercise induced angina)",step=1)
    p10 =st.number_input("Enter Value of oldpeak(oldpeak = ST depression induced by exercise relative to rest)")
    p11 =st.number_input("Enter Value of slope(the slope of the peak exercise ST segment)",step=1)
    p12 =st.number_input("Enter Value of ca(number of major vessels (0-3) colored by flourosopy)",step=1)
    p13=st.number_input("Enter Value of thal(thal: 0 = normal; 1 = fixed defect; 2 = reversable defect)",step=1)
    
    if st.button('Predict'):
        prediction = model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        prob = model.predict_proba([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]])[:, 0]
        prob_1 = model.predict_proba([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]])[:, 1]
        res = max(prob, prob_1)
        percentage_of_heart_disease = res * 100
        st.success(f"Probability of person that he/she can suffer from Heart Disease is {int(percentage_of_heart_disease[0])}%")


#if __name__ =='_main_':
main()
    
        


# In[ ]:




