#!/usr/bin/env python
# coding: utf-8

# # Building the Web Appplication

# This web app will be built using the streamlit library. A lot of this code is from @dataprofessor's Penguin Classification App (https://github.com/dataprofessor/penguins-heroku/blob/master/penguins-app.py). Thank you Chanin!

# In[1]:


#import all packages necessary
import streamlit as st
import pickle
import pandas as pd
import numpy as np


# In[2]:


#create a basic heading and intro
st.write("""

# Early Diabetes Detection Web App

The underlying model was developed using a dataset from https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.

This app is created by Joel Zcharia
""")


# In[3]:


#create a sidebar to ask questions from
st.sidebar.header('Please answer these questions!')


# In[32]:


#create an input for all the features we need on the sidebar
def user_input_features():
        Age = st.sidebar.slider('What is your age?', 0,50,100)
        Gender = st.sidebar.selectbox('What is your Gender?',('Male','Female'), key='a')
        Polyuria = st.sidebar.selectbox('Have you been peeing very frequently or abnormallly large amounts?',('Yes','No'), key='b')
        Polydipsia = st.sidebar.selectbox('Do you feel thirsty even after drinking water?',('Yes','No'), key='c')
        sudden_weight_loss = st.sidebar.selectbox('Have you loss weight unexplainably?',('Yes','No'), key='d')
        weakness = st.sidebar.selectbox('Do you feel unexplainably tired or fatigued?',('Yes','No'), key='e')
        Polyphagia = st.sidebar.selectbox('Do you have the urge to eat even after a meal?',('Yes','No'), key='f')
        Genital_thrush = st.sidebar.selectbox('Have you had a yeast infection around your genitals recently?',('Yes','No'), key='g')
        visual_blurring = st.sidebar.selectbox('Have your experieced blurring of your vision?',('Yes','No'), key='i')
        Itching = st.sidebar.selectbox('Has your body been itching more frequently?',('Yes','No'), key='j')              
        Irritability = st.sidebar.selectbox('Have you experienced increased irritability?',('Yes','No'), key='k')       
        delayed_healing = st.sidebar.selectbox('Are wounds taking longer to heal?',('Yes','No'), key='l')      
        partial_paresis = st.sidebar.selectbox('Do your muscles feel weaker or do you experience partial paralysis?',('Yes','No'), key='m')   
        muscle_stiffness = st.sidebar.selectbox('Do you feel your muscles feel stiffer than normal?',('Yes','No'), key='n')
        Alopecia = st.sidebar.selectbox('Have you experienced hair loss on any part of your body?',('Yes','No'), key='o')
        Obesity = st.sidebar.selectbox('Is your body mass index above 29?',('Yes','No'), key='p')
        data = {'Age': Age,
                'Gender': Gender,
                'Polyuria': Polyuria,
                'Polydipsia': Polydipsia,
                'sudden_weight_loss': sudden_weight_loss,
                'weakness': weakness,
                'Polyphagia': Polyphagia,
                'Genital_thrush': Genital_thrush,
                'visual_blurring': visual_blurring,
                'Itching': Itching,
                'Irritability': Irritability,
                'delayed_healing': delayed_healing,
                'partial_paresis': partial_paresis,
                'muscle_stiffness': muscle_stiffness,
                'Alopecia': Alopecia,
                'Obesity': Obesity}
        features = pd.DataFrame(data, index=[0])
        return features

input_df = user_input_features()


# In[33]:


#adding the collected information back into the original dataset
diabetes_data = pd.read_csv('diabetes_data_upload.csv')

#delete the target variable
diabetes = diabetes_data.drop(columns=['class'])

#put the collected information with original dataset
df = pd.concat([input_df,diabetes],axis=0)


# In[34]:


#Code from Pratik Mukherjee (https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering)
encode = ['Gender','Polyuria','Polydipsia','sudden_weight_loss','weakness','Polyphagia','Genital_thrush',
          'visual_blurring','Itching','Irritability','delayed_healing','partial_paresis','muscle_stiffness',
          'Alopecia','Obesity']

#we are going to get dummies for the features that need to be encoded
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]


# In[15]:


#pulling in previously saved decision tree model in pickle format
model_dt = pickle.load(open('diabetes_dt.pkl', 'rb'))


# In[35]:


# Apply model to make predictions
prediction = model_dt.predict(df)
prediction_proba = model_dt.predict_proba(df)


# In[36]:


#displaying the results
st.subheader('Based on your answers, the prediction is...')
diabetes_classification = np.array(['No','Yes'])
st.write(diabetes_classification[prediction])


# In[ ]:




