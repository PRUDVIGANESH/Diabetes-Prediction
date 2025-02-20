import pickle
import os
import streamlit as st
import pandas as pd
from sklearn. metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

diabetes_model_path ="diabetes_model.sav"
diabetes_model=pickle.load(open(diabetes_model_path,'rb'))

st.title('Diabetes prediction using ML')

col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col3:
    BloodPressure = st.text_input('Blood Pressure value')

with col1:
    SkinThickness = st.text_input('Skin Thickness value')

with col2:
    Insulin = st.text_input('Insulin Level')

with col3:
    BMI = st.text_input('BMI value')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

with col2:
    Age = st.text_input('Age of the Person')

# Prediction result
diab_diagnosis = ''

# Creating a button for Prediction
if st.button('Diabetes Test Result'):
    try:
        # Convert input to float
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        
        # Make prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)
    
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
        if st.button('Show Model Accuracy'):
            test_data=pd.read_csv("diabetes.csv")
            x_test=test_data.drop(columns=['Outcome'])
            y_test=test_data['Outcome']

            y_pred=diabetes_model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            st.write(f"model accuracy:{accuracy*100:.2f}%")
