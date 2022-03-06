# -*- coding: utf-8 -*-
import numpy as np
import pickle
import streamlit as st

model = pickle.load(open("C:/Users/renan/Google Drive/notebooks/beginning/diabetes_pred_with_streamlit/diabetes_model.sav", "rb"))

def diabetes_prediction(input_data):
    # taking a sample from dataset: 1,189,60,23,846,30.1,0.398,59 -> 1 = diabetics

    #input_data to numpy array and reshape it
    input_data = np.asarray(input_data).reshape(1, -1)

    pred = model.predict(input_data)
    if pred[0] == 0:
        return 'The person isn\'t diabetics'
    else:
        return 'The person is diabetics'

def main():
    # giving a title for the web app
    st.title('Diabetes Prediction web app')

    # getting the input data from the user
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")

    # making the prediction
    diagnosis = ''

    # creating a button to make the prediction occurs
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()
