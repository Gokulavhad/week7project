import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("ML Model Deployment App")
st.write("Input your data to get predictions")

# Input fields
age = st.slider("Age", 18, 100, 25)
salary = st.number_input("Monthly Salary", 1000, 100000, 30000)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

# Map categorical inputs if needed
edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
input_data = pd.DataFrame([[age, salary, edu_map[education]]], columns=["Age", "Salary", "Education_Level"])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")

# Visualizations (optional with sample data)
if st.checkbox("Show Sample Data & Visualization"):
    data = pd.read_csv("sample_data.csv")
    st.dataframe(data.head())

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, ax=ax)
    st.pyplot(fig)
