# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.svm import LinearSVR 

# Load the data
data=pd.read_csv("ineuron_file.csv")

# Define the features and target
X = data[["Close", "High", "Low"]].values
y = data["Open"].values

# Fit the linear regression model
SVR=LinearSVR().fit(X, y)

# Create a Streamlit app
st.title("Stock Market Open Price Predictor")

# Add a header
st.write("The area Stock market is something which is with lots of ups and downs. It can change in no time. Therefore we can use machine learning technique to identify market changes earlier than possible with traditional investment models. Using machine Learning models we can automate the investment predictions process. Input the Closing Price, High Price, and Low Price of the previous day:")

# Get user input
closing_price = st.number_input("Close:", min_value=0, max_value=1000000, value=100)
high_price = st.number_input("High:", min_value=0, max_value=1000000, value=100)
low_price = st.number_input("Low:", min_value=0, max_value=1000000, value=100)

# Make a prediction

X_new = np.array([[closing_price, high_price, low_price]])

if st.button("Predict"):
    prediction = SVR.predict(X_new)
    # Show the prediction
    st.write("The predicted Open Price for the next day is: Rs.", round(prediction[0], 2), "approximately")