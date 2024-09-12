# Install necessary packages before running the app (do not include this in the code):
pip install streamlit scikit-learn pandas numpy

import streamlit as st
import pandas as pd
import pickle

# Load the trained model (make sure the file path is correct)
filename = r"lasso_model.sav"

# Try loading the model
try:
    loaded_model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")

# Create a title for your app
st.title("Boston Housing Price Prediction")

# Create input fields for the features
crim = st.number_input("CRIM (per capita crime rate by town)", min_value=0.0)
zn = st.number_input("ZN (proportion of residential land zoned for lots over 25,000 sq.ft.)", min_value=0.0)
indus = st.number_input("INDUS (proportion of non-retail business acres per town)", min_value=0.0)
chas = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
nox = st.number_input("NOX (nitric oxides concentration)", min_value=0.0)
rm = st.number_input("RM (average number of rooms per dwelling)", min_value=0.0)
age = st.number_input("AGE (proportion of owner-occupied units built prior to 1940)", min_value=0.0)
dis = st.number_input("DIS (weighted distances to five Boston employment centres)", min_value=0.0)
rad = st.number_input("RAD (index of accessibility to radial highways)", min_value=0.0)
tax = st.number_input("TAX (full-value property-tax rate per $10,000)", min_value=0.0)
ptratio = st.number_input("PTRATIO (pupil-teacher ratio by town)", min_value=0.0)
b = st.number_input("B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)", min_value=0.0)
lstat = st.number_input("LSTAT (lower status of the population)", min_value=0.0)

# Create a button to trigger prediction
if st.button("Predict Price"):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'crim': [crim],
        'zn': [zn],
        'indus': [indus],
        'chas': [chas],
        'nox': [nox],
        'rm': [rm],
        'age': [age],
        'dis': [dis],
        'rad': [rad],
        'tax': [tax],
        'ptratio': [ptratio],
        'b': [b],
        'lstat': [lstat]
    })

    # Make the prediction
    try:
        prediction = loaded_model.predict(input_data)
        # Display the prediction
        st.write("Predicted Price:", prediction[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
