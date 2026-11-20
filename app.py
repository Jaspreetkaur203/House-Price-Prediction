
import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load trained ML model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Main heading
st.title("🏠 House Price Prediction System")

st.subheader("Developed by Jaspreet Kaur")

st.write(
    "A Machine Learning application that predicts "
    "real estate prices using property details."
)

st.divider()

# Input section
st.header("Enter Property Details")

date = st.number_input(
    "Transaction Date",
    value=2013.5
)

age = st.number_input(
    "House Age (Years)",
    min_value=0.0,
    value=10.0
)

distance = st.number_input(
    "Distance to Nearest MRT Station",
    min_value=0.0,
    value=500.0
)

stores = st.number_input(
    "Number of Convenience Stores",
    min_value=0,
    value=5
)

latitude = st.number_input(
    "Latitude",
    value=24.98
)

longitude = st.number_input(
    "Longitude",
    value=121.54
)

# Prediction button
if st.button("Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([[
        date,
        age,
        distance,
        stores,
        latitude,
        longitude
    ]], columns=[
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude"
    ])

    prediction = model.predict(input_data)

    st.success(
        f"Estimated House Price: {prediction[0]:.2f}"
    )

    st.info(
        "This prediction is based on the UCI Taiwan "
        "Real Estate Valuation dataset. It is an "
        "educational estimate, not an actual market quotation."
    )

# Footer
st.divider()

st.caption(
    "Created by Jaspreet Kaur | Machine Learning Project"
)

st.caption(
    "Technologies: Python | Pandas | Scikit-learn | Streamlit"
)
