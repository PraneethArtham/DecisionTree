import streamlit as st
import pickle
import pandas as pd

# LOAD MODEL

with open("models/decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# LOAD FEATURE COLUMNS

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# PAGE TITLE

st.title("California Housing Price Prediction")

st.write("Decision Tree Regressor")

# USER INPUTS

longitude = st.slider("Longitude", -125.0, -113.0, -120.0)

latitude = st.slider("Latitude", 32.0, 42.0, 37.0)

housing_median_age = st.slider("Housing Median Age", 1, 60, 20)

total_rooms = st.slider("Total Rooms", 2, 10000, 2000)

total_bedrooms = st.slider("Total Bedrooms", 1, 5000, 400)

population = st.slider("Population", 1, 10000, 1000)

households = st.slider("Households", 1, 5000, 500)

median_income = st.slider("Median Income", 0.0, 20.0, 5.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity", ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
)

# CREATE INPUT DATAFRAME

input_dict = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
}

# HANDLE CATEGORICAL INPUT

for col in feature_columns:
    if "ocean_proximity_" in col:
        input_dict[col] = 0

selected_col = f"ocean_proximity_{ocean_proximity}"

if selected_col in feature_columns:
    input_dict[selected_col] = 1

# CREATE DATAFRAME

input_df = pd.DataFrame([input_dict])

input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# PREDICTION

if st.button("Predict House Price"):
    prediction = model.predict(input_df)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
