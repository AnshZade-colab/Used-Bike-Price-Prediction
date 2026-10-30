import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Bike Price Prediction", page_icon="🏍️")

st.title("🏍️ Used Bike Price Prediction")
st.write("Enter the bike details to estimate its price.")


# Load dataset automatically
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/KrutinBhat/Used-Bike-Prices-Analysis/main/bikes.csv"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = pd.read_csv(BytesIO(response.content))
    data.columns = data.columns.str.strip().str.lower()

    for col in ["model_year", "kms_driven", "mileage", "power", "price"]:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+)")[0],
                errors="coerce"
            )

    data = data.drop_duplicates()
    data = data.dropna(subset=["price"])
    data = data[data["price"] > 0]

    return data


# Train model
@st.cache_resource
def train_model():
    data = load_data()

    features = [
        col for col in [
            "model_name", "model_year", "kms_driven",
            "owner", "location", "mileage", "power"
        ]
        if col in data.columns
    ]

    if not features:
        raise ValueError("No expected input features found in the dataset.")

    X = data[features]
    y = data["price"]

    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numerical]

    preprocessing = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        ), numerical),

        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return data, features, numerical, categorical, model, metrics


# Handle data and model errors
try:
    data, features, numerical, categorical, model, metrics = train_model()
except Exception as e:
    st.error(f"Unable to load the dataset or train the model: {e}")
    st.stop()


# User input form
st.subheader("Enter Bike Details")

with st.form("bike_form"):
    values = {}
    columns = st.columns(2)

    for i, col in enumerate(features):
        with columns[i % 2]:
            label = col.replace("_", " ").title()

            if col in numerical:
                median = data[col].median()
                values[col] = st.number_input(
                    label,
                    value=float(median) if pd.notna(median) else 0.0
                )
            else:
                options = sorted(
                    data[col].dropna().astype(str).unique().tolist()
                )

                if options:
                    values[col] = st.selectbox(label, options)
                else:
                    values[col] = st.text_input(label)

    submit = st.form_submit_button("Predict Bike Price")


# Display prediction
if submit:
    input_data = pd.DataFrame([values], columns=features)

    for col in categorical:
        input_data[col] = input_data[col].astype(str)

    price = model.predict(input_data)[0]

    st.success(f"Estimated Bike Price: ₹{price:,.0f}")
    st.caption(
        "This estimate is based on historical listings and may not "
        "represent the current market price."
    )


# Model evaluation
st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"₹{metrics['MAE']:,.0f}")
col2.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
col3.metric("R² Score", f"{metrics['R2']:.3f}")

st.write("Number of records:", len(data))
st.write("Features used:", features)
