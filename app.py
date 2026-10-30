
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Used Bike Price Analysis and Prediction",
    page_icon="🏍️",
    layout="wide"
)

st.title("🏍️ Used Bike Price Analysis and Prediction")
st.write(
    "Analyze used bike data and predict estimated resale prices using Machine Learning."
)

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/KrutinBhat/Used-Bike-Prices-Analysis/main/bikes.csv"

PREFERRED_FEATURES = [
    "model_name",
    "model_year",
    "kms_driven",
    "owner",
    "location",
    "mileage",
    "power"
]


def clean_dataset(raw):

    data = raw.copy()

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    aliases = {
        "bike_name": "model_name",
        "name": "model_name",
        "model": "model_name",
        "year": "model_year",
        "manufacturing_year": "model_year",
        "km_driven": "kms_driven",
        "kilometers_driven": "kms_driven",
        "seller_type": "owner",
        "ownership": "owner",
        "city": "location",
        "mileage_kmpl": "mileage",
        "engine_power": "power",
        "bhp": "power",
        "selling_price": "price",
        "bike_price": "price"
    }

    data = data.rename(
        columns={
            col: aliases[col]
            for col in data.columns
            if col in aliases
        }
    )

    if "price" not in data.columns:
        raise ValueError(
            "Dataset must contain a price column."
        )

    for col in [
        "model_year",
        "kms_driven",
        "mileage",
        "power",
        "price"
    ]:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+)")[0],
                errors="coerce"
            )

    if "model_year" in data.columns:
        data["model_year"] = data["model_year"].round()

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.drop_duplicates()
    data = data.dropna(subset=["price"])
    data = data[data["price"] > 0]

    if len(data) < 10:
        raise ValueError(
            "At least 10 valid records are required."
        )

    return data


@st.cache_data(show_spinner=False)
def load_default_data():

    response = requests.get(
        DEFAULT_DATA_URL,
        timeout=30
    )

    response.raise_for_status()

    raw = pd.read_csv(
        BytesIO(response.content)
    )

    return clean_dataset(raw)


@st.cache_resource(show_spinner=False)
def train_model(data):

    features = [
        col for col in PREFERRED_FEATURES
        if col in data.columns
    ]

    if not features:

        features = [
            col for col in data.columns
            if col != "price"
            and (
                pd.api.types.is_numeric_dtype(data[col])
                or data[col].nunique() <= 100
            )
        ][:10]

    if not features:
        raise ValueError(
            "No suitable input features found."
        )

    X = data[features].copy()
    y = data["price"].copy()

    numerical = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = [
        col for col in features
        if col not in numerical
    ]

    transformers = []

    if numerical:

        numerical_pipeline = make_pipeline(
            SimpleImputer(strategy="median")
        )

        transformers.append(
            ("num", numerical_pipeline, numerical)
        )

    if categorical:

        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        )

        transformers.append(
            ("cat", categorical_pipeline, categorical)
        )

    preprocessing = ColumnTransformer(
        transformers
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(
            y_test, predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(
            y_test, predictions
        )
    }

    return (
        features,
        numerical,
        categorical,
        model,
        metrics
    )


# DATASET UPLOAD

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your bike dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = clean_dataset(
            pd.read_csv(uploaded_file)
        )

        dataset_name = uploaded_file.name

        st.sidebar.success(
            f"{len(data):,} records uploaded successfully"
        )

    except Exception as e:

        st.sidebar.error(
            f"Dataset error: {e}"
        )

        st.stop()

else:

    try:

        with st.spinner("Loading sample dataset..."):

            data = load_default_data()

        dataset_name = "Sample Dataset"

        st.sidebar.info(
            "Using built-in sample dataset."
        )

    except Exception as e:

        st.error(
            f"Unable to load sample dataset: {e}"
        )

        st.info(
            "Please upload a valid CSV dataset using the sidebar."
        )

        st.stop()


# MODEL TRAINING

try:

    (
        features,
        numerical,
        categorical,
        model,
        metrics
    ) = train_model(data)

except Exception as e:

    st.error(
        f"Model training failed: {e}"
    )

    st.stop()


st.divider()

# DATASET OVERVIEW

st.subheader("Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Records",
    f"{len(data):,}"
)

c2.metric(
    "Features Used",
    len(features)
)

c3.metric(
    "Dataset",
    dataset_name
)

st.divider()

# BIKE INPUT FORM

st.subheader("Enter Bike Details")

st.write(
    "Select the bike details and click Predict Bike Price."
)

input_values = {}

with st.form("bike_prediction_form"):

    left, right = st.columns(2)

    for index, col in enumerate(features):

        container = (
            left if index % 2 == 0 else right
        )

        label = col.replace(
            "_", " "
        ).title()

        with container:

            if col in numerical:

                series = pd.to_numeric(
                    data[col],
                    errors="coerce"
                ).dropna()

                if col == "model_year":

                    default = (
                        int(round(series.median()))
                        if not series.empty else 2020
                    )

                    minimum = (
                        int(np.floor(series.min()))
                        if not series.empty else 1980
                    )

                    maximum = (
                        int(np.ceil(series.max()))
                        if not series.empty else 2026
                    )

                    if minimum == maximum:
                        maximum = minimum + 1

                    input_values[col] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(
                            max(default, minimum),
                            maximum
                        ),
                        step=1,
                        format="%d",
                        key=f"input_{col}"
                    )

                else:

                    default = (
                        float(series.median())
                        if not series.empty else 0.0
                    )

                    minimum = (
                        float(series.min())
                        if not series.empty else 0.0
                    )

                    maximum = (
                        float(series.max())
                        if not series.empty
                        else max(default * 2, 1.0)
                    )

                    if minimum == maximum:
                        maximum = minimum + 1.0

                    input_values[col] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(
                            max(default, minimum),
                            maximum
                        ),
                        key=f"input_{col}"
                    )

            else:

                options = sorted(
                    data[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if options:

                    input_values[col] = st.selectbox(
                        label,
                        options,
                        key=f"input_{col}"
                    )

                else:

                    input_values[col] = st.text_input(
                        label,
                        key=f"input_{col}"
                    )

    predict_button = st.form_submit_button(
        "Predict Bike Price",
        type="primary",
        use_container_width=True
    )


# PREDICTION

if predict_button:

    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    for col in categorical:
        input_data[col] = input_data[col].astype(str)

    predicted_price = float(
        model.predict(input_data)[0]
    )

    st.session_state["predicted_price"] = predicted_price

    st.session_state["prediction_dataset"] = dataset_name


if (
    "predicted_price" in st.session_state
    and st.session_state.get("prediction_dataset") == dataset_name
):

    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Estimated Bike Price: ₹{st.session_state['predicted_price']:,.0f}"
    )

    st.caption(
        "This estimate is based on historical listings "
        "and may not represent the current market price."
    )


# MODEL EVALUATION

st.divider()

st.subheader("Model Performance Evaluation")

m1, m2, m3 = st.columns(3)

m1.metric(
    "MAE",
    f"₹{metrics['MAE']:,.0f}"
)

m2.metric(
    "RMSE",
    f"₹{metrics['RMSE']:,.0f}"
)

m3.metric(
    "R² Score",
    f"{metrics['R2']:.3f}"
)

st.caption(
    "Performance metrics are calculated using an "
    "80:20 train-test split."
)


# DATASET PREVIEW

st.divider()

st.subheader("Dataset Preview")

with st.expander("View Dataset"):

    st.dataframe(
        data.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Dataset",
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="used_bike_dataset.csv",
        mime="text/csv"
    )


st.divider()

st.caption(
    "Used Bike Price Analysis and Prediction | "
    "Statistical Inference and Machine Learning Project"
)
