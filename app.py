@st.cache_resource(show_spinner=False)
def fit_model(data):
    features = [c for c in PREFERRED_FEATURES if c in data.columns]
    if not features:
        features = [
            c for c in data.columns
            if c != "price" and (pd.api.types.is_numeric_dtype(data[c]) or data[c].nunique() <= 100)
        ][:10]
    if not features:
        raise ValueError("No usable feature columns found. Include bike details such as model, year, kilometres, owner, mileage or power.")

    X = data[features].copy()
    y = data["price"].copy()

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="median")), numeric))
    if categorical:
        transformers.append(("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), categorical))

    preprocess = ColumnTransformer(transformers)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        preprocess,
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }
    return features, numeric, categorical, model, metrics

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your bike dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        data = clean_dataset(pd.read_csv(uploaded_file))
        st.sidebar.success(f"Uploaded: {len(data):,} valid records")
        dataset_label = uploaded_file.name
    except Exception as exc:
        st.sidebar.error(f"Could not use this CSV: {exc}")
        st.stop()
else:
    try:
        with st.spinner("Loading sample bike dataset..."):
            data = load_default_data()
        st.sidebar.info("Using the built-in sample dataset. Upload a CSV above to use your own data.")
        dataset_label = "Built-in sample dataset"
    except Exception as exc:
        st.error(f"Could not load the sample dataset: {exc}")
        st.info("Upload a CSV file in the Dataset section of the sidebar. It must contain a price column.")
        st.stop()

try:
    # The uploaded file's contents determine the fitted model.
    features, numeric, categorical, model, metrics = fit_model(data)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

st.caption(f"Dataset: {dataset_label}  |  Records: {len(data):,}  |  Features: {len(features)}")
st.subheader("Enter Bike Details")
st.write("Predicted price updates automatically whenever you change an input.")

input_values = {}
left, right = st.columns(2)
for index, col in enumerate(features):
    target = left if index % 2 == 0 else right
    label = col.replace("_", " ").title()
    with target:
        if col in numeric:
            series = pd.to_numeric(data[col], errors="coerce").dropna()
            default = float(series.median()) if not series.empty else 0.0
            minimum = float(series.min()) if not series.empty else 0.0
            maximum = float(series.max()) if not series.empty else max(default * 2, 1.0)
            if minimum == maximum:
                maximum = minimum + 1.0
            input_values[col] = st.number_input(
                label, min_value=minimum, max_value=maximum,
                value=min(max(default, minimum), maximum), key=f"input_{col}"
            )
        else:
            options = sorted(data[col].dropna().astype(str).unique().tolist())
            if options:
                input_values[col] = st.selectbox(label, options, key=f"input_{col}")
            else:
                input_values[col] = st.text_input(label, key=f"input_{col}")

input_frame = pd.DataFrame([input_values], columns=features)
for col in categorical:
    input_frame[col] = input_frame[col].astype(str)

predicted_price = float(model.predict(input_frame)[0])
st.success(f"Estimated Bike Price: ₹{predicted_price:,.0f}")
st.caption("This is a model-based estimate from the selected dataset, not a guaranteed market price.")

st.subheader("Model Evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"₹{metrics['MAE']:,.0f}")
m2.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
m3.metric("R² Score", f"{metrics['R2']:.3f}")
st.caption("Evaluation metrics are calculated on a held-out test split. They change when you upload a different dataset, not when you change one prediction input.")

with st.expander("Preview dataset"):
    st.dataframe(data.head(100), use_container_width=True)
    st.download_button(
        "Download dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="used_bike_dataset.csv",
        mime="text/csv"
    )
