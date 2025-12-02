import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from math import sqrt

# -------------------- Config --------------------
DEFAULT_DATA_PATH = "Walmart_Sales_Dataset_Cleaned.csv"

st.set_page_config(
    page_title="Walmart Sales – Predictive Analytics (Cloud)",
    layout="wide"
)

st.title("📈 Walmart Predictive Analytics – ML Model Comparison (Cloud)")
st.write(
    """
    This Streamlit Cloud app compares **multiple ML models** on the Walmart sales dataset.

    **Included Models:**
    - Linear Regression  
    - Random Forest  
    - Gradient Boosting  
    - Support Vector Regressor (SVR)  

    You can either **upload your own cleaned Walmart CSV** or let the app use the
    built-in sample: `Walmart_Sales_Dataset_Cleaned.csv`.
    """
)

# -------------------- Functions --------------------
@st.cache_data
def load_data(file_or_path):
    """
    Load data either from an uploaded file object or a local path.
    """
    df = pd.read_csv(file_or_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_weekly(df, date_col, target_col):
    """
    Aggregate to weekly sales.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    ts = (
        df.groupby(pd.Grouper(key=date_col, freq="W"))[target_col]
          .sum()
          .reset_index()
    )
    ts.rename(columns={target_col: "Sales", date_col: "Date"}, inplace=True)
    ts = ts.sort_values("Date")
    return ts


def feature_engineer(df):
    """
    Add calendar features + time index.
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["TimeIndex"] = np.arange(len(df))
    return df


def train_and_evaluate(df, model_name, test_size):
    """
    Time-based train/test split to avoid leakage and n_samples errors.
    """
    df = df.copy()
    X = df[["Year", "Month", "WeekOfYear", "TimeIndex"]]
    y = df["Sales"].values

    n_samples = len(df)

    if n_samples < 3:
        raise ValueError(
            f"Not enough weekly data points after aggregation: {n_samples}. "
            "You need at least 3 weeks of data to train + test."
        )

    # Time-based split index
    split_idx = int(np.floor(n_samples * (1 - test_size)))

    # Ensure at least 1 train and 1 test sample
    if split_idx < 1:
        split_idx = 1
    if split_idx >= n_samples:
        split_idx = n_samples - 1

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Choose model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "SVR":
        model = SVR(kernel="rbf")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Fit & predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": sqrt(mean_squared_error(y_test, preds)),
        "MAPE %": np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100,
        "R² Score": r2_score(y_test, preds),
    }

    results_df = pd.DataFrame({
        "Date": df.loc[X_test.index, "Date"],
        "Actual": y_test,
        "Predicted": preds
    })

    return model, metrics, results_df


def forecast(model, df, weeks):
    """
    Simple forward forecast using the same calendar features.
    """
    df = df.copy()
    last_date = df["Date"].max()
    last_index = df["TimeIndex"].iloc[-1]

    future_dates = pd.date_range(last_date, periods=weeks + 1, freq="W")[1:]
    future = pd.DataFrame({"Date": future_dates})

    future["Year"] = future["Date"].dt.year
    future["Month"] = future["Date"].dt.month
    future["WeekOfYear"] = future["Date"].dt.isocalendar().week.astype(int)
    future["TimeIndex"] = np.arange(last_index + 1, last_index + 1 + weeks)

    features = future[["Year", "Month", "WeekOfYear", "TimeIndex"]]
    future["Forecast"] = model.predict(features)

    return future


# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Cleaned Walmart CSV (optional)",
    type=["csv"],
    help="If you do not upload a file, the app will use Walmart_Sales_Dataset_Cleaned.csv from the repo."
)

model_selected = st.sidebar.selectbox(
    "Choose ML Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR"]
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100.0

forecast_weeks = st.sidebar.slider(
    "Forecast Weeks Ahead",
    min_value=4,
    max_value=24,
    value=8,
    step=2
)

run_btn = st.sidebar.button("🚀 Run Model")


# -------------------- Load Data (Upload or Default) --------------------
data_source_label = ""

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        data_source_label = "Uploaded CSV"
    else:
        df = load_data(DEFAULT_DATA_PATH)
        data_source_label = f"Default CSV: {DEFAULT_DATA_PATH}"
except FileNotFoundError:
    st.error(
        f"Could not find the default data file `{DEFAULT_DATA_PATH}`.\n\n"
        "Please upload a cleaned Walmart dataset CSV using the sidebar."
    )
    st.stop()

st.success(f"Data loaded from **{data_source_label}**")

# -------------------- Main Layout --------------------
st.subheader("📄 Raw Data (sample)")
st.dataframe(df.head())

# Column Selection
st.subheader("1️⃣ Select Columns")
date_col = st.selectbox(
    "Select Date Column",
    df.columns.tolist(),
    index=df.columns.get_loc("Date") if "Date" in df.columns else 0
)

target_col = st.selectbox(
    "Select Target (Sales) Column",
    df.columns.tolist(),
    index=df.columns.get_loc("Total_Sales") if "Total_Sales" in df.columns else len(df.columns) - 1
)

# Weekly aggregation
st.subheader("2️⃣ Weekly Aggregated Sales")
ts = prepare_weekly(df, date_col, target_col)
st.write(f"Number of weekly data points after aggregation: **{len(ts)}**")
st.line_chart(ts.set_index("Date")["Sales"])

ts = feature_engineer(ts)

# -------------------- Run Model --------------------
if run_btn:
    st.subheader(f"3️⃣ Model Results: {model_selected}")

    try:
        model, metrics, results = train_and_evaluate(ts, model_selected, test_size)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics['MAE']:.2f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col3.metric("MAPE %", f"{metrics['MAPE %']:.2f}")
        col4.metric("R² Score", f"{metrics['R² Score']:.3f}")

        st.write("📈 Actual vs Predicted (Test Period)")
        st.dataframe(results)
        st.line_chart(results.set_index("Date")[["Actual", "Predicted"]])

        # Forecast
        st.subheader("4️⃣ Forecast")
        forecast_df = forecast(model, ts, forecast_weeks)
        st.dataframe(forecast_df)
        st.line_chart(forecast_df.set_index("Date")["Forecast"])

    except ValueError as e:
        st.error(str(e))
else:
    st.info("Configure settings in the sidebar and click **🚀 Run Model** to train and forecast.")
