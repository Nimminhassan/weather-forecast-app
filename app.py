import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import requests
from datetime import datetime
end_date = datetime.today().strftime("%Y%m%d")

WINDOW_SIZE = 90

st.title("Hybrid Weather Forecasting System")
st.write("LSTM + XGBoost Temperature Prediction")

# Load models
@st.cache_resource
def load_models():
    lstm_model = load_model("optimized_lstm_model.keras")
    xgb_max = joblib.load("final_xgb_model_max.joblib")
    xgb_min = joblib.load("final_xgb_model_min.joblib")
    scaler_X = joblib.load("scaler_X.joblib")
    scaler_y = joblib.load("scaler_y.joblib")
    features = joblib.load("features.joblib")
    known_features = joblib.load("known_features.joblib")

    return lstm_model, xgb_max, xgb_min, scaler_X, scaler_y, features, known_features
lstm_model, xgb_max, xgb_min, scaler_X, scaler_y, features, known_features = load_models()

st.subheader("Select Location")

location_dict = {
    "Thiruvananthapuram": (8.5308, 76.9296),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946)
}

selected_city = st.selectbox("Choose City", list(location_dict.keys()))

latitude, longitude = location_dict[selected_city]


@st.cache_data
def download_nasa_data():

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,PRECTOTCORR,RH2M,PS,WS10M",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": 20100101,
        "end": end_date,
        "format": "JSON"
    }

    r = requests.get(url, params=params)
    data = r.json()["properties"]["parameter"]

    df = pd.DataFrame(data)

    df.index = pd.to_datetime(df.index, format="%Y%m%d")

    df = df.reset_index().rename(columns={"index": "date"})

    df["YEAR"] = df["date"].dt.year
    df["MO"] = df["date"].dt.month
    df["DY"] = df["date"].dt.day

    return df


st.subheader("Choose Data Source")

data_source = st.radio(
    "Select data source",
    ["Upload CSV", "Download NASA Data"]
)


# uploaded_file = st.file_uploader("Upload Weather Dataset (CSV)", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file, skiprows=15)

#     # Create date column
#     df["date"] = pd.to_datetime(df[["YEAR","MO","DY"]].rename(
#         columns={"YEAR":"year","MO":"month","DY":"day"}
#     ))

#     df = df.sort_values("date")

#     st.success("Dataset uploaded successfully")

#     # Feature Engineering
#     df["month_sin"] = np.sin(2*np.pi*df["MO"]/12)
#     df["month_cos"] = np.cos(2*np.pi*df["MO"]/12)

#     df["day_sin"] = np.sin(2*np.pi*df["DY"]/31)
#     df["day_cos"] = np.cos(2*np.pi*df["DY"]/31)

#     df["T2M_MAX_lag_365"] = df["T2M_MAX"].shift(365)
#     df["T2M_MIN_lag_365"] = df["T2M_MIN"].shift(365)

#     df["T2M_MAX_roll_mean_7"] = df["T2M_MAX"].rolling(7).mean()
#     df["T2M_MAX_roll_std_7"] = df["T2M_MAX"].rolling(7).std()

#     df = df.dropna().reset_index(drop=True)

#     prediction_date = st.date_input("Select prediction date")

#     if st.button("Predict Temperature"):

#         idx_list = df.index[df["date"] == pd.to_datetime(prediction_date)]

#         if len(idx_list) == 0:
#             st.error("Selected date not found in dataset")
#             st.stop()

#         idx = idx_list[0]

#         if idx < WINDOW_SIZE:
#             st.error("Not enough historical data for prediction.")
#             st.stop()

#         # LSTM sequence
#         seq = df.iloc[idx-WINDOW_SIZE:idx][features]

#         if len(seq) != WINDOW_SIZE:
#             st.error("Not enough historical data to create 90-day sequence.")
#             st.stop()

#         seq_scaled = scaler_X.transform(seq)
#         seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, len(features))

#         lstm_pred_scaled = lstm_model.predict(seq_scaled)

#         # Known features
#         known = df.iloc[idx][known_features].values

#         hybrid_input = np.concatenate(
#             [known, lstm_pred_scaled.flatten()]
#         ).reshape(1, -1)

#         pred_max_scaled = xgb_max.predict(hybrid_input)
#         pred_min_scaled = xgb_min.predict(hybrid_input)

#         pred_max = scaler_y.inverse_transform(
#             np.array([[pred_max_scaled[0], pred_min_scaled[0]]])
#         )[0][0]

#         pred_min = scaler_y.inverse_transform(
#             np.array([[pred_max_scaled[0], pred_min_scaled[0]]])
#         )[0][1]

#         st.write("Actual Max Temp:", df.iloc[idx]["T2M_MAX"])
#         st.write("Actual Min Temp:", df.iloc[idx]["T2M_MIN"])

#         st.success(f"Predicted Max Temp: {pred_max:.2f} °C")
#         st.success(f"Predicted Min Temp: {pred_min:.2f} °C")

# -----------------------------
# DATA LOADING SECTION
# -----------------------------

df = None

if data_source == "Upload CSV":

    uploaded_file = st.file_uploader("Upload Weather Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        st.session_state["df"] = pd.read_csv(uploaded_file, skiprows=15)
        df = st.session_state["df"]

elif data_source == "Download NASA Data":

    if st.button("Download NASA Weather Data"):
        st.session_state["df"] = download_nasa_data()
        st.success("NASA weather data downloaded successfully")

    if "df" in st.session_state:
        df = st.session_state["df"]

# -----------------------------
# MAIN PIPELINE (UNCHANGED)
# -----------------------------

if df is not None:

    # Create date column
    df["date"] = pd.to_datetime(df[["YEAR","MO","DY"]].rename(
        columns={"YEAR":"year","MO":"month","DY":"day"}
    ))

    df = df.sort_values("date")

    # Feature Engineering
    df["month_sin"] = np.sin(2*np.pi*df["MO"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["MO"]/12)

    df["day_sin"] = np.sin(2*np.pi*df["DY"]/31)
    df["day_cos"] = np.cos(2*np.pi*df["DY"]/31)

    df["T2M_MAX_lag_365"] = df["T2M_MAX"].shift(365)
    df["T2M_MIN_lag_365"] = df["T2M_MIN"].shift(365)

    df["T2M_MAX_roll_mean_7"] = df["T2M_MAX"].rolling(7).mean()
    df["T2M_MAX_roll_std_7"] = df["T2M_MAX"].rolling(7).std()

    df = df.dropna().reset_index(drop=True)

    prediction_date = st.date_input("Select prediction date")

    if st.button("Predict Temperature"):

        idx_list = df.index[df["date"] == pd.to_datetime(prediction_date)]

        if len(idx_list) == 0:
            st.error("Selected date not found in dataset")
            st.stop()

        idx = idx_list[0]

        if idx < WINDOW_SIZE:
            st.error("Not enough historical data for prediction.")
            st.stop()

        # -----------------------------
        # LSTM INPUT SEQUENCE
        # -----------------------------

        seq = df.iloc[idx-WINDOW_SIZE:idx][features]

        if len(seq) != WINDOW_SIZE:
            st.error("Not enough historical data to create 90-day sequence.")
            st.stop()

        seq_scaled = scaler_X.transform(seq)
        seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, len(features))

        lstm_pred_scaled = lstm_model.predict(seq_scaled)

        # -----------------------------
        # HYBRID MODEL INPUT
        # -----------------------------

        known = df.iloc[idx][known_features].values

        hybrid_input = np.concatenate(
            [known, lstm_pred_scaled.flatten()]
        ).reshape(1, -1)

        pred_max_scaled = xgb_max.predict(hybrid_input)
        pred_min_scaled = xgb_min.predict(hybrid_input)

        pred_max = scaler_y.inverse_transform(
            np.array([[pred_max_scaled[0], pred_min_scaled[0]]])
        )[0][0]

        pred_min = scaler_y.inverse_transform(
            np.array([[pred_max_scaled[0], pred_min_scaled[0]]])
        )[0][1]

        # -----------------------------
        # RESULTS
        # -----------------------------

        st.write("Actual Max Temp:", df.iloc[idx]["T2M_MAX"])
        st.write("Actual Min Temp:", df.iloc[idx]["T2M_MIN"])

        st.success(f"Predicted Max Temp: {pred_max:.2f} °C")
        st.success(f"Predicted Min Temp: {pred_min:.2f} °C")

        # -----------------------------
        # TEMPERATURE TREND GRAPH
        # -----------------------------

        st.subheader("Temperature Trend (Last 90 Days + Prediction)")

        history = df.iloc[idx-WINDOW_SIZE:idx][["date","T2M_MAX","T2M_MIN"]].copy()

        prediction_row = pd.DataFrame({
            "date":[prediction_date],
            "T2M_MAX":[pred_max],
            "T2M_MIN":[pred_min]
        })

        chart_df = pd.concat([history, prediction_row])

        chart_df = chart_df.set_index("date")

        st.line_chart(chart_df)