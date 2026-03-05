import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

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



uploaded_file = st.file_uploader("Upload Weather Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, skiprows=15)

    # Create date column
    df["date"] = pd.to_datetime(df[["YEAR","MO","DY"]].rename(
        columns={"YEAR":"year","MO":"month","DY":"day"}
    ))

    df = df.sort_values("date")

    st.success("Dataset uploaded successfully")

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

        # LSTM sequence
        seq = df.iloc[idx-WINDOW_SIZE:idx][features]

        if len(seq) != WINDOW_SIZE:
            st.error("Not enough historical data to create 90-day sequence.")
            st.stop()

        seq_scaled = scaler_X.transform(seq)
        seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, len(features))

        lstm_pred_scaled = lstm_model.predict(seq_scaled)
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)

        # Known features
        known = df.iloc[idx][known_features].values

        hybrid_input = np.concatenate(
            [known, lstm_pred.flatten()]
        ).reshape(1, -1)

        pred_max = xgb_max.predict(hybrid_input)
        pred_min = xgb_min.predict(hybrid_input)

        st.success(f"Predicted Max Temp: {pred_max[0]:.2f} °C")
        st.success(f"Predicted Min Temp: {pred_min[0]:.2f} °C")