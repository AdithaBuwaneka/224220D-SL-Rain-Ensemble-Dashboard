"""Loads the saved models and makes next-day rain predictions.
Used by the Streamlit dashboard (app.py) and the last cell of the notebook."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.features import make_features, fourier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RAW_CSV = os.path.join(ROOT, "data", "raw", "sl_weather_era5_daily.csv")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# same locations as src/download_data.py
CITY_LOCATIONS = {
    "Colombo": (6.93, 79.85), "Galle": (6.03, 80.22), "Ratnapura": (6.68, 80.40),
    "Kandy": (7.29, 80.63), "Nuwara Eliya": (6.97, 80.78), "Badulla": (6.99, 81.06),
    "Anuradhapura": (8.31, 80.40), "Trincomalee": (8.59, 81.21), "Jaffna": (9.66, 80.02),
}
DAILY = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "relative_humidity_2m_mean",
         "dew_point_2m_mean", "pressure_msl_mean", "cloud_cover_mean", "precipitation_sum", "precipitation_hours",
         "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
         "et0_fao_evapotranspiration"]
CLASSIFIERS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def load_models():
    info = json.load(open(os.path.join(MODEL_DIR, "model_info.json")))
    models = {name: joblib.load(os.path.join(MODEL_DIR, f"{file}.joblib")) for name, file in info["files"].items()}
    return info, models


def fetch_live(info, past_days=60):
    """Daily weather for every city: the last `past_days` days, today and tomorrow (Open-Meteo forecast API)."""
    cities = list(CITY_LOCATIONS)
    params = {
        "latitude": ",".join(str(CITY_LOCATIONS[c][0]) for c in cities),
        "longitude": ",".join(str(CITY_LOCATIONS[c][1]) for c in cities),
        "daily": ",".join(DAILY), "past_days": past_days, "forecast_days": 2, "timezone": "Asia/Colombo",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for city, js in zip(cities, r.json()):
        d = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
        d.insert(0, "city", city)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["elevation"] = df["city"].map(lambda c: info["city_coords"][c]["elevation"])   # same as in training
    return df


def load_era5_history():
    """Offline backup: the downloaded ERA5 data."""
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    return df.drop(columns=["rain_sum", "grid_lat", "grid_lon"])


def arima_tomorrow(city_days, info, city):
    """One-step ARIMA forecast of log(1 + rain) for the day after the last row, with the saved parameters."""
    g = city_days.sort_values("date")
    y = np.log1p(g["precipitation_sum"].values)
    model = SARIMAX(y, exog=fourier(g["date"]), order=tuple(info["arima_order"]), trend="c")
    res = model.filter(np.array(info["arima_params"][city]))
    next_day = g["date"].iloc[-1] + pd.Timedelta(days=1)
    return float(res.forecast(1, exog=fourier([next_day]))[0])


def predict(history, info, models, day):
    """Predict rain on day+1 for every city, using weather up to and including `day`."""
    day = pd.Timestamp(day)
    hist = history[history["date"] <= day]
    feats = make_features(hist, info["zone"], info["cities"])
    today = feats[feats["date"] == day].set_index("city").loc[info["cities"]]
    X = today[info["features"]]

    out = pd.DataFrame(index=today.index)
    out["zone"] = today["zone"]
    for name in CLASSIFIERS:
        out[name] = models[name].predict_proba(X)[:, 1]
    out["Soft voting"] = out[info["vote_models"]].mean(axis=1)

    today = today.copy()
    today["arima_tomorrow"] = [arima_tomorrow(hist[hist["city"] == c], info, c) for c in today.index]
    out["Hybrid: ARIMA + RF"] = models["Hybrid: ARIMA + RF"].predict_proba(today[info["features_hybrid"]])[:, 1]

    w = info["rf_ann_weights"]
    log_mm = w["intercept"] + w["rf"] * models["RF regressor"].predict(X) + w["ann"] * models["ANN regressor"].predict(X)
    out["expected_mm"] = np.clip(np.expm1(log_mm), 0, None)

    out["threshold"] = out["zone"].map(info["zone_threshold"])
    out["rain_tomorrow"] = out["Soft voting"] >= out["threshold"]
    out["today_rain_mm"] = today["precipitation_sum"]
    out["today_humidity"] = today["relative_humidity_2m_mean"]
    out["lat"] = [CITY_LOCATIONS[c][0] for c in out.index]
    out["lon"] = [CITY_LOCATIONS[c][1] for c in out.index]
    return out, X
