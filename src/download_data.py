"""Downloads daily ERA5 weather for 9 Sri Lankan cities from the Open-Meteo Historical Weather API.

Weather data by Open-Meteo.com (CC BY 4.0) - ERA5 reanalysis by ECMWF / Copernicus.
Output: data/raw/sl_weather_era5_daily.csv  (one row per city per day)
"""
import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "raw", "sl_weather_era5_daily.csv")

# city: (latitude, longitude). Zones are worked out later from the rainfall data.
CITIES = {
    "Colombo":      (6.93, 79.85),
    "Galle":        (6.03, 80.22),
    "Ratnapura":    (6.68, 80.40),
    "Kandy":        (7.29, 80.63),
    "Nuwara Eliya": (6.97, 80.78),
    "Badulla":      (6.99, 81.06),
    "Anuradhapura": (8.31, 80.40),
    "Trincomalee":  (8.59, 81.21),
    "Jaffna":       (9.66, 80.02),
}

START, END = "2010-01-01", "2026-11-15"     # ERA5 is published with a ~5-day delay

DAILY = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean", "dew_point_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(city, lat, lon):
    params = {
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "daily": ",".join(DAILY), "models": "era5", "timezone": "Asia/Colombo",
    }
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            js = r.json()
            df = pd.DataFrame(js["daily"])
            df.insert(0, "city", city)
            df["grid_lat"], df["grid_lon"], df["elevation"] = js["latitude"], js["longitude"], js["elevation"]
            return df
        print(f"  {city}: HTTP {r.status_code} - retrying in {10 * (attempt + 1)} s")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed for {city}: {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = []
    for city, (lat, lon) in CITIES.items():
        df = fetch(city, lat, lon)
        print(f"{city:13s} {len(df):6d} days  grid=({df.grid_lat[0]}, {df.grid_lon[0]})  elevation={df.elevation[0]} m")
        frames.append(df)
        time.sleep(2)                       # small pause between requests
    data = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    data.to_csv(OUT, index=False)
    print("saved", OUT, data.shape)
