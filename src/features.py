"""Features for next-day rain. Used by the notebook and the dashboard."""
import numpy as np
import pandas as pd

WEATHER = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "relative_humidity_2m_mean",
           "dew_point_2m_mean", "pressure_msl_mean", "cloud_cover_mean", "precipitation_sum", "precipitation_hours",
           "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
SEASONS = ["First inter-monsoon", "South-west monsoon", "Second inter-monsoon", "North-east monsoon"]
ZONES = ["Wet", "Intermediate", "Dry"]


def season_of(month):
    if month in (3, 4): return SEASONS[0]
    if 5 <= month <= 9: return SEASONS[1]
    if month in (10, 11): return SEASONS[2]
    return SEASONS[3]


def make_features(df, zone_map, cities):
    """One row per city per day, using only that day and earlier days."""
    out = []
    for city, g in df.sort_values("date").groupby("city", sort=False):
        g = g.copy()
        r = g["precipitation_sum"]
        g["temp_range"] = g["temperature_2m_max"] - g["temperature_2m_min"]
        g["rain_lag1"], g["rain_lag2"] = r.shift(1), r.shift(2)
        g["rain_3d"], g["rain_7d"] = r.rolling(3).sum(), r.rolling(7).sum()
        g["wet_days_7d"] = (r >= 1).rolling(7).sum()
        for col, name in [("pressure_msl_mean", "pressure"), ("relative_humidity_2m_mean", "humidity"),
                          ("temperature_2m_mean", "temp")]:
            g[f"{name}_change"] = g[col].diff()
        wd = np.deg2rad(g["wind_direction_10m_dominant"])
        g["wind_dir_sin"], g["wind_dir_cos"] = np.sin(wd), np.cos(wd)
        m, doy = g["date"].dt.month, g["date"].dt.dayofyear
        g["month_sin"], g["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
        g["doy_sin"], g["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        g["season"] = m.map(season_of)
        g["zone"] = zone_map[city]
        out.append(g)
    f = pd.concat(out, ignore_index=True)
    for s in SEASONS: f[f"season_{s}"] = (f["season"] == s).astype(int)
    for z in ZONES: f[f"zone_{z}"] = (f["zone"] == z).astype(int)
    for c in cities: f[f"city_{c}"] = (f["city"] == c).astype(int)
    return f


def feature_columns(cities):
    return (WEATHER + ["elevation", "temp_range", "rain_lag1", "rain_lag2", "rain_3d", "rain_7d", "wet_days_7d",
                       "pressure_change", "humidity_change", "temp_change", "wind_dir_sin", "wind_dir_cos",
                       "month_sin", "month_cos", "doy_sin", "doy_cos"]
            + [f"season_{s}" for s in SEASONS] + [f"zone_{z}" for z in ZONES] + [f"city_{c}" for c in cities])


def fourier(dates, K=2):
    """Yearly seasonal terms used by the ARIMA model."""
    doy = pd.to_datetime(pd.Series(dates)).dt.dayofyear.values
    return np.column_stack([f(2 * np.pi * k * doy / 365.25) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def add_targets(f):
    """Rain on the next day for each city."""
    f = f.sort_values(["city", "date"]).copy()
    f["rain_tomorrow_mm"] = f.groupby("city")["precipitation_sum"].shift(-1)
    f["rain_tomorrow_1mm"] = (f["rain_tomorrow_mm"] >= 1).astype("float").where(f["rain_tomorrow_mm"].notna())
    f["rain_tomorrow_5mm"] = (f["rain_tomorrow_mm"] >= 5).astype("float").where(f["rain_tomorrow_mm"].notna())
    return f
