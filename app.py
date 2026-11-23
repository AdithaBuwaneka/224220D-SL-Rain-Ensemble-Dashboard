"""Will it rain tomorrow? Ensemble learning dashboard for Sri Lanka's climatic zones.
Run with:  streamlit run app.py"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from src.predict import CLASSIFIERS, fetch_live, load_era5_history, load_models, predict

st.set_page_config(page_title="Sri Lanka rain tomorrow", page_icon=":material/rainy:", layout="wide")
st.markdown("""
<style>
  /* narrower sidebar: it only holds the weather data choice and the theme buttons */
  section[data-testid="stSidebar"][aria-expanded="true"] { width: 260px !important; min-width: 260px !important; }
  /* the tab bar stays at the top while the page scrolls (just below Streamlit's 60 px header) */
  /* the tab bar sits in the top row (where the empty header space was) and stays there while scrolling */
  header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
  header[data-testid="stHeader"] button, header[data-testid="stHeader"] a,
  header[data-testid="stHeader"] [data-testid="stToolbar"] { pointer-events: auto; }
  [data-testid="stMainBlockContainer"] { padding-top: 10px !important; }
  div:has(> [data-baseweb="tab-list"]) { position: sticky; top: 0; z-index: 999989; padding-top: 10px;
                                         box-shadow: 0 1px 0 rgba(128,128,128,.28); }
  header[data-testid="stHeader"] { z-index: 999990 !important; }   /* menu buttons stay above the tab bar */
</style>
""", unsafe_allow_html=True)
# colours checked for colour-blind readers in light and dark mode (same set as the notebook)
ZONE_COLORS = {"Wet": "#2a78d6", "Intermediate": "#199e70", "Dry": "#d95926"}
FAMILY = {"Baseline": "Baseline or single tree", "Single tree": "Baseline or single tree", "Bagging": "Bagging",
          "Boosting": "Boosting", "Voting": "Voting, stacking or hybrid", "Stacking": "Voting, stacking or hybrid",
          "Hybrid": "Voting, stacking or hybrid"}
FAMILY_COLORS = {"Bagging": "#2a78d6", "Boosting": "#d95926", "Voting, stacking or hybrid": "#199e70",
                 "Baseline or single tree": "#b5b4ad"}
NICE = {
    "temperature_2m_max": "Highest temperature today", "temperature_2m_min": "Lowest temperature today",
    "temperature_2m_mean": "Mean temperature today", "relative_humidity_2m_mean": "Humidity today",
    "dew_point_2m_mean": "Dew point today", "pressure_msl_mean": "Air pressure today",
    "cloud_cover_mean": "Cloud cover today", "precipitation_sum": "Rain today (mm)",
    "precipitation_hours": "Hours of rain today", "wind_speed_10m_max": "Highest wind speed today",
    "wind_gusts_10m_max": "Strongest wind gust today", "shortwave_radiation_sum": "Sunshine today (solar radiation)",
    "et0_fao_evapotranspiration": "Evaporation today", "elevation": "Height above sea level",
    "temp_range": "Temperature range today", "rain_lag1": "Rain yesterday", "rain_lag2": "Rain 2 days ago",
    "rain_3d": "Rain in the last 3 days", "rain_7d": "Rain in the last 7 days", "wet_days_7d": "Rainy days in the last 7",
    "pressure_change": "Pressure change since yesterday", "humidity_change": "Humidity change since yesterday",
    "temp_change": "Temperature change since yesterday", "wind_dir_sin": "Wind direction (east or west)",
    "wind_dir_cos": "Wind direction (north or south)", "month_sin": "Time of year (month, sine)",
    "month_cos": "Time of year (month, cosine)", "doy_sin": "Time of year (day, sine)", "doy_cos": "Time of year (day, cosine)",
    "city": "City", "zone": "Zone", "season": "Monsoon season",
}


def nice(name):
    for prefix, label in [("city_", "City: "), ("zone_", "Zone: "), ("season_", "Season: ")]:
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return NICE.get(name, name)
MODEL_COLS = CLASSIFIERS + ["Soft voting", "Hybrid: ARIMA + RF"]


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


@st.cache_data(ttl=3600, show_spinner=False)
def get_live(_info):
    return fetch_live(_info)


@st.cache_data(show_spinner=False)
def get_era5():
    return load_era5_history()


@st.cache_data(show_spinner=False)
def get_test_predictions():
    return pd.read_csv("data/processed/test_predictions.csv", parse_dates=["date"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(_history, _info, _models, day, source_key):
    """Prediction for one day; cached, so changing the city or the tab does not recompute it."""
    return predict(_history, _info, _models, day)


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# Lucide icons (MIT licence, lucide.dev), drawn as inline SVG
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" '
       'stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{p}</svg>')
ICON_PATHS = {
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/>'
           '<path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/>'
           '<path d="m19.07 4.93-1.41 1.41"/>',
    "cloud_sun": '<path d="M12 2v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="M20 12h2"/><path d="m19.07 4.93-1.41 1.41"/>'
                 '<path d="M15.947 12.65a4 4 0 0 0-5.925-4.128"/><path d="M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z"/>',
    "drizzle": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M8 19v1"/><path d="M8 14v1"/>'
               '<path d="M16 19v1"/><path d="M16 14v1"/><path d="M12 21v1"/><path d="M12 16v1"/>',
    "rain": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v6"/><path d="M8 14v6"/>'
            '<path d="M12 16v6"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/>'
               '<line x1="12" x2="12" y1="17" y2="21"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
}
WX_COLOR = {"sun": "#f08c00", "cloud_sun": "#868e96", "drizzle": "#4dabf7", "rain": "#1971c2"}
WX_NAME = {"sun": "dry", "cloud_sun": "cloudy", "drizzle": "light rain", "rain": "rain"}


def icon(name, size=20, color="currentColor"):
    return SVG.format(s=size, c=color, p=ICON_PATHS[name])


def icon_for(mm=None, chance=None, rain=None, expected=None):
    """Weather icon: from the real rain (mm) for past days, or from the prediction for tomorrow."""
    if mm is not None:
        return "rain" if mm >= 10 else "drizzle" if mm >= 1 else "cloud_sun" if mm >= 0.2 else "sun"
    if rain:
        return "rain" if expected >= 5 else "drizzle"
    return "cloud_sun" if chance >= 0.4 else "sun"


def weather_map_html(history, pred, day):
    """Leaflet + OpenStreetMap weather map: last 7 days of real rain, then tomorrow's prediction (play / slider)."""
    cities = list(pred.index)
    frames = []
    for k in range(6, -1, -1):
        d = day - pd.Timedelta(days=k)
        mm = history[history["date"] == d].set_index("city")["precipitation_sum"].reindex(cities).fillna(0)
        frames.append({"label": d.strftime("%a %d %b"), "items": [
            {"icon": icon_for(mm=float(mm[c])), "text": f"{mm[c]:.0f} mm",
             "pop": f"<b>{c}</b> ({pred.loc[c, 'zone']} zone)<br>{d.strftime('%d %b %Y')}: {mm[c]:.1f} mm of rain"}
            for c in cities]})
    nxt = (day + pd.Timedelta(days=1)).strftime("%d %b %Y")
    frames.append({"label": "Tomorrow", "items": [
        {"icon": icon_for(chance=r["Soft voting"], rain=r["rain_tomorrow"], expected=r["expected_mm"]),
         "text": f"{r['Soft voting']:.0%}",
         "pop": (f"<b>{c}</b> ({r['zone']} zone)<br>{nxt}: <b>{'Rain' if r['rain_tomorrow'] else 'No rain'}</b><br>"
                 f"chance {r['Soft voting']:.0%} (cut-off {r['threshold']:.2f})<br>expected {r['expected_mm']:.1f} mm")}
        for c, r in pred.iterrows()]})
    points = [{"name": c, "lat": float(pred.loc[c, "lat"]), "lon": float(pred.loc[c, "lon"])} for c in cities]
    data = json.dumps({"frames": frames, "points": points})
    icons_json = json.dumps({k: '<span class="badge">' + icon(k, 30, WX_COLOR[k]) + "</span>" for k in WX_COLOR})
    legend = " &nbsp; ".join(icon(k, 16, WX_COLOR[k]) + " " + WX_NAME[k] for k in WX_COLOR)
    play_html = icon("play", 14) + " Play"
    pause_html = icon("pause", 14) + " Pause"
    return """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
  #map { height: 560px; border-radius: 8px; }
  .wx { text-align: center; line-height: 1; display: flex; flex-direction: column; align-items: center; gap: 3px; }
  .wx .ic { display: inline-block; }
  .badge { display: inline-flex; background: rgba(255,255,255,.92); border-radius: 50%; padding: 4px;
           box-shadow: 0 1px 3px rgba(0,0,0,.3); }
  .legend svg, #play svg { vertical-align: -3px; }
  .wx .lb { display: inline-block; font-size: 11px; background: rgba(255,255,255,.88); color: #1d3557; border-radius: 6px; padding: 1px 5px;
            white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,.25); }
  .rain .ic { animation: drop 1.4s ease-in-out infinite; }
  .sun .ic { animation: spin 9s linear infinite; }
  @keyframes drop { 0%,100% { transform: translateY(0); } 50% { transform: translateY(5px); } }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  #bar { display: flex; align-items: center; gap: 10px; padding: 10px 2px 0; color: #888; font-size: 14px; }
  #play { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #bbb; background: #f0f2f6; color: #222; border-radius: 6px; padding: 4px 12px; cursor: pointer; }
  #slider { flex: 1; accent-color: #2a78d6; }
  #day { min-width: 120px; font-weight: 600; color: #2a78d6; }
  .legend { background: rgba(255,255,255,.9); padding: 4px 8px; border-radius: 6px; font-size: 13px; color: #1d3557; }
</style>
<div id="map"></div>
<div id="bar"><button id="play">PLAY_HTML</button><input id="slider" type="range" min="0" step="1"><span id="day"></span></div>
<script>
const D = DATA; const ICONS = ICONS_JSON; const LEGEND = LEGEND_JSON; const PLAY = PLAY_JSON; const PAUSE = PAUSE_JSON;
const SHIFT = {'Kandy': [-30, -22], 'Nuwara Eliya': [-6, 40], 'Badulla': [46, -18], 'Colombo': [-40, 0], 'Ratnapura': [-28, 30]};
const map = L.map('map', {scrollWheelZoom: false, zoomSnap: 0.25}).setView([7.75, 80.7], 7.5);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 12,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
const legend = L.control({position: 'topright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.innerHTML = LEGEND; return d; };
legend.addTo(map);
const markers = D.points.map(p => L.marker([p.lat, p.lon]).addTo(map));
const slider = document.getElementById('slider'), day = document.getElementById('day'), play = document.getElementById('play');
slider.max = D.frames.length - 1;
function show(i) {
  const f = D.frames[i];
  f.items.forEach((it, j) => {
    const cls = (it.icon === 'rain' || it.icon === 'drizzle') ? 'rain' : (it.icon === 'sun' ? 'sun' : '');
    const off = SHIFT[D.points[j].name] || [0, 0];
    markers[j].setIcon(L.divIcon({className: 'wx ' + cls, iconSize: [100, 66], iconAnchor: [50 - off[0], 19 - off[1]],
      html: '<span class="ic">' + ICONS[it.icon] + '</span><span class="lb">' + D.points[j].name + ' ' + it.text + '</span>'}));
    markers[j].bindPopup(it.pop);
  });
  slider.value = i; day.textContent = f.label;
}
let timer = null;
play.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; play.innerHTML = PLAY; return; }
  let i = 0; show(i); play.innerHTML = PAUSE;
  timer = setInterval(() => { i += 1; if (i >= D.frames.length) { clearInterval(timer); timer = null; play.innerHTML = PLAY; return; } show(i); }, 1100);
};
slider.oninput = () => show(+slider.value);
show(D.frames.length - 1);
</script>
""".replace("DATA", data).replace("PLAY_HTML", play_html).replace("ICONS_JSON", icons_json) \
        .replace("LEGEND_JSON", json.dumps(legend)).replace("PLAY_JSON", json.dumps(play_html)) \
        .replace("PAUSE_JSON", json.dumps(pause_html))


with st.sidebar, st.spinner("Loading the models ..."):
    info, models = get_models()

# ---------------------------------------------------------------- data for the prediction
st.sidebar.title("Sri Lanka rain tomorrow")
st.sidebar.caption("CM3720 hands-on project by Wimalasuriya A.G.A.B. (224220D)")
st.sidebar.subheader("Controls")
source = st.sidebar.radio("Weather data", ["Live (Open-Meteo, today)", "Past day (ERA5 data)"],
                          help="Changes the Tomorrow and Why this prediction? tabs.")

live_failed = False
if source.startswith("Live"):
    try:
        with st.sidebar, st.spinner("Getting today's weather from Open-Meteo ..."):
            history = get_live(info)
        today = pd.Timestamp.now(tz="Asia/Colombo").normalize().tz_localize(None)
        day = today if today in set(history["date"]) else history["date"].max() - pd.Timedelta(days=1)
        note = (f"Weather up to **{day.date()}** (today's values include the forecast for the rest of the day). "
                f"Prediction is for **{(day + pd.Timedelta(days=1)).date()}**.")
    except Exception as e:                       # no internet or API down: fall back to the saved data
        live_failed = True
        st.sidebar.warning(f"Live data not available ({type(e).__name__}). Showing the last ERA5 day instead.")
if not source.startswith("Live") or live_failed:
    with st.sidebar, st.spinner("Loading the saved ERA5 data ..."):
        history = get_era5()
    last = history["date"].max() - pd.Timedelta(days=1)
    day = pd.Timestamp(st.sidebar.date_input("Day (weather up to this day)", value=last.date(),
                                             min_value=pd.Timestamp("2010-01-08").date(), max_value=last.date()))
    note = (f"ERA5 weather up to **{day.date()}**, prediction for **{(day + pd.Timedelta(days=1)).date()}**. "
            "The saved models were trained on all years, so past days are a demo, not a fair test.")

# one city for every tab that shows a single city (instead of a separate city box in each tab)
CITY_LIST = sorted(info["cities"])
st.session_state.setdefault("city", "Colombo")          # one city shared by the Why and Test years tabs


def city_picker(key):
    """City dropdown placed in a tab; every copy shows and changes the same shared city."""
    st.session_state[key] = st.session_state["city"]
    return st.selectbox("City", CITY_LIST, key=key,
                        on_change=lambda: st.session_state.update(city=st.session_state[key]))


city = st.session_state["city"]

is_live = source.startswith("Live") and not live_failed
loaded_at = pd.Timestamp.now(tz="Asia/Colombo").strftime("%d %b %Y, %H:%M") + " (Sri Lanka time)"
with st.sidebar, st.spinner("Updating the prediction ..."):     # in the sidebar, so the tabs keep their place
    pred, X = get_prediction(history, info, models, day, ("live" if is_live else "era5", str(day.date())))
next_day = day + pd.Timedelta(days=1)
actual = history[history["date"] == next_day].set_index("city")["precipitation_sum"]

THEME_BUTTONS = """
<style>
  body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
  .t { display: flex; gap: 4px; border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 3px; width: fit-content; }
  .t button { display: inline-flex; border: 0; background: transparent; padding: 6px 14px; border-radius: 6px; cursor: pointer; }
  .t button:hover { background: rgba(128,128,128,.2); }
  .t button.on { background: rgba(42,120,214,.22); }
</style>
<div class="t">
  <button data-name="Light" title="Light mode" aria-label="Light mode">SUN</button>
  <button data-name="Dark" title="Dark mode" aria-label="Dark mode">MOON</button>
  <button data-name="Use system setting" title="Same as my computer" aria-label="Same as my computer">MONITOR</button>
</div>
<script>
  // The buttons use Streamlit's own theme setting (menu, Settings, theme), so the theme changes with no page reload.
  const P = window.parent, D = P.document;
  const key = "stActiveTheme-" + P.location.pathname + "-v1";
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  async function waitFor(find, ms = 3000) {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(40)) { const x = find(); if (x) return x; }
    return null;
  }
  const mine = "sl-rain-theme";                        // remembers which button was clicked last
  function active() {
    try { const m = P.localStorage.getItem(mine); if (m) return m; } catch (e) {}
    return "Light";                                     // the light default comes from config.toml
  }
  function mark() { document.querySelectorAll(".t button").forEach(b => b.classList.toggle("on", b.dataset.name === active())); }
  async function setTheme(name) {
    const hide = D.createElement("style");              // keep the menu and dialog invisible while they are used
    hide.textContent = '[data-baseweb="popover"], [role="dialog"], [data-baseweb="modal"] { opacity: 0 !important; }';
    D.head.appendChild(hide);
    try {
      const menu = D.querySelector('[data-testid="stMainMenu"] button') || D.querySelector('[data-testid="stMainMenu"]');
      menu.click();
      const item = await waitFor(() => [...D.querySelectorAll('li, [role="menuitem"]')].find(e => e.innerText.trim() === "Settings"));
      item.click();
      const dlg = await waitFor(() => D.querySelector('[role="dialog"]'));
      const select = await waitFor(() => dlg.querySelector('[data-baseweb="select"] div'));
      select.click();
      const opt = await waitFor(() => [...D.querySelectorAll('[role="option"]')].find(o => o.innerText.trim() === name));
      opt.click();
      try { P.localStorage.setItem(mine, name); } catch (e) {}
      await sleep(120);
      const close = await waitFor(() => D.querySelector('[role="dialog"] [aria-label="Close"]'), 1000);
      if (close) close.click();
    } catch (e) {
      console.warn("theme switch failed", e);
    } finally {
      setTimeout(() => hide.remove(), 400);
      setTimeout(mark, 200);
    }
  }
  document.querySelectorAll(".t button").forEach(b => b.onclick = () => setTheme(b.dataset.name));
  mark();
  // the sticky tab bar needs the page colour behind it (white in light mode, dark in dark mode)
  setInterval(() => {
    const app = D.querySelector('[data-testid="stApp"]'), bar = D.querySelector('div:has(> [data-baseweb="tab-list"])');
    if (app && bar) bar.style.backgroundColor = P.getComputedStyle(app).backgroundColor;
  }, 300);
</script>
""".replace("SUN", icon("sun", 18, "#868e96")).replace("MOON", icon("moon", 18, "#868e96")) \
   .replace("MONITOR", icon("monitor", 18, "#868e96"))

with st.sidebar:
    st.markdown("<p style='font-size:14px;margin:0 0 4px'>Theme</p>", unsafe_allow_html=True)
    components.html(THEME_BUTTONS, height=46)
    st.markdown("---")
    st.caption("Data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ERA5 by ECMWF / Copernicus.")

tab_map, tab_models, tab_why, tab_test, tab_about = st.tabs(
    ["Tomorrow", "Model comparison", "Why this prediction?", "Test years (2024 to 2026)", "About"])

# ---------------------------------------------------------------- tab 1: dashboard
with tab_map:
    st.header("Will it rain tomorrow?")
    st.caption(note.replace("**", "") + f" Data loaded {loaded_at}.")

    # 1. the answer first (KPI cards)
    n_rain = int(pred["rain_tomorrow"].sum())
    top_city = pred["Soft voting"].idxmax()
    wet_city = pred["expected_mm"].idxmax()
    rain_cities = ", ".join(pred.index[pred["rain_tomorrow"]]) or "none"
    k = st.columns(3)
    k[0].metric("Cities with rain tomorrow", f"{n_rain} of {len(pred)}", border=True,
                help=f"Rain = 1 mm or more, decided with each zone's cut-off. Rain expected in: {rain_cities}")
    k[1].metric(f"Highest chance of rain: {top_city}", f"{pred.loc[top_city, 'Soft voting']:.0%}", border=True,
                help="Soft voting (Random Forest, XGBoost, LightGBM, CatBoost)")
    k[2].metric(f"Most rain expected: {wet_city}", f"{pred.loc[wet_city, 'expected_mm']:.1f} mm", border=True,
                help="Random Forest + ANN hybrid")

    # 2. the picture: weather map + chance by city
    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.markdown("**Weather report map**")
        st.caption("Opens on tomorrow's prediction. Press Play for the last 7 days of real rain, then tomorrow. Click a city for details.")
        components.html(weather_map_html(history, pred, day), height=610)
    with right, st.container(border=True):
        st.markdown("**Chance of rain by city**")
        st.caption("Bar colour = zone. The grey tick is the zone's cut-off for saying 'rain'. "
                   "The city picked on the Why and Test years tabs is in bold.")
        b = pred.sort_values("Soft voting")
        fig = go.Figure(go.Bar(x=b["Soft voting"] * 100, y=b.index, orientation="h",
                               marker_color=[ZONE_COLORS[z] for z in b["zone"]],
                               text=[f"{v:.0%}" for v in b["Soft voting"]], textposition="outside",
                               hovertemplate="%{y}: %{x:.0f}%<extra></extra>", showlegend=False))
        fig.add_trace(go.Scatter(x=b["threshold"] * 100, y=b.index, mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="grey")),
                                 hovertemplate="cut-off %{x:.0f}%<extra></extra>"))
        for z, c in ZONE_COLORS.items():
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=f"{z} zone"))
        fig.update_yaxes(tickvals=list(b.index), ticktext=[f"<b>{c}</b>" if c == city else c for c in b.index])
        fig.update_layout(height=640, xaxis=dict(range=[0, 112], title="chance of rain (%)"), margin=dict(t=10, l=10),
                          legend=dict(orientation="h", y=-0.12), barmode="overlay")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 3. the details
    with st.container(border=True):
        st.markdown("**Per city**")
        table = pd.DataFrame({
            "zone": pred["zone"],
            "chance of rain": (pred["Soft voting"] * 100).round(0).astype(int).astype(str) + " %",
            "answer": np.where(pred["rain_tomorrow"], "Rain", "No rain"),
            "expected (mm)": pred["expected_mm"].round(1),
            "rain today (mm)": pred["today_rain_mm"].round(1),
        })
        if len(actual):
            label = "Open-Meteo forecast (mm)" if is_live else "actual (mm)"
            table[label] = actual.reindex(pred.index).round(1)
        st.dataframe(table, use_container_width=True)
        st.caption("Chance of rain = soft voting (average of Random Forest, XGBoost, LightGBM and CatBoost). "
                   "'Rain' means 1 mm or more; the cut-off is set per zone ("
                   + ", ".join(f"{z.lower()} {t:.2f}" for z, t in info["zone_threshold"].items()) + "). "
                   "Expected mm comes from the Random Forest + ANN hybrid."
                   + (" The last column is Open-Meteo's own forecast for tomorrow, shown only to compare."
                      if is_live else ""))

    with st.expander("What every model says"):
        votes = (pred[MODEL_COLS] * 100).round(0)
        fig = px.imshow(votes.T, text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto",
                        labels=dict(x="", y="", color="chance (%)"), height=360)
        fig.update_traces(hovertemplate="%{y}, %{x}: %{z:.0f}%<extra></extra>")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Chance of rain (%) from each model. Soft voting is the average of the first four.")

# ---------------------------------------------------------------- tab 2: SHAP
with tab_why:
    st.header("Why this prediction?")
    c_city, _ = st.columns([1, 2])
    with c_city:
        city = city_picker("city_why")
    st.markdown(f"**{city}** ({pred.loc[city, 'zone']} zone), {next_day.date()}: the dashboard's chance of rain is "
                f"**{pred.loc[city, 'Soft voting']:.0%}** (soft voting, the average of four models). The chart explains "
                f"**CatBoost**, one of those four models, which gives **{pred.loc[city, 'CatBoost']:.0%}**.")
    with st.spinner(f"Explaining the prediction for {city} ..."):
        sv = get_explainer(models["CatBoost"])(X.loc[[city]])
    sv.feature_names = [nice(f) for f in X.columns]
    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.markdown(f"**What pushed the prediction for {city}**")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        plt.gcf().set_size_inches(6.4, 6.2)        # SHAP sets a wide figure; a narrower one keeps the text readable here
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Read it from the bottom up: the model starts from its usual answer (E[f(X)]) and each bar shows "
                   "how much one feature moved it. Red bars push the chance of rain up, blue bars push it down. "
                   "The number before each name is today's value. f(x) is the final answer, on the model's internal "
                   "(log-odds) scale.")
    with right, st.container(border=True):
        st.markdown("**What matters most in each zone (test years)**")
        imp = pd.DataFrame(info["shap_by_zone"])[["Wet", "Intermediate", "Dry"]] * 100
        imp.index = [nice(f) for f in imp.index]
        fig = px.imshow(imp.round(1), text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
                        height=40 + 28 * len(imp),
                        labels=dict(x="", y="", color="share (%)"))
        fig.update_traces(hovertemplate="%{y}, %{x} zone: %{z:.1f}%<extra></extra>")
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Share of the total SHAP importance (%) for each feature, by zone. City, zone and season columns "
                   "are added together.")

# ---------------------------------------------------------------- tab 3: model comparison
with tab_models:
    st.header("Model comparison (test years 2024 to Nov 2026)")
    st.caption("All scores are on days the models never saw in training. ROC-AUC shows how well a model ranks rain days "
               "above dry days: 1.0 is perfect and 0.5 is no better than guessing. Balanced accuracy is the average of "
               "the share of rain days and the share of dry days predicted correctly.")
    res = pd.DataFrame(info["test_results"]).T
    num = ["accuracy", "balanced_acc", "F1", "ROC_AUC", "Brier"]
    res[num] = res[num].apply(pd.to_numeric)      # the group column made every column text; scores must be numbers
    tp_all = get_test_predictions()
    tp_pred = tp_all["prob_soft_voting"] >= tp_all["zone"].map(info["zone_threshold"])
    k = st.columns(3)
    k[0].metric("Best ROC-AUC", f"{res['ROC_AUC'].max():.3f}", border=True, help=res["ROC_AUC"].idxmax())
    k[1].metric("Better than 'tomorrow = today'", f"+{res['ROC_AUC'].max() - res.loc['Baseline: persistence', 'ROC_AUC']:.3f} AUC",
                border=True, help="Compared to the persistence baseline")
    k[2].metric("Correct days (soft voting)", f"{(tp_pred == (tp_all['rain_tomorrow_mm'] >= 1)).mean():.0%}", border=True,
                help="All 9 cities, 9,441 test days, zone cut-offs")
    res = res.sort_values("ROC_AUC", ascending=False)
    c1, c2 = st.columns([3, 2])
    with c1:
        # dots, not bars: the scores are close, and bars that do not start at 0 would exaggerate the gaps
        ci = pd.DataFrame(info.get("test_ci", {})).T.reindex(res.index)
        fig = go.Figure()
        for fam, col in FAMILY_COLORS.items():
            d = res[res["group"].map(FAMILY) == fam]
            if d.empty:
                continue
            lo, hi = ci.loc[d.index, "low"], ci.loc[d.index, "high"]
            hollow = fam.startswith("Baseline")
            fig.add_trace(go.Scatter(
                x=d["ROC_AUC"], y=d.index, mode="markers", name=fam,
                marker=dict(size=11, color="rgba(0,0,0,0)" if hollow else col, line=dict(width=2, color=col)),
                error_x=dict(type="data", symmetric=False, array=(hi - d["ROC_AUC"]).fillna(0),
                             arrayminus=(d["ROC_AUC"] - lo).fillna(0), color="#9a9a94", thickness=1.5, width=4),
                hovertext=[f"{m}: {v:.3f}" + ("" if pd.isna(a) else f" (95% interval {a:.3f} to {b:.3f})")
                           for m, v, a, b in zip(d.index, d["ROC_AUC"], lo, hi)],
                hovertemplate="%{hovertext}<extra></extra>"))
        fig.update_layout(height=480, margin=dict(t=10), xaxis_title="ROC-AUC (dot) with 95% bootstrap interval",
                          yaxis=dict(categoryorder="array", categoryarray=list(res.index[::-1])),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Intervals come from 1,000 resamples of the test days (notebook section 6.10). "
                   "Where the intervals overlap a lot, the models cannot really be told apart.")
    with c2:
        table = res[["ROC_AUC", "balanced_acc", "F1"]].rename(columns={   # family is shown by the dot colours
            "ROC_AUC": "ROC-AUC", "balanced_acc": "Balanced acc.", "F1": "F1"})
        st.dataframe(table, use_container_width=True, height=480,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["ROC-AUC", "Balanced acc.", "F1"]})

    st.subheader("By zone")
    bz = pd.DataFrame(info["test_by_zone"])
    piv = bz.pivot(index="model", columns="zone", values="ROC_AUC")[["Wet", "Intermediate", "Dry"]]
    piv = piv.loc[[m for m in res.index if m in piv.index]]          # best model first, as in the chart above
    fig = px.imshow(piv.round(3), text_auto=".3f", color_continuous_scale="Blues", aspect="auto", height=420,
                    labels=dict(x="", y="", color="ROC-AUC"))
    fig.update_traces(hovertemplate="%{y}, %{x} zone: %{z:.3f}<extra></extra>")
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("ROC-AUC for each zone. The wet zone is the easiest to predict.")

    st.subheader("By monsoon season and zone (soft voting, ROC-AUC)")
    sz = pd.DataFrame(info["test_by_season_zone"])[["Wet", "Intermediate", "Dry"]]
    fig = px.imshow(sz.round(3), text_auto=".3f", color_continuous_scale="Blues", aspect="auto", height=300,
                    labels=dict(x="", y="", color="ROC-AUC"))
    fig.update_traces(hovertemplate="%{y}, %{x} zone: %{z:.3f}<extra></extra>")
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("The lightest cell is the hardest case: the dry zone in the first inter-monsoon, when rain comes from "
               "sudden thunderstorms.")

    st.subheader("Can the chance of rain be trusted? (soft voting)")
    cal = tp_all.assign(rained=tp_all["rain_tomorrow_mm"] >= 1)
    cal["band"] = cal.groupby("zone")["prob_soft_voting"].transform(lambda p: pd.qcut(p, 10, labels=False, duplicates="drop"))
    cal = cal.groupby(["zone", "band"]).agg(predicted=("prob_soft_voting", "mean"), rained=("rained", "mean")).reset_index()
    fig = px.line(cal, x="predicted", y="rained", color="zone", markers=True, height=420,
                  category_orders={"zone": ["Wet", "Intermediate", "Dry"]},
                  color_discrete_map=ZONE_COLORS,
                  labels={"predicted": "predicted chance of rain", "rained": "share of days it really rained"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(range=[0, 1], tickformat=".0%"),
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The test days of each zone are split into 10 groups by predicted chance. Points close to the dotted "
               "line mean the chance shown on the map can be trusted.")

    st.subheader("How much rain? (Random Forest + ANN)")
    reg = pd.DataFrame(info["regression_results"]).T.rename(columns={"MAE (mm)": "Average error (mm)",
                                                                     "RMSE (mm)": "RMSE (mm)", "R2": "R²"})
    st.dataframe(reg, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in reg.columns})
    st.caption("Average error is how far the predicted amount is from the real amount, in mm (lower is better). "
               "The RF + ANN hybrid has the lowest error.")

# ---------------------------------------------------------------- tab 4: test years
with tab_test:
    st.header("How the model did in the test years")
    tp = get_test_predictions()
    c_city, _ = st.columns([1, 2])
    with c_city:
        city2 = city_picker("city_test")
    d = tp[tp["city"] == city2].copy()
    thr = info["zone_threshold"][d["zone"].iloc[0]]
    d["predicted"] = d["prob_soft_voting"] >= thr
    d["rained"] = d["rain_tomorrow_mm"] >= 1
    acc = (d["predicted"] == d["rained"]).mean()
    hit = d.loc[d["rained"], "predicted"].mean()
    dry_ok = (~d.loc[~d["rained"], "predicted"]).mean()
    k1, k2, k3 = st.columns(3)
    k1.metric("Correct days", f"{acc:.0%}", border=True)
    k2.metric("Rain days caught", f"{hit:.0%}", border=True)
    k3.metric("Dry days caught", f"{dry_ok:.0%}", border=True)

    left, right = st.columns([2, 3])
    with left, st.container(border=True):
        st.markdown(f"**Confusion matrix, all test days in {city2}**")
        cm = np.array([[int((~d["rained"] & ~d["predicted"]).sum()), int((~d["rained"] & d["predicted"]).sum())],
                       [int((d["rained"] & ~d["predicted"]).sum()), int((d["rained"] & d["predicted"]).sum())]])
        share = cm / cm.sum(axis=1, keepdims=True)
        labels = [["dry, predicted dry", "false alarm"], ["rain missed", "rain, predicted rain"]]
        text = [[f"{cm[i, j]:,}<br>{share[i, j]:.0%}" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=share, x=["predicted: no rain", "predicted: rain"], y=["actual: no rain", "actual: rain"],
                                   text=text, texttemplate="%{text}", textfont=dict(size=16), colorscale="Blues",
                                   zmin=0, zmax=1, showscale=False, customdata=labels,
                                   hovertemplate="%{customdata}: %{text}<extra></extra>"))
        fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Number of days and the share of each actual class (row).")
    with right, st.container(border=True):
        st.markdown(f"**Correct days by month in {city2}**")
        mon = d.assign(ok=d["predicted"] == d["rained"], m=d["date"].dt.to_period("M").dt.to_timestamp())
        mon = mon.groupby("m")["ok"].mean().reset_index()
        fig = px.bar(mon, x="m", y="ok", labels={"m": "", "ok": "correct days"}, height=360)
        fig.update_traces(marker_color="#1f77b4", hovertemplate="%{x|%b %Y}: %{y:.0%}<extra></extra>")
        fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    months = st.slider("Months to show", 1, 35, 6)
    dd = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    # two panels on one shared time axis (not two y-axes on one plot)
    when = dd["date"] + pd.Timedelta(days=1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        subplot_titles=("Predicted chance of rain (dotted line: the zone's cut-off)",
                                        "Rain that really fell (dotted line: 1 mm, a rain day)"))
    fig.add_trace(go.Scatter(x=when, y=dd["prob_soft_voting"], name="predicted chance", line=dict(color="#2a78d6", width=2),
                             hovertemplate="%{x|%d %b %Y}: %{y:.0%}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=thr, line_dash="dot", line_color="#9a9a94", annotation_text=f"cut-off {thr:.2f}",
                  annotation_position="bottom left", row=1, col=1)
    fig.add_trace(go.Bar(x=when, y=dd["rain_tomorrow_mm"], name="rain (mm)", marker_color="#2a78d6",
                         hovertemplate="%{x|%d %b %Y}: %{y:.1f} mm<extra></extra>"), row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#9a9a94", row=2, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("These predictions come from the models trained only on 2010 to 2021, so this is a fair test.")

# ---------------------------------------------------------------- tab 5: about
with tab_about:
    st.header("About this project")
    st.markdown("""
**Question:** will it rain (1 mm or more) tomorrow in 9 Sri Lankan cities, and does it work equally well in the
wet, intermediate and dry zones?

**Data:** daily ERA5 reanalysis weather for 2010 to Nov 2026 from the Open-Meteo Historical Weather API. The live
page uses the Open-Meteo forecast API (last 60 days + today), so its values come from forecast models, not ERA5.

**Models:** bagging (Random Forest), boosting (XGBoost, LightGBM, CatBoost), soft voting, stacking, a hybrid
ARIMA + Random Forest, and a Random Forest + ANN hybrid for the rainfall amount. Full details, results and the
discussion are in the notebook in the GitHub repository.

**AI-use declaration:** AI tools (Claude) were used to write all of the code and its comments (the notebook, this Streamlit dashboard and
the scripts that draw the charts and images), and to help with the wording of my written sections (the points and
decisions in them are my own). The project idea, the choice of data and cities, the planning, the modelling decisions, and the
plan and Excalidraw-style look of the architecture diagram are my own. I reviewed the code and the results,
and I take responsibility for the submitted work.

Wimalasuriya A.G.A.B. (224220D), CM3720 Machine Learning & Pattern Recognition
""")
    st.caption(f"Models trained until {info['trained_until']}.")
