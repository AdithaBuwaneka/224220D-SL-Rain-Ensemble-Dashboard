"""Will it rain tomorrow? Ensemble learning dashboard for Sri Lanka's climatic zones.
Run with:  streamlit run app.py"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

from src.predict import CLASSIFIERS, fetch_live, load_era5_history, load_models, predict

st.set_page_config(page_title="Sri Lanka rain tomorrow", page_icon=":material/rainy:", layout="wide")
ZONE_COLORS = {"Wet": "#1f77b4", "Intermediate": "#2ca02c", "Dry": "#d62728"}
MODEL_COLS = CLASSIFIERS + ["Soft voting", "Hybrid: ARIMA + RF"]


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data(ttl=3600, show_spinner="Getting today's weather from Open-Meteo ...")
def get_live(_info):
    return fetch_live(_info)


@st.cache_data
def get_era5():
    return load_era5_history()


@st.cache_data
def get_test_predictions():
    return pd.read_csv("data/processed/test_predictions.csv", parse_dates=["date"])


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


def weather_map_html(history, pred, day, info):
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
  #slider { flex: 1; accent-color: #1f77b4; }
  #day { min-width: 120px; font-weight: 600; color: #1f77b4; }
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


info, models = get_models()

# ---------------------------------------------------------------- data for the prediction
st.sidebar.title("Sri Lanka rain tomorrow")
st.sidebar.caption("CM3720 hands-on project by Wimalasuriya A.G.A.B. (224220D)")
source = st.sidebar.radio("Weather data", ["Live (Open-Meteo, today)", "Past day (ERA5 data)"])

live_failed = False
if source.startswith("Live"):
    try:
        history = get_live(info)
        today = pd.Timestamp.now(tz="Asia/Colombo").normalize().tz_localize(None)
        day = today if today in set(history["date"]) else history["date"].max() - pd.Timedelta(days=1)
        note = (f"Weather up to **{day.date()}** (today's values include the forecast for the rest of the day). "
                f"Prediction is for **{(day + pd.Timedelta(days=1)).date()}**.")
    except Exception as e:                       # no internet or API down: fall back to the saved data
        live_failed = True
        st.sidebar.warning(f"Live data not available ({type(e).__name__}). Showing the last ERA5 day instead.")
if not source.startswith("Live") or live_failed:
    history = get_era5()
    last = history["date"].max() - pd.Timedelta(days=1)
    day = pd.Timestamp(st.sidebar.date_input("Day (weather up to this day)", value=last.date(),
                                             min_value=pd.Timestamp("2010-01-08").date(), max_value=last.date()))
    note = (f"ERA5 weather up to **{day.date()}**, prediction for **{(day + pd.Timedelta(days=1)).date()}**. "
            "The saved models were trained on all years, so past days are a demo, not a fair test.")

is_live = source.startswith("Live") and not live_failed
loaded_at = pd.Timestamp.now(tz="Asia/Colombo").strftime("%d %b %Y, %H:%M") + " (Sri Lanka time)"
pred, X = predict(history, info, models, day)
next_day = day + pd.Timedelta(days=1)
actual = history[history["date"] == next_day].set_index("city")["precipitation_sum"]

st.sidebar.markdown("---")
st.sidebar.caption("Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ERA5 by ECMWF / Copernicus.")
with st.sidebar:
    components.html("""
<style>
  body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
  .t { display: flex; gap: 4px; border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 3px; width: fit-content; }
  .t button { display: inline-flex; border: 0; background: transparent; padding: 6px 12px; border-radius: 6px; cursor: pointer; }
  .t button:hover { background: rgba(128,128,128,.2); }
  .t button.on { background: rgba(31,119,180,.25); }
</style>
<div class="t">
  <button id="Light" title="Light mode">SUN</button>
  <button id="Dark" title="Dark mode">MOON</button>
  <button id="Use system setting" title="Same as my computer">MONITOR</button>
</div>
<script>
  const key = "stActiveTheme-" + window.parent.location.pathname + "-v1";
  let saved = null;
  try { saved = window.parent.localStorage.getItem(key); } catch (e) {}
  if (!saved) {            // first visit: start in light mode
    try { window.parent.localStorage.setItem(key, JSON.stringify({name: "Light"})); window.parent.location.reload(); } catch (e) {}
  }
  let now = "Light";
  try { now = JSON.parse(saved || "{}").name || now; } catch (e) {}
  document.querySelectorAll(".t button").forEach(b => {
    if (b.id === now) b.classList.add("on");
    b.onclick = () => { window.parent.localStorage.setItem(key, JSON.stringify({name: b.id})); window.parent.location.reload(); };
  });
</script>
""".replace("SUN", icon("sun", 18, "#868e96")).replace("MOON", icon("moon", 18, "#868e96"))
       .replace("MONITOR", icon("monitor", 18, "#868e96")), height=48)

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
    k[1].metric("Highest chance of rain", f"{top_city} ({pred.loc[top_city, 'Soft voting']:.0%})", border=True)
    k[2].metric("Most rain expected", f"{wet_city} ({pred.loc[wet_city, 'expected_mm']:.1f} mm)", border=True,
                help="Random Forest + ANN hybrid")

    # 2. the picture: weather map + chance by city
    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.markdown("**Weather report map**")
        st.caption("Opens on tomorrow's prediction. Press Play for the last 7 days of real rain, then tomorrow. Click a city for details.")
        components.html(weather_map_html(history, pred, day, info), height=610)
    with right, st.container(border=True):
        st.markdown("**Chance of rain by city**")
        st.caption("Bar colour = zone. The grey tick is the zone's cut-off for saying 'rain'.")
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
                   "'Rain' means 1 mm or more; the cut-off is set per zone (wet 0.80, intermediate 0.67, dry 0.46). "
                   "Expected mm comes from the Random Forest + ANN hybrid."
                   + (" The last column is Open-Meteo's own forecast for tomorrow, shown only to compare."
                      if is_live else ""))

    with st.expander("What every model says"):
        votes = (pred[MODEL_COLS] * 100).round(0)
        fig = go.Figure()
        for col in MODEL_COLS:
            fig.add_trace(go.Bar(name=col, x=votes.index, y=votes[col]))
        fig.update_layout(barmode="group", yaxis_title="chance of rain (%)", height=380, margin=dict(t=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------- tab 2: SHAP
with tab_why:
    st.header("Why this prediction?")
    city = st.selectbox("City", list(pred.index), index=list(pred.index).index("Colombo"))
    st.markdown(f"**{city}** ({pred.loc[city, 'zone']} zone): CatBoost gives a "
                f"**{pred.loc[city, 'CatBoost']:.0%}** chance of rain on {next_day.date()}.")
    explainer = shap.TreeExplainer(models["CatBoost"])
    sv = explainer(X.loc[[city]])
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push the chance of rain up, blue bars push it down (SHAP values, in log-odds). "
               "E[f(X)] is the average prediction and f(x) is this prediction.")

# ---------------------------------------------------------------- tab 3: model comparison
with tab_models:
    st.header("Model comparison (test years 2024 to Nov 2026)")
    res = pd.DataFrame(info["test_results"]).T
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
        fig = px.bar(res.reset_index(), x="ROC_AUC", y="index", color="group", orientation="h", height=480,
                     range_x=[0.5, 0.95], labels={"index": "", "ROC_AUC": "ROC-AUC"}, text="ROC_AUC")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(res[["group", "ROC_AUC", "balanced_acc", "F1", "accuracy"]].round(3), use_container_width=True, height=480)

    st.subheader("By zone")
    bz = pd.DataFrame(info["test_by_zone"])
    piv = bz.pivot(index="model", columns="zone", values="ROC_AUC")[["Wet", "Intermediate", "Dry"]]
    fig = px.imshow(piv.round(3), text_auto=True, color_continuous_scale="Blues", aspect="auto", height=420)
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("By monsoon season and zone (soft voting, ROC-AUC)")
    sz = pd.DataFrame(info["test_by_season_zone"])[["Wet", "Intermediate", "Dry"]]
    fig = px.imshow(sz.round(3), text_auto=True, color_continuous_scale="Blues", aspect="auto", height=300)
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Can the chance of rain be trusted? (soft voting)")
    cal = tp_all.assign(rained=tp_all["rain_tomorrow_mm"] >= 1)
    cal["band"] = cal.groupby("zone")["prob_soft_voting"].transform(lambda p: pd.qcut(p, 10, labels=False, duplicates="drop"))
    cal = cal.groupby(["zone", "band"]).agg(predicted=("prob_soft_voting", "mean"), rained=("rained", "mean")).reset_index()
    fig = px.line(cal, x="predicted", y="rained", color="zone", markers=True, height=420,
                  category_orders={"zone": ["Wet", "Intermediate", "Dry"]},
                  color_discrete_map={"Wet": "#1f77b4", "Intermediate": "#2ca02c", "Dry": "#d62728"},
                  labels={"predicted": "predicted chance of rain", "rained": "share of days it really rained"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(range=[0, 1], tickformat=".0%"),
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The test days of each zone are split into 10 groups by predicted chance. Points close to the dotted "
               "line mean the chance shown on the map can be trusted.")

    st.subheader("Rainfall amount (regression): Random Forest + ANN")
    st.dataframe(pd.DataFrame(info["regression_results"]).T.round(3), use_container_width=True)

# ---------------------------------------------------------------- tab 4: test years
with tab_test:
    st.header("How the model did in the test years")
    tp = get_test_predictions()
    city2 = st.selectbox("City ", sorted(tp["city"].unique()), index=sorted(tp["city"].unique()).index("Colombo"))
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
        st.markdown(f"**All test days in {city2}: what happened**")
        outcome = pd.Series({
            "Rain, predicted": int((d["rained"] & d["predicted"]).sum()),
            "Dry, predicted": int((~d["rained"] & ~d["predicted"]).sum()),
            "Rain missed": int((d["rained"] & ~d["predicted"]).sum()),
            "False alarm": int((~d["rained"] & d["predicted"]).sum()),
        })
        fig = go.Figure(go.Pie(labels=outcome.index, values=outcome.values, hole=0.6, sort=False,
                               marker_colors=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], textinfo="percent"))
        fig.add_annotation(text=f"{acc:.0%}<br>correct", showarrow=False, font_size=18)
        fig.update_layout(height=360, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dd["date"] + pd.Timedelta(days=1), y=dd["rain_tomorrow_mm"], name="actual rain (mm)",
                         marker_color="#9ecae1", yaxis="y2"))
    fig.add_trace(go.Scatter(x=dd["date"] + pd.Timedelta(days=1), y=dd["prob_soft_voting"], name="predicted chance",
                             line=dict(color="#08519c")))
    fig.add_hline(y=thr, line_dash="dot", annotation_text=f"cut-off {thr}")
    fig.update_layout(height=420, yaxis=dict(title="chance of rain", range=[0, 1]),
                      yaxis2=dict(title="rain (mm)", overlaying="y", side="right"), legend=dict(orientation="h", y=1.1),
                      margin=dict(t=10))
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
