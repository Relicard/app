import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(page_title="Simulatore Uptime Backtesting", layout="wide")

st.title("⛏️ Simulatore Uptime – Backtesting Mining vs ERCOT RTM")
st.caption(
    "Backtest storico: mining revenue (hashprice) – costo elettrico (ERCOT RTM LZ_WEST) "
    "con curtailment basato su cutoff o uptime target."
)

# =========================
# Helpers
# =========================

@st.cache_data(show_spinner=False)
def load_ercot_rtm_xlsx(file) -> pd.DataFrame:
    """
    ERCOT RTM export format (like your file):
    - Row 0: "RTM Prices"
    - Row 1: header: Date, LoadZone, Settlement, Intervals, Sum, Min, Max, Avg, 1.0..100.0
    - Data rows start at row 2
    Handles DST days with 92 or 100 intervals.
    Returns long df: ts, price_usd_per_mwh
    """
    df_raw = pd.read_excel(file, header=None)

    if str(df_raw.iat[0, 0]).strip().upper() == "RTM PRICES":
        header = df_raw.iloc[1].tolist()
        df = df_raw.iloc[2:].copy()
        df.columns = header
    else:
        df = pd.read_excel(file)

    def find_col(name: str):
        name = name.strip().lower()
        for c in df.columns:
            if str(c).strip().lower() == name:
                return c
        return None

    col_date = find_col("date")
    col_zone = find_col("loadzone")
    col_sett = find_col("settlement")
    col_intervals = find_col("intervals")

    if not (col_date and col_zone and col_sett and col_intervals):
        raise ValueError("Formato ERCOT non riconosciuto: mancano colonne Date/LoadZone/Settlement/Intervals.")

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_zone] = df[col_zone].astype(str).str.upper().str.strip()
    df[col_sett] = df[col_sett].astype(str).str.upper().str.strip()
    df[col_intervals] = pd.to_numeric(df[col_intervals], errors="coerce")

    df = df[(df[col_zone] == "WEST") & (df[col_sett] == "LZ")].copy()
    if df.empty:
        raise ValueError("Non trovo righe con LoadZone=WEST e Settlement=LZ nel file ERCOT.")

    interval_cols = []
    interval_map = {}
    for c in df.columns:
        try:
            val = float(c)
            if np.isfinite(val):
                i = int(round(val))
                if 1 <= i <= 300 and abs(val - i) < 1e-9:
                    interval_cols.append(c)
                    interval_map[c] = i
        except Exception:
            pass

    if len(interval_cols) < 80:
        raise ValueError(
            "Non trovo le colonne intervalli nel file ERCOT. "
            "Mi aspetto intestazioni numeriche tipo 1.0..96.0 (o 100.0 per DST)."
        )

    long_df = df[[col_date, col_intervals] + interval_cols].melt(
        id_vars=[col_date, col_intervals],
        value_vars=interval_cols,
        var_name="interval_col",
        value_name="price_usd_per_mwh",
    )

    long_df["interval"] = long_df["interval_col"].map(interval_map).astype(int)
    long_df = long_df[long_df["interval"] <= long_df[col_intervals]].copy()

    long_df["ts"] = long_df[col_date] + pd.to_timedelta((long_df["interval"] - 1) * 15, unit="m")
    long_df["price_usd_per_mwh"] = pd.to_numeric(long_df["price_usd_per_mwh"], errors="coerce")

    long_df = long_df.dropna(subset=["ts", "price_usd_per_mwh"]).sort_values("ts")
    return long_df[["ts", "price_usd_per_mwh"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_btc_csv(file) -> pd.DataFrame:
    """
    Parses BTC daily price CSV like your "Prezzo bitcoin 2025.csv".
    Handles:
      - Date format like '01/01/2025 23.58.00' (dots in time)
      - Close with decimal comma '93462,86'
    Returns columns: date (python date), btc_usd (float)
    """
    df = pd.read_csv(file)

    col_date = next((c for c in df.columns if str(c).strip().lower() in ["date", "data"]), df.columns[0])
    col_close = next((c for c in df.columns if str(c).strip().lower() in ["close", "chiusura", "price"]), df.columns[1])

    def fix_dt(x: str) -> str:
        x = str(x).strip()
        return re.sub(r"(\d{2})\.(\d{2})\.(\d{2})$", r"\1:\2:\3", x)

    df[col_date] = df[col_date].apply(fix_dt)
    df["dt"] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")

    df["btc_usd"] = (
        df[col_close]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["btc_usd"] = pd.to_numeric(df["btc_usd"], errors="coerce")

    df = df.dropna(subset=["dt", "btc_usd"]).copy()
    df["date"] = df["dt"].dt.date

    df = df.sort_values("dt").groupby("date", as_index=False)["btc_usd"].last()
    return df


@st.cache_data(show_spinner=False)
def fetch_hashprice_blockchain(start: date, end: date) -> pd.DataFrame:
    """
    Compute daily hashprice (USD per TH/s per day) from Blockchain.com Charts API:
      hashprice = miners_revenue_usd / network_hashrate_ths
    """
    base = "https://api.blockchain.info/charts"

    days = (end - start).days + 1
    timespan = f"{days}days"
    start_str = start.isoformat()

    def fetch_chart(chart_name: str) -> pd.DataFrame:
        url = f"{base}/{chart_name}"
        params = {"timespan": timespan, "start": start_str, "format": "json", "sampled": "false"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        unit = j.get("unit", "")
        values = j.get("values", [])
        dfx = pd.DataFrame(values)
        dfx["date"] = pd.to_datetime(dfx["x"], unit="s", utc=True).dt.date
        dfx = dfx.rename(columns={"y": chart_name})
        dfx.attrs["unit"] = unit
        return dfx[["date", chart_name]]

    rev = fetch_chart("miners-revenue")  # USD/day
    hr = fetch_chart("hash-rate")        # typically TH/s (unit may vary)

    df = rev.merge(hr, on="date", how="inner").sort_values("date")

    hr_unit = getattr(hr, "attrs", {}).get("unit", "")
    if isinstance(hr_unit, str) and "GH" in hr_unit.upper():
        df["hash-rate"] = df["hash-rate"] / 1000.0

    df["miners-revenue"] = pd.to_numeric(df["miners-revenue"], errors="coerce")
    df["hash-rate"] = pd.to_numeric(df["hash-rate"], errors="coerce")
    df = df.dropna(subset=["miners-revenue", "hash-rate"])
    df = df[df["hash-rate"] > 0].copy()

    df["hashprice_usd_per_th_day"] = df["miners-revenue"] / df["hash-rate"]
    return df[["date", "hashprice_usd_per_th_day"]]


def compute_uptime_from_cutoff(prices_15m: pd.Series, cutoff: float) -> float:
    if prices_15m.empty:
        return 0.0
    return float((prices_15m <= cutoff).mean())


def compute_cutoff_from_target_uptime(prices_15m: pd.Series, target_uptime: float) -> float:
    if prices_15m.empty:
        return np.nan
    target_uptime = float(np.clip(target_uptime, 0.0, 1.0))
    return float(np.quantile(prices_15m.dropna().values, target_uptime))


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


@st.cache_data(show_spinner=False)
def optimize_cutoff(
    ercot_15m_period: pd.DataFrame,
    hp_daily: pd.DataFrame,
    ths: float,
    kw: float,
    cutoff_values: np.ndarray,
    adder_usd_per_mwh: float,
) -> pd.DataFrame:
    """
    Robust cutoff optimization:
      - Merge hashprice daily into 15m data
      - Forward/back-fill missing hashprice inside period
      - Compute profit for each cutoff
      - Adds avg_cost_usd_per_mwh (weighted, ON only)
    NOTE: cutoff compares against raw ERCOT RTM price; adder affects cost only.
    """
    df15 = ercot_15m_period.copy()
    df15 = df15.dropna(subset=["ts", "price_usd_per_mwh"]).copy()
    df15["date"] = df15["ts"].dt.date

    hp2 = hp_daily.copy()
    hp2 = hp2.dropna(subset=["date", "hashprice_usd_per_th_day"]).copy()
    hp2["date"] = pd.to_datetime(hp2["date"]).dt.date

    df15 = df15.merge(hp2, on="date", how="left").sort_values("ts")
    df15["hashprice_usd_per_th_day"] = df15["hashprice_usd_per_th_day"].ffill().bfill()

    if df15["hashprice_usd_per_th_day"].isna().all():
        return pd.DataFrame(columns=[
            "cutoff", "uptime_overall", "cost_usd", "revenue_usd", "profit_usd",
            "mwh_on", "avg_cost_usd_per_mwh"
        ])

    mwh_per_interval = (kw * 0.25) / 1000.0

    price_raw = df15["price_usd_per_mwh"].astype(float).to_numpy()
    price_eff = price_raw + float(adder_usd_per_mwh)

    day = df15["date"].to_numpy()
    hp = df15["hashprice_usd_per_th_day"].astype(float).to_numpy()

    unique_days, day_idx = np.unique(day, return_inverse=True)
    intervals_per_day = np.bincount(day_idx)

    hp_sum = np.bincount(day_idx, weights=hp)
    hp_daily_arr = hp_sum / np.maximum(intervals_per_day, 1)

    results = []
    for c in cutoff_values:
        on = price_raw <= float(c)

        mwh_on = float(np.sum(on) * mwh_per_interval)
        cost = float(np.sum(price_eff[on] * mwh_per_interval))

        on_count = np.bincount(day_idx, weights=on.astype(float), minlength=len(unique_days))
        uptime_daily = on_count / np.maximum(intervals_per_day, 1)

        revenue = float(np.sum(ths * hp_daily_arr * uptime_daily))
        profit = revenue - cost
        uptime_overall = float(on.mean())

        avg_cost = (cost / mwh_on) if mwh_on > 0 else np.nan

        results.append((float(c), uptime_overall, cost, revenue, profit, mwh_on, avg_cost))

    out = pd.DataFrame(
        results,
        columns=["cutoff", "uptime_overall", "cost_usd", "revenue_usd", "profit_usd", "mwh_on", "avg_cost_usd_per_mwh"],
    )
    return out


# =========================
# ASIC dropdown (simple, hardcoded)
# =========================
ASIC_PRESETS = [
    {"name": "Manuale (nessun ASIC)", "ths": None, "kw": None},

    # -----------------------------
    # Bitmain — Antminer (air)
    # -----------------------------
    {"name": "Bitmain Antminer S9 (13.5 TH) ~1.35 kW", "ths": 13.5, "kw": 1.35},
    {"name": "Bitmain Antminer S9i (14 TH) ~1.32 kW", "ths": 14.0, "kw": 1.32},
    {"name": "Bitmain Antminer S11 (20.5 TH) ~1.44 kW", "ths": 20.5, "kw": 1.44},
    {"name": "Bitmain Antminer S15 (28 TH) ~1.60 kW", "ths": 28.0, "kw": 1.60},
    {"name": "Bitmain Antminer S17 (56 TH) ~2.52 kW", "ths": 56.0, "kw": 2.52},
    {"name": "Bitmain Antminer S17 Pro (53 TH) ~2.09 kW", "ths": 53.0, "kw": 2.09},
    {"name": "Bitmain Antminer S17+ (73 TH) ~2.92 kW", "ths": 73.0, "kw": 2.92},

    {"name": "Bitmain Antminer T19 (84 TH) ~3.15 kW", "ths": 84.0, "kw": 3.15},
    {"name": "Bitmain Antminer S19 (95 TH) ~3.25 kW", "ths": 95.0, "kw": 3.25},
    {"name": "Bitmain Antminer S19 Pro (110 TH) ~3.25 kW", "ths": 110.0, "kw": 3.25},
    {"name": "Bitmain Antminer S19j (90 TH) ~3.10 kW", "ths": 90.0, "kw": 3.10},
    {"name": "Bitmain Antminer S19j Pro (104 TH) ~3.07 kW", "ths": 104.0, "kw": 3.07},
    {"name": "Bitmain Antminer S19j Pro+ (122 TH) ~3.35 kW", "ths": 122.0, "kw": 3.35},
    {"name": "Bitmain Antminer S19k Pro (120 TH) ~2.76 kW", "ths": 120.0, "kw": 2.76},

    {"name": "Bitmain Antminer S19 XP (140 TH) ~3.01 kW", "ths": 140.0, "kw": 3.01},

    {"name": "Bitmain Antminer S21 (200 TH) ~3.55 kW", "ths": 200.0, "kw": 3.55},
    {"name": "Bitmain Antminer S21 Pro (234 TH) ~3.51 kW", "ths": 234.0, "kw": 3.51},
    {"name": "Bitmain Antminer S21 Pro (245 TH) ~3.51 kW", "ths": 245.0, "kw": 3.51},
    {"name": "Bitmain Antminer S21 XP (270 TH) ~3.65 kW", "ths": 270.0, "kw": 3.65},

    # -----------------------------
    # Bitmain — Antminer (hydro / immersion)
    # -----------------------------
    {"name": "Bitmain Antminer S19 XP Hydro (255 TH) ~5.34 kW", "ths": 255.0, "kw": 5.34},
    {"name": "Bitmain Antminer S21 Hydro (335 TH) ~5.36 kW", "ths": 335.0, "kw": 5.36},
    {"name": "Bitmain Antminer S23 Hyd (580 TH) ~5.51 kW", "ths": 580.0, "kw": 5.51},
    {"name": "Bitmain Antminer S23 Hyd 3U (1160 TH) ~11.02 kW", "ths": 1160.0, "kw": 11.02},
    {"name": "Bitmain Antminer S23 Immersion (442 TH) ~5.304 kW", "ths": 442.0, "kw": 5.304},

    # -----------------------------
    # Bitdeer — Sealminer
    # -----------------------------
    {"name": "SEALMINER A3 Pro Hydro (660 TH) ~8.25 kW", "ths": 660.0, "kw": 8.25},
    {"name": "SEALMINER A3 Pro Air (290 TH) ~3.625 kW", "ths": 290.0, "kw": 3.625},
    {"name": "SEALMINER A3 Hydro (500 TH) ~6.75 kW", "ths": 500.0, "kw": 6.75},
    {"name": "SEALMINER A3 Air (260 TH) ~3.64 kW", "ths": 260.0, "kw": 3.64},
    {"name": "SEALMINER A2 Pro Hyd (500 TH) ~7.45 kW", "ths": 500.0, "kw": 7.45},
    {"name": "SEALMINER A2 Pro Air (255 TH) ~3.79 kW", "ths": 255.0, "kw": 3.79},
    {"name": "Sealminer A2Hyd (446 TH) ~7.359 kW", "ths": 446.0, "kw": 7.359},
    {"name": "Sealminer A2 (226 TH) ~3.729 kW", "ths": 226.0, "kw": 3.729},

    # -----------------------------
    # MicroBT — Whatsminer (air)
    # -----------------------------
    {"name": "MicroBT Whatsminer M20S (68 TH) ~3.36 kW", "ths": 68.0, "kw": 3.36},
    {"name": "MicroBT Whatsminer M21S (56 TH) ~3.36 kW", "ths": 56.0, "kw": 3.36},
    {"name": "MicroBT Whatsminer M30S (86 TH) ~3.26 kW", "ths": 86.0, "kw": 3.26},
    {"name": "MicroBT Whatsminer M30S+ (100 TH) ~3.40 kW", "ths": 100.0, "kw": 3.40},
    {"name": "MicroBT Whatsminer M30S++ (112 TH) ~3.47 kW", "ths": 112.0, "kw": 3.47},
    {"name": "MicroBT Whatsminer M31S+ (80 TH) ~3.36 kW", "ths": 80.0, "kw": 3.36},

    {"name": "MicroBT Whatsminer M50 (114 TH) ~3.30 kW", "ths": 114.0, "kw": 3.30},
    {"name": "MicroBT Whatsminer M50S (126 TH) ~3.27 kW", "ths": 126.0, "kw": 3.27},
    {"name": "MicroBT Whatsminer M53 (226 TH) ~6.554 kW", "ths": 226.0, "kw": 6.554},

    {"name": "MicroBT Whatsminer M60 (172 TH) ~3.432 kW", "ths": 172.0, "kw": 3.432},
    {"name": "MicroBT Whatsminer M60S (186 TH) ~3.441 kW", "ths": 186.0, "kw": 3.441},

    # -----------------------------
    # MicroBT — Whatsminer (hydro)
    # -----------------------------
    {"name": "MicroBT Whatsminer M63S Hydro (390 TH) ~7.215 kW", "ths": 390.0, "kw": 7.215},

    # -----------------------------
    # Canaan — Avalon (air)
    # -----------------------------
    {"name": "Canaan Avalon A1246 (90 TH) ~3.42 kW", "ths": 90.0, "kw": 3.42},
    {"name": "Canaan Avalon A1346 (110 TH) ~3.30 kW", "ths": 110.0, "kw": 3.30},
    {"name": "Canaan Avalon A1366 (130 TH) ~3.25 kW", "ths": 130.0, "kw": 3.25},
    {"name": "Canaan Avalon A1466 (150 TH) ~3.23 kW", "ths": 150.0, "kw": 3.23},
    {"name": "Canaan Avalon A1566 (185 TH) ~3.42 kW", "ths": 185.0, "kw": 3.42},
]


# =========================
# Sidebar - Inputs
# =========================
with st.sidebar:
    st.header("📥 Dati & Parametri")

    ercot_file = st.file_uploader("Carica ERCOT RTM (xlsx)", type=["xlsx"])
    btc_file = st.file_uploader("Carica BTC storico (csv) [opzionale]", type=["csv"])

    st.divider()

    # =========================
    # NEW: Adders input ($/MWh)
    # =========================
    if "adder_usd_per_mwh" not in st.session_state:
        st.session_state.adder_usd_per_mwh = 0.0

    adder_usd_per_mwh = st.number_input(
        "Adders energia ($/MWh)",
        value=float(st.session_state.adder_usd_per_mwh),
        step=0.5,
        help="Valore aggiunto al prezzo RTM ($/MWh) per simulare adders/fee/TDSP ecc. "
             "Impattano solo il costo quando sei ON.",
    )
    st.session_state.adder_usd_per_mwh = float(adder_usd_per_mwh)

    st.divider()

    # --- ASIC dropdown in an expander (tendina) ---
    with st.expander("🧰 Seleziona ASIC (tendina)", expanded=False):
        if "ths" not in st.session_state:
            st.session_state.ths = 200.0
        if "kw" not in st.session_state:
            st.session_state.kw = 3.5

        # SEARCH
        if "asic_search" not in st.session_state:
            st.session_state.asic_search = ""

        query = st.text_input(
            "Cerca ASIC (scrivi parte del nome)",
            key="asic_search",
            placeholder="es: S21, M60, S19...",
        ).strip().lower()

        preset_names_all = [a["name"] for a in ASIC_PRESETS]
        if query:
            preset_names = [n for n in preset_names_all if query in n.lower()]
        else:
            preset_names = preset_names_all

        if not preset_names:
            st.warning("Nessun ASIC trovato con questa ricerca. Torno su 'Manuale'.")
            selected_name = "Manuale (nessun ASIC)"
        else:
            # key fixed to keep state stable even when list changes
            selected_name = st.selectbox("Modello ASIC", preset_names, index=0, key="asic_selectbox")

        units = st.number_input("Units", min_value=1, value=1, step=1, key="asic_units")

        preset = next(a for a in ASIC_PRESETS if a["name"] == selected_name)

        if preset["ths"] is not None and preset["kw"] is not None:
            sug_ths = float(preset["ths"]) * float(units)
            sug_kw = float(preset["kw"]) * float(units)

            cA, cB = st.columns(2)
            cA.metric("Hashrate suggerito", f"{sug_ths:,.0f} TH/s")
            cB.metric("Consumo suggerito", f"{sug_kw:,.2f} kW")

            if st.button("➡️ Applica valori ASIC", key="apply_asic"):
                st.session_state.ths = float(sug_ths)
                st.session_state.kw = float(sug_kw)
                st.rerun()
        else:
            st.caption("Seleziona un ASIC diverso oppure usa i campi manuali sotto.")

    # defaults if state missing
    if "ths" not in st.session_state:
        st.session_state.ths = 200.0
    if "kw" not in st.session_state:
        st.session_state.kw = 3.5

    ths = st.number_input("Hashrate totale (TH/s)", min_value=0.0, value=float(st.session_state.ths), step=10.0)
    kw = st.number_input(
        "Consumo totale (kW)",
        min_value=0.0,
        value=float(st.session_state.kw),
        step=0.1,
        help="Interpreto questo valore come kW (cioè kWh per ora). Es: 3.5 kW ≈ 3.5 kWh/h.",
    )

    st.session_state.ths = float(ths)
    st.session_state.kw = float(kw)

    st.divider()
    mode = st.radio("Controllo curtailment tramite:", ["Cutoff ($/MWh)", "Uptime (%)"], index=0)

    if "cutoff" not in st.session_state:
        st.session_state.cutoff = 100.0
    if "uptime" not in st.session_state:
        st.session_state.uptime = 0.95

    cutoff = st.number_input("Cutoff ERCOT ($/MWh)", min_value=-1000.0, value=float(st.session_state.cutoff), step=5.0)
    uptime_pct = st.number_input("Uptime target (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.uptime * 100), step=1.0)

    st.divider()
    include_btc_price_chart = st.checkbox("Mostra anche grafico BTC (se caricato)", value=True)
    show_debug = st.checkbox("Debug (mostra tabelle intermedie)", value=False)


# =========================
# Main logic
# =========================
if not ercot_file:
    st.info("Carica il file **ERCOT RTM xlsx** (quello con intervalli 15 min). Poi procediamo.")
    st.stop()

with st.spinner("Leggo e preparo i dati ERCOT..."):
    ercot_15m = load_ercot_rtm_xlsx(ercot_file)

if ercot_15m.empty:
    st.error("Dati ERCOT vuoti dopo il parsing.")
    st.stop()

start_dt = ercot_15m["ts"].min().date()
end_dt = ercot_15m["ts"].max().date()

st.subheader("🗓️ Periodo di backtest")
c1, c2 = st.columns([1, 1])
with c1:
    start_date = st.date_input("Start", value=start_dt, min_value=start_dt, max_value=end_dt)
with c2:
    end_date = st.date_input("End", value=end_dt, min_value=start_dt, max_value=end_dt)

if start_date > end_date:
    st.error("Start date non può essere dopo end date.")
    st.stop()

mask = (ercot_15m["ts"].dt.date >= start_date) & (ercot_15m["ts"].dt.date <= end_date)
ercot_15m_period = ercot_15m.loc[mask].copy()
prices_series = ercot_15m_period["price_usd_per_mwh"]

# Sync cutoff/uptime based on selected mode
if mode == "Cutoff ($/MWh)":
    implied_uptime = compute_uptime_from_cutoff(prices_series, cutoff)
    st.session_state.cutoff = float(cutoff)
    st.session_state.uptime = float(implied_uptime)
else:
    target_uptime = float(uptime_pct) / 100.0
    implied_cutoff = compute_cutoff_from_target_uptime(prices_series, target_uptime)
    st.session_state.uptime = float(target_uptime)
    st.session_state.cutoff = float(implied_cutoff)

uptime_overall = float(st.session_state.uptime)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Uptime (periodo)", f"{uptime_overall*100:.2f}%")
kpi2.metric("Cutoff equivalente", f"{st.session_state.cutoff:,.2f} $/MWh")
kpi3.metric("Intervalli 15-min analizzati", f"{len(ercot_15m_period):,}")
kpi4.metric("Adders applicati", f"{float(st.session_state.adder_usd_per_mwh):,.2f} $/MWh")

# =========================
# Build daily electricity cost from 15m prices
# =========================
mwh_per_interval = (kw * 0.25) / 1000.0
adder_usd_per_mwh = float(st.session_state.adder_usd_per_mwh)

ercot_15m_period["mine_on"] = ercot_15m_period["price_usd_per_mwh"] <= float(st.session_state.cutoff)

# Effective price includes adders (cost only)
ercot_15m_period["price_eff_usd_per_mwh"] = ercot_15m_period["price_usd_per_mwh"] + adder_usd_per_mwh

ercot_15m_period["interval_cost_usd"] = np.where(
    ercot_15m_period["mine_on"],
    ercot_15m_period["price_eff_usd_per_mwh"] * mwh_per_interval,
    0.0,
)

ercot_15m_period["interval_mwh"] = np.where(
    ercot_15m_period["mine_on"],
    mwh_per_interval,
    0.0,
)

df_daily_power = (
    ercot_15m_period.assign(date=ercot_15m_period["ts"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        uptime=("mine_on", "mean"),
        elec_cost_usd=("interval_cost_usd", "sum"),
        mwh_mined=("interval_mwh", "sum"),
        avg_rtm_price=("price_usd_per_mwh", "mean"),
        avg_rtm_price_when_on=(
            "price_usd_per_mwh",
            lambda s: float(s[ercot_15m_period.loc[s.index, "mine_on"]].mean())
            if (ercot_15m_period.loc[s.index, "mine_on"].any())
            else np.nan,
        ),
    )
)

df_daily_power["avg_cost_usd_per_mwh"] = np.where(
    df_daily_power["mwh_mined"] > 0,
    df_daily_power["elec_cost_usd"] / df_daily_power["mwh_mined"],
    np.nan,
)

# =========================
# Hashprice daily (Blockchain.com)
# =========================
with st.spinner("Calcolo Hashprice storico (Blockchain.com) per il periodo selezionato..."):
    try:
        hp = fetch_hashprice_blockchain(start_date, end_date)
    except Exception as e:
        st.error("Errore nel download dati (Blockchain.com Charts API).\n\nDettaglio:\n" + str(e))
        st.stop()

df_daily = df_daily_power.merge(hp, on="date", how="left").sort_values("date")
df_daily["hashprice_usd_per_th_day"] = df_daily["hashprice_usd_per_th_day"].ffill().bfill()

df_daily["revenue_usd"] = ths * df_daily["hashprice_usd_per_th_day"] * df_daily["uptime"]
df_daily["profit_usd"] = df_daily["revenue_usd"] - df_daily["elec_cost_usd"]
df_daily["revenue_per_th_day_usd"] = df_daily["hashprice_usd_per_th_day"] * df_daily["uptime"]

# =========================
# Monthly aggregations
# =========================
df_month = (
    df_daily.assign(month=pd.to_datetime(df_daily["date"]).dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(
        elec_cost_usd=("elec_cost_usd", "sum"),
        revenue_usd=("revenue_usd", "sum"),
        profit_usd=("profit_usd", "sum"),
        mwh_mined=("mwh_mined", "sum"),
    )
)

df_month["avg_cost_usd_per_mwh"] = np.where(
    df_month["mwh_mined"] > 0,
    df_month["elec_cost_usd"] / df_month["mwh_mined"],
    np.nan,
)

total_cost = float(df_daily["elec_cost_usd"].sum())
total_rev = float(df_daily["revenue_usd"].sum())
total_profit = float(df_daily["profit_usd"].sum())
total_mwh = float(df_daily["mwh_mined"].sum())
avg_cost_total = (total_cost / total_mwh) if total_mwh > 0 else np.nan

st.subheader("📌 Risultati")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Costo elettrico totale", fmt_usd(total_cost))
r2.metric("Revenue mining totale", fmt_usd(total_rev))
r3.metric("Profit / Loss totale", fmt_usd(total_profit))
r4.metric("Costo medio energia (incl. adders)", f"{avg_cost_total:,.2f} $/MWh" if np.isfinite(avg_cost_total) else "n/a")

st.divider()

# =========================
# Charts
# =========================
st.subheader("📊 Grafici mensili (Costi / Revenue / Profit)")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="Costo elettrico", x=df_month["month"], y=df_month["elec_cost_usd"], marker_color="#8B0000"))
fig_bar.add_trace(go.Bar(name="Revenue mining", x=df_month["month"], y=df_month["revenue_usd"], marker_color="#B8860B"))
fig_bar.add_trace(go.Bar(name="Profit", x=df_month["month"], y=df_month["profit_usd"], marker_color="#006400"))

fig_bar.update_layout(
    barmode="group",
    height=450,
    margin=dict(l=20, r=20, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Mese",
    yaxis_title="USD",
)


# Linea profitto mensile (stessi valori delle barre)
fig_bar.add_trace(
    go.Scatter(
        x=df_month["month"],
        y=df_month["profit_usd"],
        mode="lines+markers",
        name="Profitto mensile (linea)",
        line=dict(color="white", width=3),
        marker=dict(size=6),
    )
)


st.plotly_chart(fig_bar, use_container_width=True)


st.subheader("⚡ Costo medio energia ($/MWh) – mensile (incl. adders)")
fig_cost = px.line(df_month, x="month", y="avg_cost_usd_per_mwh", markers=True)
fig_cost.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=30), xaxis_title="Mese", yaxis_title="$ / MWh")
st.plotly_chart(fig_cost, use_container_width=True)

st.subheader("📈 Cumulativo profit (giornaliero)")
df_daily["cum_profit_usd"] = df_daily["profit_usd"].cumsum()
fig_cum = px.line(df_daily, x="date", y="cum_profit_usd")
fig_cum.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=30), xaxis_title="Data", yaxis_title="USD")
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("⚡ Uptime e RTM (giornaliero)")
fig_uptime = go.Figure()
fig_uptime.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["uptime"] * 100, name="Uptime (%)", mode="lines"))
fig_uptime.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["avg_rtm_price"], name="RTM Avg ($/MWh)", mode="lines", yaxis="y2"))
fig_uptime.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=30),
    xaxis_title="Data",
    yaxis=dict(title="Uptime (%)"),
    yaxis2=dict(title="RTM Avg ($/MWh)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_uptime, use_container_width=True)

if btc_file and include_btc_price_chart:
    st.subheader("₿ Prezzo BTC")
    btc = load_btc_csv(btc_file)
    btc = btc[(btc["date"] >= start_date) & (btc["date"] <= end_date)]
    fig_btc = px.line(btc, x="date", y="btc_usd")
    fig_btc.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=30), xaxis_title="Data", yaxis_title="USD")
    st.plotly_chart(fig_btc, use_container_width=True)

# =========================
# Tables & Export
# =========================
st.subheader("📄 Tabella mensile")

df_month_display = df_month.copy()
for c in ["elec_cost_usd", "revenue_usd", "profit_usd", "mwh_mined", "avg_cost_usd_per_mwh"]:
    df_month_display[c] = pd.to_numeric(df_month_display[c], errors="coerce")

df_month_display["elec_cost_usd"] = df_month_display["elec_cost_usd"].round(2)
df_month_display["revenue_usd"] = df_month_display["revenue_usd"].round(2)
df_month_display["profit_usd"] = df_month_display["profit_usd"].round(2)
df_month_display["mwh_mined"] = df_month_display["mwh_mined"].round(4)
df_month_display["avg_cost_usd_per_mwh"] = df_month_display["avg_cost_usd_per_mwh"].round(2)

st.dataframe(df_month_display, use_container_width=True, hide_index=True)

csv_out = df_month_display.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica tabella mensile (CSV)", data=csv_out, file_name="backtest_mensile.csv", mime="text/csv")

if show_debug:
    st.subheader("🧪 Debug – daily")
    st.dataframe(df_daily.head(100), use_container_width=True, hide_index=True)
    st.subheader("🧪 Debug – ercot 15m sample")
    st.dataframe(ercot_15m_period.head(100), use_container_width=True, hide_index=True)

st.caption(
    "Note: i costi elettrici sono calcolati usando i prezzi RTM ($/MWh) e il consumo (kW) su intervalli da 15 minuti. "
    "Gli Adders ($/MWh) vengono sommati al prezzo RTM solo quando ON, e quindi incidono su costi, costo medio e profit. "
    "Le revenue mining usano l'hashprice giornaliero (USD per TH/day) e l'uptime effettivo del giorno."
)

st.divider()
st.subheader("🧠 Cervellone: ottimizzazione Cutoff (massimo profit)")

with st.expander("Impostazioni ottimizzazione", expanded=True):
    colA, colB, colC = st.columns(3)

    with colA:
        opt_min = st.number_input(
            "Cutoff minimo ($/MWh)",
            value=float(np.nanmin(ercot_15m_period["price_usd_per_mwh"])),
            step=5.0,
        )
    with colB:
        opt_max = st.number_input(
            "Cutoff massimo ($/MWh)",
            value=float(np.nanmax(ercot_15m_period["price_usd_per_mwh"])),
            step=5.0,
        )
    with colC:
        opt_steps = st.number_input(
            "Numero test (più alto = più preciso)",
            min_value=50,
            max_value=2000,
            value=300,
            step=50,
        )

    strategy = st.selectbox(
        "Spazio di ricerca cutoff",
        ["Lineare (min→max)", "Quantili (più intelligente)"],
        index=1,
        help="Quantili concentra test dove i prezzi cambiano più spesso.",
    )

prices = ercot_15m_period["price_usd_per_mwh"].dropna().astype(float).to_numpy()
if opt_min > opt_max:
    st.error("Cutoff minimo > cutoff massimo.")
    st.stop()

if strategy.startswith("Lineare"):
    cutoff_values = np.linspace(opt_min, opt_max, int(opt_steps))
else:
    pr = prices[(prices >= opt_min) & (prices <= opt_max)]
    if len(pr) < 100:
        cutoff_values = np.linspace(opt_min, opt_max, int(opt_steps))
    else:
        qs = np.linspace(0.0, 1.0, int(opt_steps))
        cutoff_values = np.quantile(pr, qs)
        cutoff_values = np.unique(np.clip(cutoff_values, opt_min, opt_max))

with st.spinner("Sto testando tanti cutoff..."):
    opt_df = optimize_cutoff(
        ercot_15m_period=ercot_15m_period,
        hp_daily=hp,
        ths=ths,
        kw=kw,
        cutoff_values=cutoff_values,
        adder_usd_per_mwh=float(st.session_state.adder_usd_per_mwh),
    )

opt_df = opt_df.dropna(subset=["profit_usd"]).copy()

if opt_df.empty:
    st.error("Ottimizzazione fallita: profit_usd è tutto NaN. Probabile mismatch date ERCOT vs hashprice.")
    st.stop()

best_idx = opt_df["profit_usd"].idxmax()
best = opt_df.loc[best_idx].copy()

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("✅ Cutoff migliore", f"{best['cutoff']:,.2f} $/MWh")
b2.metric("Uptime risultante", f"{best['uptime_overall']*100:.2f}%")
b3.metric("Profit totale", fmt_usd(best["profit_usd"]))
b4.metric("Revenue totale", fmt_usd(best["revenue_usd"]))
b5.metric("Prezzo medio pagato (incl. adders)", f"{best['avg_cost_usd_per_mwh']:,.2f} $/MWh" if np.isfinite(best["avg_cost_usd_per_mwh"]) else "n/a")

st.caption("Il cutoff migliore è quello che massimizza il profit totale nel periodo selezionato (costi includono adders).")

fig_opt = px.line(opt_df.sort_values("cutoff"), x="cutoff", y="profit_usd")
fig_opt.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=30),
    xaxis_title="Cutoff ($/MWh)",
    yaxis_title="Profit totale (USD)",
)
fig_opt.add_trace(
    go.Scatter(
        x=[best["cutoff"]],
        y=[best["profit_usd"]],
        mode="markers",
        name="Best",
        marker=dict(size=12),
    )
)
st.plotly_chart(fig_opt, use_container_width=True)

st.subheader("🏁 Top 10 cutoffs (per profit)")
top10 = opt_df.sort_values("profit_usd", ascending=False).head(10).copy()
top10["cutoff"] = top10["cutoff"].round(2)
top10["uptime_overall"] = (top10["uptime_overall"] * 100).round(2)
top10["cost_usd"] = top10["cost_usd"].round(2)
top10["revenue_usd"] = top10["revenue_usd"].round(2)
top10["profit_usd"] = top10["profit_usd"].round(2)
top10["avg_cost_usd_per_mwh"] = top10["avg_cost_usd_per_mwh"].round(2)

top10 = top10.rename(columns={
    "uptime_overall": "uptime_%_overall",
    "avg_cost_usd_per_mwh": "avg_$perMWh_paid",
})

st.dataframe(
    top10[["cutoff", "uptime_%_overall", "avg_$perMWh_paid", "cost_usd", "revenue_usd", "profit_usd"]],
    use_container_width=True,
    hide_index=True,
)

if st.button("➡️ Applica il cutoff migliore alla simulazione"):
    st.session_state.cutoff = float(best["cutoff"])
    st.session_state.uptime = float(best["uptime_overall"])
    st.rerun()

