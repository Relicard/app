import re
from datetime import datetime, timedelta, date

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
    Expected format (as your file):
    - First row contains headers like Date, LoadZone, Settlement, Intervals, Sum, Min, Max, Avg, 1..96
    - Each row is a day + zone + settlement type, with 96 interval prices.
    Returns a long dataframe with columns: ts, price_usd_per_mwh
    """
    df_raw = pd.read_excel(file)

    # Normalize headers: first row seems already headers (pandas uses row 0 as header)
    # In your file, columns include: 'RTM Prices', 'Unnamed: 1', ... but row0 contains real header names.
    # So we detect that pattern and rebuild columns if needed.
    if "RTM Prices" in df_raw.columns:
        # First row contains actual header names
        new_cols = df_raw.iloc[0].astype(str).tolist()
        df = df_raw.iloc[1:].copy()
        df.columns = new_cols
    else:
        df = df_raw.copy()

    # Standardize key columns (robust matching)
    col_date = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    col_zone = next((c for c in df.columns if str(c).strip().lower() == "loadzone"), None)
    col_sett = next((c for c in df.columns if str(c).strip().lower() == "settlement"), None)

    if not (col_date and col_zone and col_sett):
        raise ValueError("Formato ERCOT non riconosciuto: mancano colonne Date/LoadZone/Settlement.")

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_zone] = df[col_zone].astype(str).str.upper().str.strip()
    df[col_sett] = df[col_sett].astype(str).str.upper().str.strip()

    # Filter LZ_WEST + LZ settlement
    df = df[(df[col_zone] == "WEST") & (df[col_sett] == "LZ")].copy()
    if df.empty:
        raise ValueError("Non trovo righe con LoadZone=WEST e Settlement=LZ nel file ERCOT.")

    # Interval columns are "1".."96" (as strings or numbers)
    interval_cols = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"\d+", s):
            interval_cols.append(s)

    if len(interval_cols) < 80:
        raise ValueError("Non trovo le colonne intervalli 1..96 nel file ERCOT (formato inatteso).")

    # Melt to long
    long_df = df[[col_date] + interval_cols].melt(
        id_vars=[col_date],
        value_vars=interval_cols,
        var_name="interval",
        value_name="price_usd_per_mwh",
    )
    long_df["interval"] = long_df["interval"].astype(int)

    # Build timestamps: date + (interval-1)*15min
    long_df["ts"] = long_df[col_date] + pd.to_timedelta((long_df["interval"] - 1) * 15, unit="m")
    long_df["price_usd_per_mwh"] = pd.to_numeric(long_df["price_usd_per_mwh"], errors="coerce")

    long_df = long_df.dropna(subset=["ts", "price_usd_per_mwh"]).sort_values("ts")
    long_df = long_df[["ts", "price_usd_per_mwh"]].reset_index(drop=True)

    return long_df


@st.cache_data(show_spinner=False)
def load_btc_csv(file) -> pd.DataFrame:
    """
    Your BTC file example:
      Date: '01/01/2025 23.58.00'
      Close: '93462,86'
    We'll parse dayfirst and fix decimal comma.
    Returns columns: date (daily), btc_usd (float)
    """
    df = pd.read_csv(file)

    # Basic column guess
    col_date = next((c for c in df.columns if str(c).strip().lower() in ["date", "data"]), df.columns[0])
    col_close = next((c for c in df.columns if str(c).strip().lower() in ["close", "chiusura", "price"]), df.columns[1])

    # Fix weird time separators "23.58.00" -> "23:58:00"
    def fix_dt(x: str) -> str:
        x = str(x).strip()
        # replace last two '.' in time part with ':'
        # e.g. "01/01/2025 23.58.00"
        return re.sub(r"(\d{2})\.(\d{2})\.(\d{2})$", r"\1:\2:\3", x)

    df[col_date] = df[col_date].apply(fix_dt)
    df["dt"] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")

    # Close with decimal comma
    df["btc_usd"] = (
        df[col_close]
        .astype(str)
        .str.replace(".", "", regex=False)   # in case thousands separators appear
        .str.replace(",", ".", regex=False)
    )
    df["btc_usd"] = pd.to_numeric(df["btc_usd"], errors="coerce")

    df = df.dropna(subset=["dt", "btc_usd"]).copy()
    df["date"] = df["dt"].dt.date
    # Keep one per day (take last)
    df = df.sort_values("dt").groupby("date", as_index=False)["btc_usd"].last()

    return df


@st.cache_data(show_spinner=False)
def fetch_hashprice_bitbo(start: date, end: date) -> pd.DataFrame:
    """
    Bitbo API:
      https://charts.bitbo.io/api/v1/hashprice/?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    Returns columns: date, hashprice_usd_per_th_day
    """
    url = "https://charts.bitbo.io/api/v1/hashprice/"
    params = {"start_date": start.isoformat(), "end_date": end.isoformat()}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    data = payload.get("data", [])
    if not data:
        raise ValueError("Bitbo API: nessun dato hashprice restituito per il range selezionato.")

    out = pd.DataFrame(data, columns=["date", "hashprice"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["hashprice_usd_per_th_day"] = pd.to_numeric(out["hashprice"], errors="coerce")
    out = out.dropna(subset=["date", "hashprice_usd_per_th_day"])[["date", "hashprice_usd_per_th_day"]]
    return out


def compute_uptime_from_cutoff(prices_15m: pd.Series, cutoff: float) -> float:
    """Fraction of intervals where mining is ON (price <= cutoff)."""
    if prices_15m.empty:
        return 0.0
    return float((prices_15m <= cutoff).mean())


def compute_cutoff_from_target_uptime(prices_15m: pd.Series, target_uptime: float) -> float:
    """
    Given a target uptime fraction (0..1), return the price quantile such that
    share of prices <= cutoff is approximately target_uptime.
    """
    if prices_15m.empty:
        return np.nan
    target_uptime = float(np.clip(target_uptime, 0.0, 1.0))
    return float(np.quantile(prices_15m.dropna().values, target_uptime))


def monthly_agg(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df_daily with a 'date' column (python date) and numeric columns.
    Returns monthly aggregates with 'month' as YYYY-MM.
    """
    d = df_daily.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    num_cols = [c for c in d.columns if c not in ["date", "month"]]
    out = d.groupby("month", as_index=False)[num_cols].sum()
    return out


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


# =========================
# Sidebar - Inputs
# =========================
with st.sidebar:
    st.header("📥 Dati & Parametri")

    ercot_file = st.file_uploader("Carica ERCOT RTM (xlsx)", type=["xlsx"])
    btc_file = st.file_uploader("Carica BTC storico (csv) [opzionale]", type=["csv"])

    st.divider()
    ths = st.number_input("Hashrate totale (TH/s)", min_value=0.0, value=200.0, step=10.0)
    kw = st.number_input("Consumo totale (kW)", min_value=0.0, value=3.5, step=0.1,
                         help="Interpreto questo valore come kW (cioè kWh per ora). Es: 3.5 kW ≈ 3.5 kWh/h.")

    st.divider()
    mode = st.radio("Controllo curtailment tramite:", ["Cutoff ($/MWh)", "Uptime (%)"], index=0)

    # Use session_state to sync fields
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

# Determine backtest range
start_dt = ercot_15m["ts"].min().date()
end_dt = ercot_15m["ts"].max().date()

st.subheader("🗓️ Periodo di backtest")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    start_date = st.date_input("Start", value=start_dt, min_value=start_dt, max_value=end_dt)
with c2:
    end_date = st.date_input("End", value=end_dt, min_value=start_dt, max_value=end_dt)
with c3:
    st.caption("Consiglio: lascia l’intero range del file ERCOT per un backtest completo.")

if start_date > end_date:
    st.error("Start date non può essere dopo end date.")
    st.stop()

# Filter ERCOT to period
mask = (ercot_15m["ts"].dt.date >= start_date) & (ercot_15m["ts"].dt.date <= end_date)
ercot_15m_period = ercot_15m.loc[mask].copy()
prices_series = ercot_15m_period["price_usd_per_mwh"]

# Sync cutoff/uptime based on selected mode
if mode == "Cutoff ($/MWh)":
    # compute implied uptime
    implied_uptime = compute_uptime_from_cutoff(prices_series, cutoff)
    st.session_state.cutoff = cutoff
    st.session_state.uptime = implied_uptime
else:
    # compute implied cutoff
    target_uptime = uptime_pct / 100.0
    implied_cutoff = compute_cutoff_from_target_uptime(prices_series, target_uptime)
    st.session_state.uptime = target_uptime
    st.session_state.cutoff = implied_cutoff
    cutoff = implied_cutoff

uptime_overall = st.session_state.uptime

# Show quick KPI for curtailment
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Uptime (periodo)", f"{uptime_overall*100:.2f}%")
kpi2.metric("Cutoff equivalente", f"{st.session_state.cutoff:,.2f} $/MWh")
kpi3.metric("Intervalli 15-min analizzati", f"{len(ercot_15m_period):,}")

# =========================
# Build daily electricity cost from 15m prices
# =========================
# Energy per interval (MWh) = kW * 0.25h / 1000
mwh_per_interval = (kw * 0.25) / 1000.0

ercot_15m_period["mine_on"] = ercot_15m_period["price_usd_per_mwh"] <= float(st.session_state.cutoff)
ercot_15m_period["interval_cost_usd"] = np.where(
    ercot_15m_period["mine_on"],
    ercot_15m_period["price_usd_per_mwh"] * mwh_per_interval,
    0.0
)

# Aggregate to daily
df_daily_power = (
    ercot_15m_period.assign(date=ercot_15m_period["ts"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        uptime=("mine_on", "mean"),
        elec_cost_usd=("interval_cost_usd", "sum"),
        avg_rtm_price=("price_usd_per_mwh", "mean"),
        avg_rtm_price_when_on=("price_usd_per_mwh", lambda s: float(s[ercot_15m_period.loc[s.index, "mine_on"]].mean()) if (ercot_15m_period.loc[s.index, "mine_on"].any()) else np.nan),
    )
)

# =========================
# Hashprice (daily) - fetch from Bitbo
# =========================
with st.spinner("Scarico Hashprice storico (Bitbo) per il periodo selezionato..."):
    try:
        hp = fetch_hashprice_bitbo(start_date, end_date)
    except Exception as e:
        st.error(
            "Errore nel download hashprice da Bitbo.\n\n"
            "Se sei offline o bloccato da firewall/proxy, serve una serie hashprice alternativa.\n\n"
            f"Dettaglio: {e}"
        )
        st.stop()

# Merge daily datasets
df_daily = df_daily_power.merge(hp, on="date", how="left")

# If any missing hashprice, forward fill inside range
df_daily = df_daily.sort_values("date")
df_daily["hashprice_usd_per_th_day"] = df_daily["hashprice_usd_per_th_day"].ffill().bfill()

# Revenue:
# Revenue/day = TH/s * (USD per TH/day) * uptime(day)
df_daily["revenue_usd"] = ths * df_daily["hashprice_usd_per_th_day"] * df_daily["uptime"]

# Profit:
df_daily["profit_usd"] = df_daily["revenue_usd"] - df_daily["elec_cost_usd"]

# Add some useful derived metrics
df_daily["revenue_per_th_day_usd"] = df_daily["hashprice_usd_per_th_day"] * df_daily["uptime"]
df_daily["cost_per_mwh_effective"] = np.where(
    df_daily["elec_cost_usd"] > 0,
    (df_daily["elec_cost_usd"] / (mwh_per_interval * 96 * df_daily["uptime"])),
    np.nan
)

# =========================
# Monthly aggregations
# =========================
df_month = monthly_agg(df_daily[["date", "elec_cost_usd", "revenue_usd", "profit_usd"]])

total_cost = float(df_daily["elec_cost_usd"].sum())
total_rev = float(df_daily["revenue_usd"].sum())
total_profit = float(df_daily["profit_usd"].sum())

st.subheader("📌 Risultati")
r1, r2, r3 = st.columns(3)
r1.metric("Costo elettrico totale", fmt_usd(total_cost))
r2.metric("Revenue mining totale", fmt_usd(total_rev))
r3.metric("Profit / Loss totale", fmt_usd(total_profit))

st.divider()

# =========================
# Charts
# =========================
st.subheader("📊 Grafici mensili (Costi / Revenue / Profit)")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="Costo elettrico", x=df_month["month"], y=df_month["elec_cost_usd"]))
fig_bar.add_trace(go.Bar(name="Revenue mining", x=df_month["month"], y=df_month["revenue_usd"]))
fig_bar.add_trace(go.Bar(name="Profit", x=df_month["month"], y=df_month["profit_usd"]))
fig_bar.update_layout(
    barmode="group",
    height=450,
    margin=dict(l=20, r=20, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Mese",
    yaxis_title="USD",
)
st.plotly_chart(fig_bar, use_container_width=True)

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

# Optional BTC chart (only if uploaded)
if btc_file and include_btc_price_chart:
    st.subheader("₿ Prezzo BTC (da CSV caricato)")
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
df_month_display["elec_cost_usd"] = df_month_display["elec_cost_usd"].round(2)
df_month_display["revenue_usd"] = df_month_display["revenue_usd"].round(2)
df_month_display["profit_usd"] = df_month_display["profit_usd"].round(2)
st.dataframe(df_month_display, use_container_width=True, hide_index=True)

csv_out = df_month_display.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica tabella mensile (CSV)", data=csv_out, file_name="backtest_mensile.csv", mime="text/csv")

if show_debug:
    st.subheader("🧪 Debug – daily")
    st.dataframe(df_daily.head(50), use_container_width=True, hide_index=True)
    st.subheader("🧪 Debug – ercot 15m sample")
    st.dataframe(ercot_15m_period.head(50), use_container_width=True, hide_index=True)

st.caption(
    "Note: i costi elettrici sono calcolati usando i prezzi RTM ($/MWh) e il consumo (kW) su intervalli da 15 minuti. "
    "Le revenue mining usano l'hashprice giornaliero (USD per TH/day) e l'uptime effettivo del giorno."
)
