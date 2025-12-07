import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from gridstatus import Ercot

# -------------------------------
# CONFIGURAZIONE BASE STREAMLIT
# -------------------------------
st.set_page_config(
    page_title="ERCOT LZ_WEST - Prezzi medi giornalieri",
    layout="wide",
)

st.title("📈 ERCOT LZ_WEST – Prezzi medi giornalieri (RTM SPP)")
st.markdown(
    """
Questa pagina scarica i **Real-Time Settlement Point Prices (SPP)** storici di ERCOT  
(per Load Zone **LZ_WEST**) e calcola il **prezzo medio giornaliero**.

- Fonte dati: report ufficiale ERCOT *NP6-785-ER* (via libreria `gridstatus`)
- Market: **Real-Time 15 min (RTM SPP)**
"""
)

# -----------------------------------
# FUNZIONE PER CARICARE I DATI ERCOT
# -----------------------------------
@st.cache_data(show_spinner="📡 Scarico i dati ERCOT dalla fonte ufficiale…")
def load_ercot_lz_west_daily(year: int) -> pd.DataFrame:
    """
    Scarica gli SPP RTM per tutto l'anno e calcola il prezzo medio giornaliero
    per la Load Zone LZ_WEST.
    """
    iso = Ercot()

    # get_rtm_spp usa il data product NP6-785-ER (file .zip intero anno)
    df = iso.get_rtm_spp(year=year, verbose=False)

    # Ci aspettiamo colonne tipo: Time, Location, Location Type, Market, SPP
    # Filtra sulla Load Zone LZ_WEST
    df_lz_west = df[df["Location"] == "LZ_WEST"].copy()

    if df_lz_west.empty:
        return pd.DataFrame(columns=["Date", "Price"])

    # Converte Time in datetime se non lo è già
    if not pd.api.types.is_datetime64_any_dtype(df_lz_west["Time"]):
        df_lz_west["Time"] = pd.to_datetime(df_lz_west["Time"])

    # Usa solo la data (no orario) per aggregare a livello giornaliero
    df_lz_west["Date"] = df_lz_west["Time"].dt.date

    # Calcola il prezzo medio giornaliero ($/MWh)
    daily = (
        df_lz_west.groupby("Date")["SPP"]
        .mean()
        .reset_index()
        .rename(columns={"SPP": "Price"})
    )

    # Riconverte Date a Timestamp per Plotly
    daily["Date"] = pd.to_datetime(daily["Date"])

    # Ordina per data
    daily = daily.sort_values("Date").reset_index(drop=True)

    return daily


# -----------------------------
# SELEZIONE PARAMETRI UTENTE
# -----------------------------
current_year = datetime.now().year

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox(
        "Seleziona l'anno",
        options=list(range(2011, current_year + 1)),
        index=list(range(2011, current_year + 1)).index(current_year),
        help="ERCOT fornisce i dati RTM SPP storici per anno, a partire dal 2011.",
    )

with col2:
    default_days = 60  # di default ultimi 60 giorni
    max_days = 365
    days_window = st.slider(
        "Quanti giorni mostrare (a ritroso dall'ultima data disponibile)",
        min_value=7,
        max_value=max_days,
        value=default_days,
    )

# -----------------------------
# CARICAMENTO DATI
# -----------------------------
with st.spinner("⏳ Carico e processo i dati, potrebbe volerci qualche secondo…"):
    daily_df = load_ercot_lz_west_daily(year)

if daily_df.empty:
    st.error(
        "Non sono riuscito a trovare dati per LZ_WEST in questo anno. "
        "Prova a selezionare un altro anno."
    )
    st.stop()

# Limita alla finestra temporale scelta
last_date = daily_df["Date"].max()
first_date_shown = last_date - timedelta(days=days_window - 1)
filtered = daily_df[daily_df["Date"].between(first_date_shown, last_date)]

# -----------------------------
# METRICHE SINTETICHE
# -----------------------------
st.subheader("📊 Metriche sintetiche")

metric_col1, metric_col2, metric_col3 = st.columns(3)

avg_price = filtered["Price"].mean()
min_price = filtered["Price"].min()
max_price = filtered["Price"].max()

metric_col1.metric(
    "Prezzo medio (periodo mostrato)",
    f"{avg_price:.2f} $/MWh",
)
metric_col2.metric(
    "Min giornaliero",
    f"{min_price:.2f} $/MWh",
)
metric_col3.metric(
    "Max giornaliero",
    f"{max_price:.2f} $/MWh",
)

st.caption(
    f"Dati mostrati da **{filtered['Date'].min().date()}** a **{filtered['Date'].max().date()}** "
    f"(anno {year}, zona **LZ_WEST**)."
)

# -----------------------------
# GRAFICO INTERATTIVO PLOTLY
# -----------------------------
st.subheader("📉 Prezzo medio giornaliero LZ_WEST (RTM SPP)")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["Date"],
        y=filtered["Price"],
        mode="lines+markers",
        name="Prezzo medio giornaliero",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} $/MWh<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Prezzo medio giornaliero ($/MWh)",
    hovermode="x unified",
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="7g", step="day", stepmode="backward"),
                    dict(count=30, label="30g", step="day", stepmode="backward"),
                    dict(count=90, label="90g", step="day", stepmode="backward"),
                    dict(step="all", label="Tutto"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    ),
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TABELLA RAW (OPZIONALE)
# -----------------------------
with st.expander("📄 Vedi tabella dati grezzi (giornalieri)"):
    st.dataframe(
        filtered.sort_values("Date").reset_index(drop=True),
        use_container_width=True,
    )

st.caption(
    "Nota: i prezzi sono **Real-Time Settlement Point Prices (RTM SPP)** "
    "per la Load Zone **LZ_WEST**, aggregati a media giornaliera."
)
