import io
import datetime as dt

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

# --- Proviamo a importare gridstatus (ERCOT ufficiale) ---
try:
    import gridstatus
except ImportError:
    gridstatus = None

# ---------------------------
# FUNZIONI DI SUPPORTO
# ---------------------------

def load_antpool_income_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """
    Legge il CSV di Antpool 'Income List' come quello che mi hai mandato:
    colonne: 'Earnings date', 'Daily hashrate', ecc.
    Converte in una serie oraria con ffill (hashrate costante per giorno).
    """
    df = pd.read_csv(uploaded_file)

    # Normalizziamo nomi colonne per sicurezza
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    if "Earnings date" not in df.columns or "Daily hashrate" not in df.columns:
        raise ValueError(
            "CSV Antpool non nel formato atteso: devo trovare 'Earnings date' e 'Daily hashrate'."
        )

    df["Earnings date"] = pd.to_datetime(df["Earnings date"])
    df = df.sort_values("Earnings date")

    # hashrate in TH/s (già così in Antpool)
    df["hashrate_ths"] = pd.to_numeric(df["Daily hashrate"], errors="coerce")

    # Costruiamo serie giornaliera -> oraria con ffill
    hashrate_daily = (
        df[["Earnings date", "hashrate_ths"]]
        .rename(columns={"Earnings date": "Time"})
        .set_index("Time")
        .sort_index()
    )

    # Resample orario: ogni giorno viene "spalmato" su 24 ore
    hashrate_hourly = (
        hashrate_daily
        .resample("H")
        .ffill()
    )

    hashrate_hourly = hashrate_hourly.reset_index()

    return hashrate_hourly


def load_ercot_prices_lz_west(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Usa gridstatus.Ercot().get_spp per scaricare i prezzi REAL_TIME_15_MIN di LZ_WEST
    dall'inizio di 'start' alla fine di 'end', e li porta a media oraria.
    """
    if gridstatus is None:
        raise RuntimeError(
            "La libreria 'gridstatus' non è installata. "
            "Installa con: pip install gridstatus"
        )

    iso = gridstatus.Ercot()

    # Assicuriamoci che siano Timestamp
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # get_spp con intervallo [start, end]
    # market: REAL_TIME_15_MIN; location_type: ZONE; poi filtriamo LZ_WEST
    spp = iso.get_spp(
        date=start,
        end=end,
        market="REAL_TIME_15_MIN",
        location_type="zone",
    )

    # Filtra solo LZ_WEST
    spp = spp[spp["Location"] == "LZ_WEST"].copy()

    if spp.empty:
        return pd.DataFrame(columns=["Time", "ercot_price"])

    # Time è già in US/Central con timezone; togliamo il tz per allinearci a Serie Antpool
    spp["Time"] = pd.to_datetime(spp["Time"])
    if spp["Time"].dt.tz is not None:
        spp["Time"] = spp["Time"].dt.tz_convert("US/Central").dt.tz_localize(None)

    # Media oraria
    price_hourly = (
        spp.set_index("Time")["SPP"]
        .resample("H")
        .mean()
        .rename("ercot_price")
        .to_frame()
        .reset_index()
    )

    return price_hourly


def build_merged_df(hashrate_hourly: pd.DataFrame, price_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per ora su Time tra hashrate e prezzo ERCOT.
    """
    # Allineiamo colonne & ordini
    h = hashrate_hourly.copy()
    p = price_hourly.copy()

    h["Time"] = pd.to_datetime(h["Time"])
    p["Time"] = pd.to_datetime(p["Time"])

    h = h.sort_values("Time")
    p = p.sort_values("Time")

    merged = pd.merge_asof(
        p,                        # sinistra: prezzi
        h,                        # destra: hashrate
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=30),
    )

    return merged


def compute_metrics(merged: pd.DataFrame) -> dict:
    """
    Calcola uptime e correlazione prezzo-hashrate.
    """
    m = merged.copy()

    # Consideriamo solo righe dove abbiamo sia prezzo che hashrate
    m = m.dropna(subset=["ercot_price", "hashrate_ths"])
    if m.empty:
        return {
            "uptime_pct": None,
            "corr": None,
            "n_hours": 0,
        }

    # Definiamo "online" se hashrate > 0
    m["online"] = m["hashrate_ths"] > 0

    total_hours = len(m)
    online_hours = m["online"].sum()
    uptime_pct = online_hours / total_hours * 100 if total_hours > 0 else None

    if m["hashrate_ths"].nunique() <= 1:
        corr = None
    else:
        corr = m[["ercot_price", "hashrate_ths"]].corr().iloc[0, 1]

    return {
        "uptime_pct": uptime_pct,
        "corr": corr,
        "n_hours": total_hours,
    }


def make_plot(merged: pd.DataFrame):
    """
    Plotly fig con doppio asse:
    - ERCOT LZ_WEST ($/MWh)
    - Hashrate (PH/s)
    con range slider e hover figo.
    """
    if merged.empty:
        st.warning("Nessun dato da mostrare nel grafico.")
        return

    df = merged.dropna(subset=["ercot_price", "hashrate_ths"]).copy()
    if df.empty:
        st.warning("Non ci sono abbastanza punti con sia prezzo che hashrate.")
        return

    # Convertiamo TH/s -> PH/s per avere numeri più leggibili
    df["hashrate_phs"] = df["hashrate_ths"] / 1_000.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["ercot_price"],
            name="ERCOT LZ_WEST ($/MWh)",
            mode="lines",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["hashrate_phs"],
            name="Hashrate (PH/s)",
            mode="lines",
            opacity=0.7,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="DESMO – Hashrate vs Prezzo ERCOT LZ_WEST",
        hovermode="x unified",
        xaxis_title="Tempo (US Central)",
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Prezzo ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="Hashrate (PH/s)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# APP STREAMLIT
# ---------------------------

st.set_page_config(page_title="DESMO – Hashrate vs ERCOT", layout="wide")

st.title("📈 DESMO – Hashrate Antpool vs Prezzo ERCOT LZ_WEST")

st.markdown(
    """
Carica il CSV Antpool (Income List, con **Earnings date** e **Daily hashrate**).  
L'app scarica i prezzi ERCOT LZ_WEST (REAL_TIME_15_MIN) **da inizio novembre ad oggi**,  
li porta a media oraria e li sovrappone all'hashrate.
"""
)

uploaded_file = st.file_uploader("Carica CSV Antpool", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Carica il CSV per iniziare.")
    st.stop()

# --- 1) CSV ANTPPOOL -> HASHRATE ORARIO ---
try:
    hashrate_hourly = load_antpool_income_csv(uploaded_file)
except Exception as e:
    st.error(f"Errore nel leggere il CSV Antpool: {e}")
    st.stop()

if hashrate_hourly.empty:
    st.error("Il CSV Antpool non contiene dati validi.")
    st.stop()

hash_start = hashrate_hourly["Time"].min()
hash_end = hashrate_hourly["Time"].max()

st.success(
    f"CSV Antpool caricato correttamente: {len(hashrate_hourly)} punti orari, "
    f"dal {hash_start} al {hash_end}."
)

# --- 2) ERCOT LZ_WEST STORICO DA INIZIO NOVEMBRE AD OGGI ---
today_central = dt.datetime.now(tz=dt.timezone.utc).astimezone(
    dt.timezone(dt.timedelta(hours=-6))
)  # US Central approx

start_ercot = max(pd.to_datetime("2025-11-01"), hash_start)
end_ercot = max(hash_end, pd.to_datetime("2025-11-01")) + pd.Timedelta(days=1)

st.write(
    f"Scarico prezzi ERCOT **LZ_WEST** dal **{start_ercot}** al **{end_ercot}** "
    "(REAL_TIME_15_MIN → media oraria)…"
)

try:
    price_hourly = load_ercot_prices_lz_west(start_ercot, end_ercot)
except Exception as e:
    st.error(f"Errore nel download dei prezzi ERCOT: {e}")
    st.stop()

if price_hourly.empty:
    st.error("Non ho trovato dati di prezzo ERCOT per l'intervallo richiesto.")
    st.stop()

p_start = price_hourly["Time"].min()
p_end = price_hourly["Time"].max()

st.success(
    f"Prezzi ERCOT LZ_WEST scaricati: {len(price_hourly)} ore, "
    f"dal {p_start} al {p_end}."
)

# --- 3) MERGE + METRICHE ---
merged = build_merged_df(hashrate_hourly, price_hourly)

metrics = compute_metrics(merged)

st.subheader("📊 Metriche sintetiche")

col1, col2, col3 = st.columns(3)

with col1:
    if metrics["uptime_pct"] is not None:
        st.metric(
            "Uptime (su ore con prezzo disponibile)",
            f"{metrics['uptime_pct']:.1f} %",
        )
    else:
        st.metric("Uptime", "N/D")

with col2:
    if metrics["corr"] is not None:
        st.metric(
            "Correlazione prezzo ↔ hashrate",
            f"{metrics['corr']:.3f}",
        )
    else:
        st.metric("Correlazione prezzo ↔ hashrate", "N/D")

with col3:
    st.metric("Ore considerate", f"{metrics['n_hours']}")

st.markdown("---")

st.subheader("📈 Grafico navigabile")

make_plot(merged)

st.caption(
    "📌 Nota: l'hashrate giornaliero di Antpool viene interpolato come profilo orario costante per giorno; "
    "i prezzi ERCOT sono media oraria dei 15-minuti RTM LZ_WEST."
)
