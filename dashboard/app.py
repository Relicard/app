import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests


# -----------------------------
# CONFIGURAZIONE BASE STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="DESMO Power & Mining Dashboard",
    layout="wide",
)


# -----------------------------
# FUNZIONI DI UTILITY
# -----------------------------
def parse_synota_csv(file) -> pd.DataFrame:
    """Parsa il CSV Synota con i campi: Settlement Start, Energy Delivered, Invoice Amount, Effective Rate."""
    df = pd.read_csv(file)

    # Data
    df["date"] = pd.to_datetime(df["Settlement Start"], format="%m/%d/%Y")

    # Energia kWh -> MWh
    def parse_energy(s):
        if isinstance(s, str):
            s = s.replace("kWh", "").replace(",", "").strip()
            return float(s) if s else None
        return s

    df["energy_kwh"] = df["Energy Delivered"].apply(parse_energy)
    df["energy_mwh"] = df["energy_kwh"] / 1000.0

    # Importi $
    def parse_money(s):
        if isinstance(s, str):
            s = s.replace("$", "").replace(",", "").strip()
            return float(s) if s else None
        return s

    df["invoice_amount_usd"] = df["Invoice Amount"].apply(parse_money)

    # $/MWh
    def parse_rate(s):
        if isinstance(s, str):
            s = (
                s.replace("$", "")
                .replace("/MWh", "")
                .replace(",", "")
                .strip()
            )
            return float(s) if s else None
        return s

    df["effective_rate_usd_per_mwh"] = df["Effective Rate"].apply(parse_rate)

    # Ordina per data
    df = df.sort_values("date")
    return df


def parse_antpool_csv(file) -> pd.DataFrame:
    """Parsa il CSV Antpool con i campi: Earnings date, Daily hashrate, Total Earnings."""
    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["Earnings date"])
    df["daily_hashrate_ths"] = df["Daily hashrate"]
    df["total_earnings_btc"] = df["Total Earnings"]

    df = df.sort_values("date")
    return df


def parse_btc_price_csv(file) -> pd.DataFrame:
    """
    Parsa il CSV del prezzo BTC.
    Si aspetta colonne: 'Date' e 'Close'
    Esempio Date: '01/11/2025 23.58.00'
    Esempio Close: '110267,2'
    """
    df = pd.read_csv(file)

    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    # Parsing data/ora (formato: gg/mm/YYYY HH.MM.SS)
    df["datetime_raw"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    # Teniamo solo la parte di data (00:00:00)
    df["date"] = df["datetime_raw"].dt.normalize()

    # Converte il prezzo con virgola decimale -> punto
    df["btc_price_usd"] = (
        df["Close"]
        .astype(str)
        .str.replace(".", "", regex=False)   # rimuove eventuali separatori migliaia
        .str.replace(",", ".", regex=False)  # virgola -> punto
        .astype(float)
    )

    # Pulisce righe non valide
    df = df.dropna(subset=["date", "btc_price_usd"])

    # Ordina e, se ci sono più righe per la stessa data, tiene l’ultima (es. close giornaliero)
    df = df.sort_values("date")
    df = df.groupby("date", as_index=False)["btc_price_usd"].last()

    return df

def get_btc_price_from_api() -> float | None:
    """
    Recupera il prezzo BTC/USD da CoinGecko.
    In caso di errore, stampa info in console e restituisce None.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}

    try:
        r = requests.get(url, params=params, timeout=10)
        print("BTC API status:", r.status_code)
        if r.status_code != 200:
            print("BTC API response:", r.text)
            return None

        data = r.json()
        price = data.get("bitcoin", {}).get("usd")
        if price is None:
            print("BTC API: campo 'bitcoin.usd' mancante:", data)
            return None

        return float(price)
    except Exception as e:
        print("Errore chiamata BTC API:", e)
        return None



def filter_by_date(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df[mask]



# -----------------------------
# SIDEBAR: CARICAMENTO & FILTRI
# -----------------------------
st.sidebar.title("⚙️ Configurazione")

st.sidebar.subheader("1. Carica i CSV principali")

synota_file = st.sidebar.file_uploader("Synota CSV (energia/costi)", type=["csv"], key="synota")
antpool_file = st.sidebar.file_uploader("Antpool CSV (hashrate/ricavi)", type=["csv"], key="antpool")

synota_df = None
antpool_df = None

if synota_file is not None:
    synota_df = parse_synota_csv(synota_file)

if antpool_file is not None:
    antpool_df = parse_antpool_csv(antpool_file)

# -----------------------------
# Prezzo BTC (attuale / manuale) + CSV storico
# -----------------------------
st.sidebar.subheader("2. Prezzo Bitcoin")

btc_price_mode = st.sidebar.radio(
    "Seleziona modalità",
    options=["Input manuale", "Prezzo di mercato attuale"],
)

btc_price_market = None

if btc_price_mode == "Prezzo di mercato attuale":
    btc_price_market = get_btc_price_from_api()
    if btc_price_market is None:
        st.sidebar.error("❌ Non riesco a recuperare il prezzo BTC da API.\nUsa l'input manuale qui sotto.")
    else:
        st.sidebar.success(f"✅ Prezzo BTC attuale: {btc_price_market:,.2f} USD")

# Input manuale sempre disponibile (override)
manual_btc_price = st.sidebar.number_input(
    "Prezzo BTC/USD (manuale, override)",
    min_value=0.0,
    value=float(btc_price_market) if btc_price_market is not None else 60000.0,
    step=100.0,
)

# Valore effettivo usato nel resto dell'app
btc_price_used = manual_btc_price if manual_btc_price > 0 else btc_price_market


st.sidebar.subheader("3. CSV prezzo BTC (opzionale)")

btc_price_df = None
btc_price_file = st.sidebar.file_uploader(
    "Carica CSV prezzo BTC (Date / Close)",
    type=["csv"],
    key="btc_price_csv",
)

if btc_price_file is not None:
    btc_price_df = parse_btc_price_csv(btc_price_file)
    if not btc_price_df.empty:
        st.sidebar.success(
            f"Prezzo BTC caricato: {btc_price_df['date'].min().date()} → {btc_price_df['date'].max().date()}"
        )
    else:
        st.sidebar.error("CSV prezzo BTC non valido. Controlla che abbia le colonne 'Date' e 'Close'.")


# -----------------------------
# Hosting: CSV clienti + prezzo vendita
# -----------------------------
st.sidebar.subheader("4. Hosting clients")

hosting_price_per_kwh = st.sidebar.number_input(
    "Prezzo vendita elettricità hosting [USD/kWh]",
    min_value=0.0,
    value=0.07,
    step=0.005,
    format="%.4f",
)

hosting_files = st.sidebar.file_uploader(
    "CSV Antpool clienti in hosting (uno o più)",
    type=["csv"],
    accept_multiple_files=True,
    key="hosting_csvs",
)

hosting_df_all = None
if hosting_files:
    frames = []
    for f in hosting_files:
        df_h = parse_antpool_csv(f)
        # Nome cliente = nome file senza estensione
        client_name = f.name.rsplit(".", 1)[0]
        df_h["client"] = client_name
        frames.append(df_h)

    if frames:
        hosting_df_all = pd.concat(frames, ignore_index=True)
        hosting_df_all = hosting_df_all.sort_values("date")


# -----------------------------
# Range date globale (Synota / Antpool)
# -----------------------------
if synota_df is not None or antpool_df is not None:
    all_dates = []
    if synota_df is not None:
        all_dates.extend(list(synota_df["date"]))
    if antpool_df is not None:
        all_dates.extend(list(antpool_df["date"]))

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    st.sidebar.subheader("5. Filtro timeframe")
    date_range = st.sidebar.date_input(
        "Seleziona intervallo di date",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
else:
    start_date = end_date = None

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 DESMO Power & Mining Dashboard")
st.markdown(
    """
Dashboard interattiva per analizzare:
- **Energia & costi** (Synota)  
- **Hashrate & ricavi** (Antpool)  
- **Hosting clienti** (ricavi elettricità)  
- **Metriche combinate** (costo vs ricavo, $/MWh, $/BTC, ecc.)
"""
)


# -----------------------------
# SE NESSUN CSV PRINCIPALE
# -----------------------------
if synota_df is None and antpool_df is None and hosting_df_all is None:
    st.info("👈 Carica almeno un CSV nella sidebar per iniziare.")
    st.stop()


# -----------------------------
# APPLICAZIONE FILTRO DATA
# -----------------------------
if start_date and end_date:
    if synota_df is not None:
        synota_filtered = filter_by_date(synota_df, start_date, end_date)
    else:
        synota_filtered = None

    if antpool_df is not None:
        antpool_filtered = filter_by_date(antpool_df, start_date, end_date)
    else:
        antpool_filtered = None

    if hosting_df_all is not None:
        hosting_filtered = filter_by_date(hosting_df_all, start_date, end_date)
    else:
        hosting_filtered = None
else:
    synota_filtered = synota_df
    antpool_filtered = antpool_df
    hosting_filtered = hosting_df_all


# -----------------------------
# TABS PRINCIPALI
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⚡ Energia (Synota)",
        "⛏️ Mining (Antpool)",
        "🔗 Combined View",
        "📈 Metriche globali",
        "🤝 Hosting",
    ]
)


# -----------------------------
# TAB 1 – ENERGIA / SYNOTA
# -----------------------------
with tab1:
    st.header("⚡ Energia & Costi (Synota)")

    if synota_filtered is None or synota_filtered.empty:
        st.warning("Nessun dato Synota nel range selezionato.")
    else:
        col1, col2, col3 = st.columns(3)
        total_mwh = synota_filtered["energy_mwh"].sum()
        total_cost = synota_filtered["invoice_amount_usd"].sum()
        avg_rate = total_cost / total_mwh if total_mwh > 0 else None

        with col1:
            st.metric("Energia totale [MWh]", f"{total_mwh:,.2f}")
        with col2:
            st.metric("Costo totale energia [USD]", f"{total_cost:,.2f}")
        with col3:
            st.metric("Costo medio [USD/MWh]", f"{avg_rate:,.2f}" if avg_rate else "N/A")

        st.markdown("### 📉 Grafico giornaliero")

        # Selezione campi da visualizzare
        show_energy = st.checkbox("Mostra energia (MWh)", value=True, key="syn_energy")
        show_invoice = st.checkbox("Mostra Invoice Amount [USD]", value=True, key="syn_invoice")
        show_rate = st.checkbox("Mostra Effective Rate [USD/MWh]", value=True, key="syn_rate")

        fig = go.Figure()

        # Assi
        yaxis_config = dict(title="Energia [MWh] / Costo [USD]")
        yaxis2_config = dict(
            title="Tariffa [USD/MWh]",
            overlaying="y",
            side="right",
        )

        if show_energy:
            fig.add_bar(
                x=synota_filtered["date"],
                y=synota_filtered["energy_mwh"],
                name="Energia [MWh]",
            )

        if show_invoice:
            fig.add_bar(
                x=synota_filtered["date"],
                y=synota_filtered["invoice_amount_usd"],
                name="Invoice [USD]",
            )

        if show_rate:
            fig.add_scatter(
                x=synota_filtered["date"],
                y=synota_filtered["effective_rate_usd_per_mwh"],
                name="Tariffa [USD/MWh]",
                mode="lines+markers",
                yaxis="y2",
            )

        fig.update_layout(
            barmode="group",
            xaxis_title="Data",
            yaxis=yaxis_config,
            yaxis2=yaxis2_config,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # GRAFICO 2: CONFRONTO RELATIVO (SERIE INDICIZZATE)
        # -----------------------------
        st.markdown("### 🔍 Confronto relativo energia vs costi (serie indicizzate)")

        synota_rel = synota_filtered.copy()

        # Base = media
        base_energy = synota_rel["energy_mwh"].mean()
        base_cost = synota_rel["invoice_amount_usd"].mean()

        # Evita divisioni per zero
        if base_energy and base_cost:
            synota_rel["energy_index"] = synota_rel["energy_mwh"] / base_energy * 100
            synota_rel["cost_index"] = synota_rel["invoice_amount_usd"] / base_cost * 100

            fig_rel = go.Figure()

            fig_rel.add_scatter(
                x=synota_rel["date"],
                y=synota_rel["energy_index"],
                name="Energia (indice, 100 = media)",
                mode="lines+markers",
            )

            fig_rel.add_scatter(
                x=synota_rel["date"],
                y=synota_rel["cost_index"],
                name="Costo energia (indice, 100 = media)",
                mode="lines+markers",
            )

            fig_rel.update_layout(
                xaxis_title="Data",
                yaxis_title="Indice (100 = media periodo)",
                hovermode="x unified",
            )

            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.info("Impossibile calcolare le serie indicizzate (media energia o costi = 0).")

        st.markdown("### 📄 Dati Synota (filtrati)")
        st.dataframe(
            synota_filtered[
                [
                    "date",
                    "energy_mwh",
                    "invoice_amount_usd",
                    "effective_rate_usd_per_mwh",
                ]
            ].rename(
                columns={
                    "date": "Date",
                    "energy_mwh": "Energy [MWh]",
                    "invoice_amount_usd": "Invoice [USD]",
                    "effective_rate_usd_per_mwh": "Rate [USD/MWh]",
                }
            ),
            use_container_width=True,
        )


# -----------------------------
# TAB 2 – MINING / ANTPOOL
# -----------------------------
with tab2:
    st.header("⛏️ Hashrate & Ricavi (Antpool)")

    if antpool_filtered is None or antpool_filtered.empty:
        st.warning("Nessun dato Antpool nel range selezionato.")
    else:
        # Copia locale
        antpool_with_price = antpool_filtered.copy()

        # -----------------------------
        # Applica prezzo BTC per ogni giorno
        # -----------------------------
        if btc_price_df is not None and not btc_price_df.empty:
            # merge con prezzi storici per data
            antpool_with_price = antpool_with_price.merge(
                btc_price_df,  # colonne: date, btc_price_usd
                on="date",
                how="left",
            )
            # fallback: se un giorno non ha prezzo nel CSV, usa il prezzo attuale / manuale
            fallback_price = btc_price_used if btc_price_used else 0.0
            antpool_with_price["btc_price_usd"] = antpool_with_price["btc_price_usd"].fillna(fallback_price)
        else:
            # Nessun CSV caricato → un solo prezzo per tutto il periodo (attuale o manuale)
            fallback_price = btc_price_used if btc_price_used else 0.0
            antpool_with_price["btc_price_usd"] = fallback_price

        # Ricavi in USD per giorno
        antpool_with_price["earnings_usd"] = (
            antpool_with_price["total_earnings_btc"] * antpool_with_price["btc_price_usd"]
        )

        # -----------------------------
        # Metriche aggregate
        # -----------------------------
        total_btc = antpool_with_price["total_earnings_btc"].sum()
        avg_hashrate = antpool_with_price["daily_hashrate_ths"].mean()

        if antpool_with_price["earnings_usd"].notna().any():
            total_revenue_usd = antpool_with_price["earnings_usd"].sum()
            valid_days = antpool_with_price["earnings_usd"].notna().sum()
            avg_daily_revenue_usd = total_revenue_usd / valid_days if valid_days > 0 else None
        else:
            total_revenue_usd = None
            avg_daily_revenue_usd = None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BTC totali minati", f"{total_btc:.6f}")
        with col2:
            st.metric("Hashrate medio [TH/s]", f"{avg_hashrate:,.2f}")
        with col3:
            if total_revenue_usd is not None:
                st.metric("Ricavi totali [USD]", f"{total_revenue_usd:,.2f}")
            else:
                st.metric("Ricavi totali [USD]", "N/A")

        # -----------------------------
        # 3) Grafico giornaliero
        # -----------------------------
        st.markdown("### 📉 Grafico giornaliero")

        show_hashrate = st.checkbox("Mostra hashrate [TH/s]", value=True, key="ant_hash")
        show_btc = st.checkbox("Mostra ricavi BTC", value=True, key="ant_btc")
        show_usd = st.checkbox("Mostra ricavi USD", value=True, key="ant_usd")

        fig2 = go.Figure()

        # Barre: hashrate
        if show_hashrate:
            fig2.add_bar(
                x=antpool_with_price["date"],
                y=antpool_with_price["daily_hashrate_ths"],
                name="Hashrate [TH/s]",
            )

        # Linee: BTC e USD
        if show_btc:
            fig2.add_scatter(
                x=antpool_with_price["date"],
                y=antpool_with_price["total_earnings_btc"],
                name="Ricavi BTC",
                mode="lines+markers",
                yaxis="y2",
            )

        if show_usd and antpool_with_price["earnings_usd"].notna().any():
            fig2.add_scatter(
                x=antpool_with_price["date"],
                y=antpool_with_price["earnings_usd"],
                name="Ricavi USD",
                mode="lines+markers",
                yaxis="y2",
            )

        fig2.update_layout(
            xaxis_title="Data",
            yaxis=dict(title="Hashrate [TH/s]"),
            yaxis2=dict(
                title="Ricavi [BTC / USD]",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # 4) Tabella dati
        # -----------------------------
        st.markdown("### 📄 Dati Antpool (filtrati)")
        st.dataframe(
            antpool_with_price[
                [
                    "date",
                    "daily_hashrate_ths",
                    "total_earnings_btc",
                    "btc_price_usd",
                    "earnings_usd",
                ]
            ].rename(
                columns={
                    "date": "Date",
                    "daily_hashrate_ths": "Hashrate [TH/s]",
                    "total_earnings_btc": "Earnings [BTC]",
                    "btc_price_usd": "BTC price [USD]",
                    "earnings_usd": "Earnings [USD]",
                }
            ),
            use_container_width=True,
        )



# -----------------------------
# TAB 3 – COMBINED VIEW
# -----------------------------
with tab3:
    st.header("🔗 Vista combinata energia vs ricavi")

    if synota_filtered is None or synota_filtered.empty or antpool_filtered is None or antpool_filtered.empty:
        st.warning("Servono **entrambi** i CSV (Synota + Antpool) con date sovrapposte.")
    else:
        combined = pd.merge(
            synota_filtered,
            antpool_filtered,
            on="date",
            how="inner",
            suffixes=("_synota", "_antpool"),
        )

        if combined.empty:
            st.warning("Nessun giorno in cui Synota e Antpool hanno entrambi dati (dopo il filtro date).")
        else:
            # Applica prezzo BTC per giorno (CSV + fallback)
            if btc_price_df is not None and not btc_price_df.empty:
                combined = combined.merge(btc_price_df, on="date", how="left")
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined["btc_price_usd"] = combined["btc_price_usd"].fillna(fallback_price)
            else:
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined["btc_price_usd"] = fallback_price

            combined["earnings_usd"] = combined["total_earnings_btc"] * combined["btc_price_usd"]

            # Metriche base
            total_mwh_c = combined["energy_mwh"].sum()
            total_cost_c = combined["invoice_amount_usd"].sum()
            total_btc_c = combined["total_earnings_btc"].sum()
            total_rev_usd_c = combined["earnings_usd"].sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Energia totale [MWh]", f"{total_mwh_c:,.2f}")
            with col2:
                st.metric("Costo totale energia [USD]", f"{total_cost_c:,.2f}")
            with col3:
                st.metric("BTC totali minati", f"{total_btc_c:.6f}")
            with col4:
                st.metric("Ricavi totali [USD]", f"{total_rev_usd_c:,.2f}")

            # Metriche avanzate
            st.markdown("### 📌 Metriche chiave")
            colA, colB, colC = st.columns(3)

            with colA:
                # Costo medio $/MWh (pesato)
                avg_rate_c = total_cost_c / total_mwh_c if total_mwh_c > 0 else None
                st.write("**Costo medio energia [USD/MWh]**")
                st.write(f"{avg_rate_c:,.2f}" if avg_rate_c else "N/A")

            with colB:
                # Costo per BTC
                cost_per_btc = total_cost_c / total_btc_c if total_btc_c > 0 else None
                st.write("**Costo energia per 1 BTC [USD/BTC]**")
                st.write(f"{cost_per_btc:,.2f}" if cost_per_btc else "N/A")

            with colC:
                profit_usd = total_rev_usd_c - total_cost_c
                st.write("**Margine lordo totale [USD]**")
                st.write(f"{profit_usd:,.2f}")

            st.markdown("### 📉 Grafico combinato (Costi vs Ricavi)")

            show_cost = st.checkbox("Mostra costo energia [USD]", value=True, key="comb_cost")
            show_rev = st.checkbox("Mostra ricavi [USD]", value=True, key="comb_rev")
            show_rate_mwh = st.checkbox("Mostra $/MWh", value=True, key="comb_rate")
            show_btc_day = st.checkbox("Mostra BTC/day", value=False, key="comb_btcday")

            fig3 = go.Figure()

            # Asse sinistro: costi/ricavi
            if show_cost:
                fig3.add_bar(
                    x=combined["date"],
                    y=combined["invoice_amount_usd"],
                    name="Costo energia [USD]",
                )
            if show_rev:
                fig3.add_bar(
                    x=combined["date"],
                    y=combined["earnings_usd"],
                    name="Ricavi [USD]",
                )

            # Asse destro: rate & BTC/day
            if show_rate_mwh:
                fig3.add_scatter(
                    x=combined["date"],
                    y=combined["effective_rate_usd_per_mwh"],
                    name="Tariffa [USD/MWh]",
                    mode="lines+markers",
                    yaxis="y2",
                )

            if show_btc_day:
                fig3.add_scatter(
                    x=combined["date"],
                    y=combined["total_earnings_btc"],
                    name="BTC/day",
                    mode="lines+markers",
                    yaxis="y2",
                )

            fig3.update_layout(
                barmode="group",
                xaxis_title="Data",
                yaxis=dict(title="USD (Costi / Ricavi)"),
                yaxis2=dict(
                    title="Tariffa [USD/MWh] / BTC/day",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### 📄 Dati combinati (per giorno)")
            show_cols = [
                "date",
                "energy_mwh",
                "invoice_amount_usd",
                "effective_rate_usd_per_mwh",
                "daily_hashrate_ths",
                "total_earnings_btc",
                "btc_price_usd",
                "earnings_usd",
            ]
            st.dataframe(
                combined[show_cols].rename(
                    columns={
                        "date": "Date",
                        "energy_mwh": "Energy [MWh]",
                        "invoice_amount_usd": "Cost [USD]",
                        "effective_rate_usd_per_mwh": "Rate [USD/MWh]",
                        "daily_hashrate_ths": "Hashrate [TH/s]",
                        "total_earnings_btc": "Earnings [BTC]",
                        "btc_price_usd": "BTC price [USD]",
                        "earnings_usd": "Earnings [USD]",
                    }
                ),
                use_container_width=True,
            )



# -----------------------------
# TAB 4 – METRICHE GLOBALI
# -----------------------------
with tab4:
    st.header("📈 Metriche globali riassuntive")

    # Energia (Synota)
    if synota_filtered is not None and not synota_filtered.empty:
        st.subheader("⚡ Energia (Synota)")
        total_mwh = synota_filtered["energy_mwh"].sum()
        total_cost = synota_filtered["invoice_amount_usd"].sum()
        avg_rate = total_cost / total_mwh if total_mwh > 0 else None

        st.write(f"- Giorni considerati: **{len(synota_filtered)}**")
        st.write(f"- Energia totale: **{total_mwh:,.2f} MWh**")
        st.write(f"- Costo totale energia: **{total_cost:,.2f} USD**")
        st.write(
            f"- Costo medio: **{avg_rate:,.2f} USD/MWh**"
            if avg_rate
            else "- Costo medio: N/A"
        )

    # Mining (Antpool)
    if antpool_filtered is not None and not antpool_filtered.empty:
        st.subheader("⛏️ Mining (Antpool)")
        total_btc = antpool_filtered["total_earnings_btc"].sum()
        avg_hashrate = antpool_filtered["daily_hashrate_ths"].mean()

        st.write(f"- Giorni considerati: **{len(antpool_filtered)}**")
        st.write(f"- BTC totali minati: **{total_btc:.6f} BTC**")
        st.write(f"- Hashrate medio: **{avg_hashrate:,.2f} TH/s**")

        # Ricavi totali stimati con CSV + fallback
        if btc_price_df is not None and not btc_price_df.empty:
            tmp = antpool_filtered.merge(btc_price_df, on="date", how="left")
            fallback_price = btc_price_used if btc_price_used else 0.0
            tmp["btc_price_usd"] = tmp["btc_price_usd"].fillna(fallback_price)
            total_rev_usd = (tmp["total_earnings_btc"] * tmp["btc_price_usd"]).sum()
        else:
            fallback_price = btc_price_used if btc_price_used else 0.0
            total_rev_usd = total_btc * fallback_price

        st.write(f"- Ricavi totali stimati: **{total_rev_usd:,.2f} USD**")


    # Rapporto energia vs mining
    if (synota_filtered is not None and not synota_filtered.empty) and (
        antpool_filtered is not None and not antpool_filtered.empty
    ):
        st.subheader("🔗 Rapporto energia vs mining")
        combined = pd.merge(
            synota_filtered,
            antpool_filtered,
            on="date",
            how="inner",
            suffixes=("_synota", "_antpool"),
        )

        if not combined.empty:
            total_mwh = combined["energy_mwh"].sum()
            total_cost = combined["invoice_amount_usd"].sum()
            total_btc = combined["total_earnings_btc"].sum()

            if total_btc > 0:
                cost_per_btc = total_cost / total_btc
                st.write(f"- Costo energia per 1 BTC: **{cost_per_btc:,.2f} USD/BTC**")
            else:
                st.write("- Costo energia per 1 BTC: N/A")

            if btc_price_df is not None and not btc_price_df.empty:
                combined = combined.merge(btc_price_df, on="date", how="left")
                combined["btc_price_usd"] = combined["btc_price_usd"].fillna(btc_price_used if btc_price_used > 0 else 0.0)
            else:
                combined["btc_price_usd"] = btc_price_used if btc_price_used > 0 else 0.0

            total_rev_usd = (combined["total_earnings_btc"] * combined["btc_price_usd"]).sum()
            profit = total_rev_usd - total_cost
            st.write(f"- Margine lordo totale (ricavi - costi): **{profit:,.2f} USD**")


# -----------------------------
# TAB 5 – HOSTING CLIENTI
# -----------------------------
with tab5:
    st.header("🤝 Hosting clienti – potenza e ricavi elettricità")

    if hosting_filtered is None or hosting_filtered.empty:
        st.info("Nessun CSV hosting caricato nella sidebar.")
    else:
        # Costanti modello hosting
        ASIC_TH = 200.0      # TH/s per S21
        ASIC_KW = 3.5        # kW per S21
        HOURS_PER_DAY = 24.0

        df_host = hosting_filtered.copy()

        # Calcoli base per giorno/cliente
        df_host["asic_equivalent"] = df_host["daily_hashrate_ths"] / ASIC_TH
        df_host["energy_kwh"] = df_host["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
        df_host["hosting_revenue_usd"] = df_host["energy_kwh"] * hosting_price_per_kwh

        # Metriche DESMO globali hosting
        total_clients = df_host["client"].nunique()
        total_asic_eq = df_host["asic_equivalent"].mean()
        total_energy_mwh = df_host["energy_kwh"].sum() / 1000.0
        total_hosting_rev = df_host["hosting_revenue_usd"].sum()
        total_btc_host = df_host["total_earnings_btc"].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clienti in hosting", str(total_clients))
        with col2:
            st.metric("ASIC S21 equivalenti medi", f"{total_asic_eq:,.2f}")
        with col3:
            st.metric("Energia fatturata [MWh]", f"{total_energy_mwh:,.2f}")
        with col4:
            st.metric("Ricavi hosting totali [USD]", f"{total_hosting_rev:,.2f}")

        st.markdown("### 👤 Seleziona cliente per dettaglio")

        client_options = ["Tutti"] + sorted(df_host["client"].unique().tolist())
        selected_client = st.selectbox("Cliente", client_options)

        if selected_client != "Tutti":
            df_view = df_host[df_host["client"] == selected_client].copy()
        else:
            # aggregato per data
            df_view = df_host.groupby("date", as_index=False).agg(
                {
                    "daily_hashrate_ths": "sum",
                    "total_earnings_btc": "sum",
                    "asic_equivalent": "sum",
                    "energy_kwh": "sum",
                    "hosting_revenue_usd": "sum",
                }
            )

        st.markdown("### 📉 Hosting: BTC minati vs ricavi elettricità")

        show_host_rev = st.checkbox("Mostra ricavi hosting [USD]", value=True, key="host_rev")
        show_host_btc = st.checkbox("Mostra BTC minati", value=True, key="host_btc")

        fig_host = go.Figure()

        if show_host_rev:
            fig_host.add_bar(
                x=df_view["date"],
                y=df_view["hosting_revenue_usd"],
                name="Ricavi hosting [USD]",
            )

        if show_host_btc:
            fig_host.add_scatter(
                x=df_view["date"],
                y=df_view["total_earnings_btc"],
                name="BTC minati",
                mode="lines+markers",
                yaxis="y2",
            )

        fig_host.update_layout(
            xaxis_title="Data",
            yaxis=dict(title="Ricavi hosting [USD]"),
            yaxis2=dict(
                title="BTC minati",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig_host, use_container_width=True)

        st.markdown("### 📊 Riepilogo per cliente")

        # Tabella per cliente (aggregata)
        summary_clients = df_host.groupby("client", as_index=False).agg(
            {
                "daily_hashrate_ths": "mean",
                "asic_equivalent": "mean",
                "energy_kwh": "sum",
                "hosting_revenue_usd": "sum",
                "total_earnings_btc": "sum",
            }
        )

        summary_clients["energy_mwh"] = summary_clients["energy_kwh"] / 1000.0

        st.dataframe(
            summary_clients[
                [
                    "client",
                    "daily_hashrate_ths",
                    "asic_equivalent",
                    "energy_mwh",
                    "hosting_revenue_usd",
                    "total_earnings_btc",
                ]
            ].rename(
                columns={
                    "client": "Client",
                    "daily_hashrate_ths": "Avg Hashrate [TH/s]",
                    "asic_equivalent": "Avg ASIC S21 eq.",
                    "energy_mwh": "Energy billed [MWh]",
                    "hosting_revenue_usd": "Hosting revenue [USD]",
                    "total_earnings_btc": "BTC mined (period)",
                }
            ),
            use_container_width=True,
        )
