import streamlit as st # type: ignore 
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
from datetime import datetime, timedelta  # type: ignore
import requests # type: ignore 



# -----------------------------
# CONFIGURAZIONE BASE STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="DESMO Power & Mining Dashboard",
    layout="wide",
)

# Stato per memorizzare il prezzo medio di vendita selezionato nella tab Vendite BTC
if "selected_sell_avg_price" not in st.session_state:
    st.session_state["selected_sell_avg_price"] = None


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


DESMO_WALLET_ADDRESS = "bc1qtu7w3e67xwudtv6ddq4ru9ppw9p63dnd3jph2q"
KRAKEN_DEPOSIT_ADDRESS = "36dHd14U5a8F2yckQC6cadR1HSUbWwUr5c"


def parse_kraken_trades_csv(file) -> pd.DataFrame:
    """
    Parsa il CSV trade Kraken.
    Filtra solo le vendite BTC/USD e calcola BTC venduti e USD incassati.
    """
    df = pd.read_csv(file)

    # Tieni solo BTC/USD e type == sell
    df = df[(df["pair"] == "BTC/USD") & (df["type"] == "sell")].copy()

    # Parsing timestamp
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["date"] = df["time"].dt.date

    # BTC venduti (vol è in BTC)
    df["btc_sold"] = df["vol"].astype(float)

    # Incasso in USD: usa costusd se presente, altrimenti cost
    if "costusd" in df.columns:
        df["usd_proceeds"] = df["costusd"].astype(float)
    else:
        df["usd_proceeds"] = df["cost"].astype(float)

    df = df.sort_values("time")
    return df


def get_wallet_balance_btc(address: str) -> float | None:
    """
    Restituisce il saldo BTC (in BTC) di un address usando Blockstream API.
    Usa i campi chain_stats + mempool_stats.
    """
    url = f"https://blockstream.info/api/address/{address}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print("Wallet API error:", r.status_code, r.text)
            return None

        data = r.json()
        chain_stats = data.get("chain_stats", {})
        mempool_stats = data.get("mempool_stats", {})

        funded = chain_stats.get("funded_txo_sum", 0) + mempool_stats.get("funded_txo_sum", 0)
        spent = chain_stats.get("spent_txo_sum", 0) + mempool_stats.get("spent_txo_sum", 0)

        balance_sats = funded - spent
        return balance_sats / 1e8  # BTC
    except Exception as e:
        print("Errore wallet API:", e)
        return None
    

def get_wallet_outgoing_txs(address: str) -> pd.DataFrame:
    """
    Restituisce tutte le transazioni in uscita da un address usando Blockstream API.
    Ritorna un DataFrame con: time, date, txid, recipient, amount_btc.
    """
    url = f"https://blockstream.info/api/address/{address}/txs"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print("Wallet TX API error:", r.status_code, r.text)
            return pd.DataFrame()

        txs = r.json()
        rows = []

        for tx in txs:
            txid = tx.get("txid")
            status = tx.get("status", {})
            block_time = status.get("block_time")

            dt = datetime.fromtimestamp(block_time) if block_time is not None else None

            # Ogni output è un possibile destinatario
            for vout in tx.get("vout", []):
                addr_out = vout.get("scriptpubkey_address")
                val_sats = vout.get("value", 0)

                # Consideriamo solo output con address valido, diverso dal nostro e importo > 0
                if addr_out and addr_out != address and val_sats > 0:
                    rows.append(
                        {
                            "txid": txid,
                            "time": dt,
                            "date": dt.date() if dt else None,
                            "recipient": addr_out,
                            "amount_btc": val_sats / 1e8,
                        }
                    )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("time")

        return df

    except Exception as e:
        print("Errore wallet TX API:", e)
        return pd.DataFrame()




def filter_by_date(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df[mask]



# -----------------------------
# SIDEBAR: CARICAMENTO & FILTRI
# -----------------------------
st.sidebar.title("⚙️ Configurazione")

# 1. CSV principali
st.sidebar.subheader("1. Carica i CSV principali")

synota_file = st.sidebar.file_uploader("Synota CSV (energia/costi)", type=["csv"], key="synota")
antpool_file = st.sidebar.file_uploader("Antpool CSV (hashrate/ricavi)", type=["csv"], key="antpool")

synota_df = None
antpool_df = None

if synota_file is not None:
    synota_df = parse_synota_csv(synota_file)

if antpool_file is not None:
    antpool_df = parse_antpool_csv(antpool_file)


# 2. Vendite BTC (Kraken) – così possiamo usare il prezzo medio di vendita
st.sidebar.subheader("2. Vendite BTC (Kraken)")

kraken_file = st.sidebar.file_uploader(
    "CSV vendite BTC (esportazione Kraken)",
    type=["csv"],
    key="kraken_trades",
)

kraken_trades_df = None
if kraken_file is not None:
    kraken_trades_df = parse_kraken_trades_csv(kraken_file)


st.sidebar.subheader("3. Prezzo Bitcoin")

btc_price_mode = st.sidebar.radio(
    "Seleziona modalità",
    options=["Input manuale", "Prezzo di mercato attuale", "Prezzo medio di vendita"],
)

btc_price_market = None
btc_price_sell_avg = st.session_state.get("selected_sell_avg_price")

# 1) Modalità: prezzo di mercato attuale
if btc_price_mode == "Prezzo di mercato attuale":
    btc_price_market = get_btc_price_from_api()
    if btc_price_market is None:
        st.sidebar.error("❌ Non riesco a recuperare il prezzo BTC da API.\nUsa l'input manuale qui sotto.")
    else:
        st.sidebar.success(f"✅ Prezzo BTC attuale: {btc_price_market:,.2f} USD")

# 2) Modalità: prezzo medio di vendita (calcolato in tab Vendite BTC)
elif btc_price_mode == "Prezzo medio di vendita":
    if btc_price_sell_avg is not None:
        st.sidebar.success(
            f"✅ Prezzo medio di vendita selezionato: {btc_price_sell_avg:,.2f} USD/BTC"
        )
    else:
        st.sidebar.warning(
            "Nessun prezzo medio di vendita disponibile.\n"
            "Vai nella tab '💰 Vendite BTC' e seleziona alcune transazioni rilevanti."
        )

# Default per l’input manuale
if btc_price_mode == "Prezzo di mercato attuale" and btc_price_market is not None:
    default_manual = float(btc_price_market)
elif btc_price_mode == "Prezzo medio di vendita" and btc_price_sell_avg is not None:
    default_manual = float(btc_price_sell_avg)
else:
    default_manual = 60000.0

# Input manuale sempre disponibile (override)
manual_btc_price = st.sidebar.number_input(
    "Prezzo BTC/USD (manuale, override)",
    min_value=0.0,
    value=default_manual,
    step=100.0,
)

# Valore effettivo usato nel resto dell'app
if manual_btc_price > 0:
    btc_price_used = manual_btc_price
elif btc_price_mode == "Prezzo di mercato attuale":
    btc_price_used = btc_price_market
elif btc_price_mode == "Prezzo medio di vendita":
    btc_price_used = btc_price_sell_avg
else:
    btc_price_used = None



# 4. CSV prezzo BTC (opzionale)
st.sidebar.subheader("4. CSV prezzo BTC (opzionale)")

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


# 5. Hosting: CSV clienti + prezzo vendita
st.sidebar.subheader("5. Hosting clients")

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


# 6. Range date globale (Synota / Antpool)
if synota_df is not None or antpool_df is not None:
    all_dates = []
    if synota_df is not None:
        all_dates.extend(list(synota_df["date"]))
    if antpool_df is not None:
        all_dates.extend(list(antpool_df["date"]))

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    st.sidebar.subheader("6. Filtro timeframe")

    # 👇 Preset rapidi
    preset = st.sidebar.radio(
        "Seleziona periodo rapido",
        [
            "Tutti i dati",
            "Ultima data disponibile",
            "Ultimi 7 giorni",
            "Ultimi 30 giorni",
            "Ultimi 90 giorni",
            "Personalizzato",
        ],
        index=0,
    )

    if preset == "Tutti i dati":
        start_date, end_date = min_date, max_date

    elif preset == "Ultima data disponibile":
        start_date = max_date
        end_date = max_date

    elif preset == "Ultimi 7 giorni":
        start_date = max(min_date, max_date - timedelta(days=6))
        end_date = max_date

    elif preset == "Ultimi 30 giorni":
        start_date = max(min_date, max_date - timedelta(days=29))
        end_date = max_date

    elif preset == "Ultimi 90 giorni":
        start_date = max(min_date, max_date - timedelta(days=89))
        end_date = max_date

    else:  # "Personalizzato"
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

    # Piccolo riepilogo del range scelto
    st.sidebar.caption(f"Range selezionato: {start_date} → {end_date}")
else:
    start_date = end_date = None



# -----------------------------
# HEADER
# -----------------------------
st.title("📊 DESMO Power & Mining Dashboard")
st.markdown(
    """
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

    if kraken_trades_df is not None:
        kraken_filtered = kraken_trades_df[
            (kraken_trades_df["date"] >= start_date)
            & (kraken_trades_df["date"] <= end_date)
        ]
    else:
        kraken_filtered = None
else:
    synota_filtered = synota_df
    antpool_filtered = antpool_df
    hosting_filtered = hosting_df_all
    kraken_filtered = kraken_trades_df



# -----------------------------
# TABS PRINCIPALI
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "📌 Overview & Trends",
        "📊 Charts",
        "⚡ Energia (Synota)",
        "⛏️ Mining (Antpool)",
        "🤝 Hosting",
        "🧠 Smart Analytics",
        "📈 Metriche globali",
        "🔗 Combined View",
        "💰 Vendite BTC",
    ]
)

# -----------------------------
# TAB 3 – ENERGIA / SYNOTA
# -----------------------------
with tab3:
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

            fig_rel.add_bar(
                x=synota_rel["date"],
                y=synota_rel["energy_index"],
                name="Energia (indice, 100 = media)",
            )

            fig_rel.add_bar(
                x=synota_rel["date"],
                y=synota_rel["cost_index"],
                name="Costo energia (indice, 100 = media)",
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
# TAB 4 – MINING / ANTPOOL
# -----------------------------
with tab4:
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
# TAB 7 – COMBINED VIEW
# -----------------------------
with tab7:
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

            # Ricavi DESMO da mining (self-mining)
            combined["earnings_usd"] = combined["total_earnings_btc"] * combined["btc_price_usd"]

            # -----------------------------
            # Ricavi DESMO da hosting (per giorno)
            # -----------------------------
            if hosting_filtered is not None and not hosting_filtered.empty:
                ASIC_TH = 200.0
                ASIC_KW = 3.5
                HOURS_PER_DAY = 24.0

                hf = hosting_filtered.copy()
                hf["asic_equivalent"] = hf["daily_hashrate_ths"] / ASIC_TH
                hf["energy_kwh"] = hf["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
                hf["hosting_revenue_usd"] = hf["energy_kwh"] * hosting_price_per_kwh

                hosting_daily = hf.groupby("date", as_index=False)["hosting_revenue_usd"].sum()
                combined = combined.merge(hosting_daily, on="date", how="left")
                combined["hosting_revenue_usd"] = combined["hosting_revenue_usd"].fillna(0.0)
            else:
                combined["hosting_revenue_usd"] = 0.0

            # Profitto / perdita giornaliero DESMO:
            # mining USD + hosting USD - costo energia
            combined["daily_profit_usd"] = (
                combined["earnings_usd"] + combined["hosting_revenue_usd"] - combined["invoice_amount_usd"]
            )

            # Metriche base (sul periodo)
            total_mwh_c = combined["energy_mwh"].sum()
            total_cost_c = combined["invoice_amount_usd"].sum()
            total_btc_c = combined["total_earnings_btc"].sum()

            total_rev_mining_usd_c = combined["earnings_usd"].sum()
            total_hosting_rev_c = combined["hosting_revenue_usd"].sum()
            total_rev_usd_c = total_rev_mining_usd_c + total_hosting_rev_c

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Energia totale [MWh]", f"{total_mwh_c:,.2f}")
            with col2:
                st.metric("Costo totale energia [USD]", f"{total_cost_c:,.2f}")
            with col3:
                st.metric("BTC totali minati", f"{total_btc_c:.6f}")
            with col4:
                st.metric("Ricavi totali [USD]", f"{total_rev_usd_c:,.2f}")

            st.caption(f"Di cui ricavi hosting: {total_hosting_rev_c:,.2f} USD")

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
                "hosting_revenue_usd",
                "daily_profit_usd",
            ]
            st.dataframe(
                combined[show_cols].rename(
                    columns={
                        "date": "Date",
                        "energy_mwh": "Energy [MWh]",
                        "invoice_amount_usd": "Energy cost [USD]",
                        "effective_rate_usd_per_mwh": "Rate [USD/MWh]",
                        "daily_hashrate_ths": "Hashrate [TH/s]",
                        "total_earnings_btc": "Earnings [BTC]",
                        "btc_price_usd": "BTC price [USD]",
                        "earnings_usd": "Mining revenue [USD]",
                        "hosting_revenue_usd": "Hosting revenue [USD]",
                        "daily_profit_usd": "Daily profit [USD] (mining + hosting - energy)",
                    }
                ),
                use_container_width=True,
            )

            # Grafico profitto / perdita giornaliero
            st.markdown("### 📈 Profitto / perdita giornaliero (mining + hosting)")
            fig_profit = go.Figure()
            fig_profit.add_bar(
                x=combined["date"],
                y=combined["daily_profit_usd"],
                name="Profitto giornaliero [USD]",
            )
            fig_profit.update_layout(
                xaxis_title="Data",
                yaxis_title="Profitto / Perdita [USD]",
                hovermode="x unified",
            )
            st.plotly_chart(fig_profit, use_container_width=True)

            # Profitto medio giornaliero sul periodo
            avg_daily_profit = combined["daily_profit_usd"].mean()
            st.markdown("### 💰 Profitto medio giornaliero")
            st.metric("Profitto medio giornaliero [USD]", f"{avg_daily_profit:,.2f}")



# -----------------------------
# TAB 8 – METRICHE GLOBALI
# -----------------------------
with tab8:
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

        # Calcoli base per giorno/cliente (potenza e consumo)
        df_host["asic_equivalent"] = df_host["daily_hashrate_ths"] / ASIC_TH
        df_host["energy_kwh"] = df_host["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
        df_host["hosting_revenue_usd"] = df_host["energy_kwh"] * hosting_price_per_kwh

        # Prezzo BTC per i clienti hosting (stessa logica globale: CSV + fallback)
        if btc_price_df is not None and not btc_price_df.empty:
            df_host = df_host.merge(btc_price_df, on="date", how="left")
            fallback_price = btc_price_used if btc_price_used else 0.0
            df_host["btc_price_usd"] = df_host["btc_price_usd"].fillna(fallback_price)
        else:
            fallback_price = btc_price_used if btc_price_used else 0.0
            df_host["btc_price_usd"] = fallback_price

        # Ricavi del cliente da mining e profitto (ricavi mining - costi elettrici)
        df_host["client_mining_revenue_usd"] = df_host["total_earnings_btc"] * df_host["btc_price_usd"]
        df_host["client_profit_usd"] = df_host["client_mining_revenue_usd"] - df_host["hosting_revenue_usd"]

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
                "client_mining_revenue_usd": "sum",
                "client_profit_usd": "sum",
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
                    "client_mining_revenue_usd",
                    "client_profit_usd",
                    "total_earnings_btc",
                ]
            ].rename(
                columns={
                    "client": "Client",
                    "daily_hashrate_ths": "Avg Hashrate [TH/s]",
                    "asic_equivalent": "Avg ASIC S21 eq.",
                    "energy_mwh": "Energy billed [MWh]",
                    "hosting_revenue_usd": "Hosting revenue [USD] (DESMO)",
                    "client_mining_revenue_usd": "Client mining revenue [USD]",
                    "client_profit_usd": "Client profit from mining [USD]",
                    "total_earnings_btc": "BTC mined (period)",
                }
            ),
            use_container_width=True,
        )

# -----------------------------
# TAB 6 – SMART ANALYTICS
# -----------------------------
with tab6:
    st.header("🧠 Smart analytics – scenari & break-even")

    if synota_filtered is None or synota_filtered.empty or antpool_filtered is None or antpool_filtered.empty:
        st.info("Servono almeno i CSV Synota + Antpool per le analisi avanzate.")
    else:
        # Base: merge Synota + Antpool sul range filtrato
        combined_sa = pd.merge(
            synota_filtered,
            antpool_filtered,
            on="date",
            how="inner",
            suffixes=("_synota", "_antpool"),
        )

        if combined_sa.empty:
            st.warning("Nessun giorno in comune tra Synota e Antpool nel range selezionato.")
        else:
            # Prezzo BTC giornaliero (CSV + fallback a prezzo usato in app)
            if btc_price_df is not None and not btc_price_df.empty:
                combined_sa = combined_sa.merge(btc_price_df, on="date", how="left")
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_sa["btc_price_usd"] = combined_sa["btc_price_usd"].fillna(fallback_price)
            else:
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_sa["btc_price_usd"] = fallback_price

            # Ricavi DESMO da mining (self-mining)
            combined_sa["earnings_usd"] = combined_sa["total_earnings_btc"] * combined_sa["btc_price_usd"]

            # -----------------------------
            # Ricavi & consumi hosting per giorno (per togliere l'energia clienti)
            # -----------------------------
            if hosting_filtered is not None and not hosting_filtered.empty:
                ASIC_TH = 200.0
                ASIC_KW = 3.5
                HOURS_PER_DAY = 24.0

                hf = hosting_filtered.copy()
                hf["asic_equivalent"] = hf["daily_hashrate_ths"] / ASIC_TH
                hf["energy_kwh"] = hf["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
                hf["hosting_revenue_usd"] = hf["energy_kwh"] * hosting_price_per_kwh

                hosting_daily = hf.groupby("date", as_index=False).agg(
                    {
                        "energy_kwh": "sum",
                        "hosting_revenue_usd": "sum",
                    }
                )
                hosting_daily.rename(columns={"energy_kwh": "hosting_energy_kwh"}, inplace=True)

                combined_sa = combined_sa.merge(hosting_daily, on="date", how="left")
                combined_sa["hosting_energy_kwh"] = combined_sa["hosting_energy_kwh"].fillna(0.0)
                combined_sa["hosting_revenue_usd"] = combined_sa["hosting_revenue_usd"].fillna(0.0)
            else:
                combined_sa["hosting_energy_kwh"] = 0.0
                combined_sa["hosting_revenue_usd"] = 0.0

            # Energia DESMO (self-mining) = totale Synota - hosting
            combined_sa["energy_total_kwh"] = combined_sa["energy_mwh"] * 1000.0
            combined_sa["energy_desmo_kwh"] = combined_sa["energy_total_kwh"] - combined_sa["hosting_energy_kwh"]
            combined_sa["energy_desmo_kwh"] = combined_sa["energy_desmo_kwh"].clip(lower=0.0)

            # Profitto giornaliero DESMO = mining USD + hosting USD - costi energia
            combined_sa["daily_profit_usd"] = (
                combined_sa["earnings_usd"] + combined_sa["hosting_revenue_usd"] - combined_sa["invoice_amount_usd"]
            )

            # -----------------------------
            # Snapshot globale
            # -----------------------------
            total_mwh = combined_sa["energy_mwh"].sum()
            total_cost = combined_sa["invoice_amount_usd"].sum()
            total_btc = combined_sa["total_earnings_btc"].sum()
            total_rev_mining = combined_sa["earnings_usd"].sum()
            total_rev_hosting = combined_sa["hosting_revenue_usd"].sum()
            total_rev = total_rev_mining + total_rev_hosting
            total_profit = total_rev - total_cost

            avg_rate = total_cost / total_mwh if total_mwh > 0 else None
            breakeven_rate = total_rev / total_mwh if total_mwh > 0 else None
            avg_daily_profit = combined_sa["daily_profit_usd"].mean()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profitto medio giornaliero [USD]", f"{avg_daily_profit:,.2f}")
            with col2:
                st.metric("Ricavi totali [USD] (mining + hosting)", f"{total_rev:,.2f}")
            with col3:
                st.metric("Costo medio attuale [USD/MWh]", f"{avg_rate:,.2f}" if avg_rate else "N/A")
            with col4:
                st.metric(
                    "Costo break-even [USD/MWh]",
                    f"{breakeven_rate:,.2f}" if breakeven_rate else "N/A",
                )

            if avg_rate and breakeven_rate:
                delta_rate = avg_rate - breakeven_rate
                st.write(
                    f"👉 Per andare a break-even con i dati di questo periodo, il costo energia dovrebbe essere circa "
                    f"**{breakeven_rate:,.2f} USD/MWh**, rispetto agli attuali **{avg_rate:,.2f} USD/MWh** "
                    f"({delta_rate:,.2f} USD/MWh di differenza)."
                )

            st.markdown("---")

            # -----------------------------
            # Uptime stimato (serve hashrate teorico)
            # -----------------------------
            st.markdown("### ⏱ Uptime DESMO (basato su hashrate)")

            avg_actual_hashrate = combined_sa["daily_hashrate_ths"].mean()
            suggested_installed = float(combined_sa["daily_hashrate_ths"].max()) if not combined_sa.empty else 0.0

            expected_hashrate = st.number_input(
                "Hashrate teorico installato DESMO [TH/s]",
                min_value=0.0,
                value=suggested_installed if suggested_installed > 0 else 1000.0,
                step=100.0,
            )

            if expected_hashrate > 0:
                uptime_ratio = avg_actual_hashrate / expected_hashrate
                uptime_pct = max(0.0, min(1.0, uptime_ratio)) * 100.0
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    st.metric("Hashrate medio effettivo [TH/s]", f"{avg_actual_hashrate:,.2f}")
                with col_u2:
                    st.metric("Uptime stimato", f"{uptime_pct:,.1f}%")
            else:
                st.info("Imposta un hashrate teorico > 0 per calcolare l'uptime.")

            # -----------------------------
            # Uptime clienti hosting (Gilga & Soci)
            # -----------------------------
            if hosting_filtered is not None and not hosting_filtered.empty:
                st.markdown("### 👥 Uptime clienti hosting (S21)")

                df_host_sa = hosting_filtered.copy()

                client_configs = {
                    "Gilga": {"asic_count": 20, "th_per_asic": 200.0},
                    "Soci": {"asic_count": 15, "th_per_asic": 200.0},
                }

                cols_clients = st.columns(len(client_configs))
                for i, (key, cfg) in enumerate(client_configs.items()):
                    asic_count = cfg["asic_count"]
                    th_per_asic = cfg["th_per_asic"]
                    theoretical_hashrate = asic_count * th_per_asic  # TH/s

                    # match cliente per substring nel nome file (Antpool_Gilga_CSV ecc.)
                    mask_client = df_host_sa["client"].str.contains(key, case=False, na=False)
                    df_c = df_host_sa[mask_client]

                    if not df_c.empty and theoretical_hashrate > 0:
                        avg_hashrate_c = df_c["daily_hashrate_ths"].mean()
                        uptime_ratio_c = avg_hashrate_c / theoretical_hashrate
                        uptime_pct_c = max(0.0, min(1.0, uptime_ratio_c)) * 100.0

                        with cols_clients[i]:
                            st.metric(
                                f"{key} uptime",
                                f"{uptime_pct_c:,.1f}%",
                                help=f"{asic_count}x S21 @ {th_per_asic:.0f} TH/s (theor: {theoretical_hashrate:.0f} TH/s)",
                            )
                    else:
                        with cols_clients[i]:
                            st.metric(f"{key} uptime", "N/A")
            else:
                st.markdown("### 👥 Uptime clienti hosting")
                st.info("Nessun CSV hosting disponibile per calcolare l'uptime clienti.")

            st.markdown("---")


            # -----------------------------
            # Efficienza elettrica kWh/TH (energia specifica)
            # -----------------------------
            st.markdown("### ⚡ Efficienza elettrica kWh/TH (energia specifica)")

            # Efficienza effettiva DESMO (solo energia self-mining)
            mask_eff = (combined_sa["daily_hashrate_ths"] > 0) & (combined_sa["energy_desmo_kwh"] > 0)
            combined_sa.loc[mask_eff, "kwh_per_th_desmo"] = (
                combined_sa.loc[mask_eff, "energy_desmo_kwh"] / combined_sa.loc[mask_eff, "daily_hashrate_ths"]
            )

            avg_kwh_per_th_desmo = combined_sa["kwh_per_th_desmo"].mean()

            # Teorico S19 Pro 110 TH @ ~3.25 kW (approssimazione)
            S19_TH = 110.0
            S19_KW = 3.25
            S19_HOURS = 24.0
            s19_daily_kwh = S19_KW * S19_HOURS
            s19_kwh_per_th = s19_daily_kwh / S19_TH  # kWh/TH

            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric("Energia specifica DESMO [kWh/TH]", f"{avg_kwh_per_th_desmo:,.3f}")
            with col_e2:
                st.metric("S19 Pro teorico [kWh/TH]", f"{s19_kwh_per_th:,.3f}")
            with col_e3:
                if avg_kwh_per_th_desmo and s19_kwh_per_th:
                    ratio = avg_kwh_per_th_desmo / s19_kwh_per_th
                    st.metric("Consumo vs S19 teorico", f"{ratio*100:,.1f}%")
                else:
                    st.metric("Consumo vs S19 teorico", "N/A")

            st.caption(
                "Valori più bassi = più efficienti. Consumo calcolato con energia Synota al netto dei consumi hosting."
            )

            # Mini grafico kWh/TH nel tempo
            if "kwh_per_th_desmo" in combined_sa.columns and combined_sa["kwh_per_th_desmo"].notna().any():
                fig_eff = go.Figure()
                fig_eff.add_scatter(
                    x=combined_sa["date"],
                    y=combined_sa["kwh_per_th_desmo"],
                    name="kWh/TH effettivi (DESMO)",
                    mode="lines+markers",
                )
                fig_eff.update_layout(
                    xaxis_title="Data",
                    yaxis_title="kWh/TH (DESMO)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_eff, use_container_width=True)


            st.markdown("---")

            # -----------------------------
            # Simulatore costo energia USD/MWh
            # -----------------------------
            st.markdown("### 🔧 Simulatore costo energia (USD/MWh)")

            if total_mwh > 0:
                current_rate = avg_rate if avg_rate else 0.0
                sim_rate = st.slider(
                    "Simula un nuovo costo medio energia [USD/MWh]",
                    min_value=0.0,
                    max_value=max(200.0, current_rate * 2 if current_rate else 100.0),
                    value=float(current_rate) if current_rate else 50.0,
                    step=1.0,
                )

                simulated_cost_total = total_mwh * sim_rate
                simulated_profit_total = total_rev - simulated_cost_total

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Profitto totale attuale [USD]", f"{total_profit:,.2f}")
                with col_s2:
                    st.metric("Profitto totale simulato [USD]", f"{simulated_profit_total:,.2f}")
                with col_s3:
                    delta_profit = simulated_profit_total - total_profit
                    st.metric("Delta profitto simulato [USD]", f"{delta_profit:,.2f}")

                # Mini grafico confronto profitto
                fig_sim = go.Figure()
                fig_sim.add_bar(
                    x=["Attuale", "Simulato"],
                    y=[total_profit, simulated_profit_total],
                    name="Profitto totale [USD]",
                )
                fig_sim.update_layout(
                    yaxis_title="Profitto [USD]",
                    hovermode="x",
                )
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.info("Energia totale = 0, impossibile simulare un costo medio USD/MWh.")



# --------------------------------------------------------------------
# TAB 9 – VENDITE BTC / KRAKEN  (AGGIORNATA COME RICHIESTO)
# --------------------------------------------------------------------
with tab9:
    st.header("💰 Vendite BTC – Kraken & wallet DESMO")

    if kraken_filtered is None or kraken_filtered.empty:
        st.info("Nessun CSV vendite BTC (Kraken) caricato o nessun trade nel range selezionato.")
    else:
        # ----- METRICHE GLOBALI (tutte le vendite) – usate più sotto in expander -----
        total_btc_sold = kraken_filtered["btc_sold"].sum()
        total_usd_proceeds = kraken_filtered["usd_proceeds"].sum()
        avg_sell_price = total_usd_proceeds / total_btc_sold if total_btc_sold > 0 else None

        # BTC minati da DESMO (solo Antpool principale, non hosting)
        if antpool_filtered is not None and not antpool_filtered.empty:
            btc_mined_desmo = antpool_filtered["total_earnings_btc"].sum()
        else:
            btc_mined_desmo = 0.0

        # ----- 1) Selettore transazioni rilevanti (con auto-match DESMO → Kraken) -----
        st.markdown("### 🎯 Selettore transazioni rilevanti")

        # Predefinito: tutte incluse
        kraken_relevant = kraken_filtered.copy()
        kraken_relevant["include"] = True
        kraken_relevant["auto_matched"] = False

        # Recupera transazioni DESMO → KRAKEN (on-chain)
        outgoing_df = get_wallet_outgoing_txs(DESMO_WALLET_ADDRESS)
        outgoing_kraken = None
        if outgoing_df is not None and not outgoing_df.empty:
            outgoing_kraken = outgoing_df[outgoing_df["recipient"] == KRAKEN_DEPOSIT_ADDRESS]

        # Auto-match per importo (ogni importo usato 1 sola volta)
        if outgoing_kraken is not None and not outgoing_kraken.empty:
            used_indices = set()
            unmatched_rows = []

            for _, row in outgoing_kraken.iterrows():
                amt = float(row["amount_btc"])
                # match con tolleranza
                candidates = kraken_filtered[
                    (kraken_filtered["btc_sold"].sub(amt).abs() < 1e-8)
                ].copy()

                # scarta quelli già usati
                candidates = candidates[~candidates.index.isin(used_indices)]

                if not candidates.empty:
                    idx = candidates.index[0]
                    used_indices.add(idx)
                    kraken_relevant.loc[idx, "include"] = True
                    kraken_relevant.loc[idx, "auto_matched"] = True
                else:
                    unmatched_rows.append(row)

        with st.expander("Mostra / nascondi selettore transazioni rilevanti", expanded=True):
            st.caption(
                "Spunta le transazioni che vuoi includere nei calcoli di competenza.\n"
                "Quelle marcate come 'auto_matched' sono state abbinate a una transazione on-chain DESMO ➜ Kraken."
            )

            df_sel = kraken_relevant[
                ["time", "btc_sold", "price", "usd_proceeds", "fee", "txid", "auto_matched", "include"]
            ].sort_values("time")

            edited = st.data_editor(
                df_sel,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "include": st.column_config.CheckboxColumn(
                        "Includi",
                        help="Spunta per includere questa transazione nei calcoli",
                        default=True,
                    ),
                    "time": "Time",
                    "btc_sold": "BTC sold",
                    "price": "Price [USD/BTC]",
                    "usd_proceeds": "Proceeds [USD]",
                    "fee": "Fee [USD]",
                    "txid": "TxID",
                    "auto_matched": "Auto-matched",
                },
                disabled=["time", "btc_sold", "price", "usd_proceeds", "fee", "txid", "auto_matched"],
                key="kraken_editor",
            )

            # --- Calcolo totali AUTO-MATCHED (indipendenti da include) ---
            auto_mask = edited["auto_matched"] == True
            btc_sold_matched = float(edited.loc[auto_mask, "btc_sold"].sum())
            usd_sold_matched = float(edited.loc[auto_mask, "usd_proceeds"].sum())

            # Salvo in session_state per usarli nell'Overview
            st.session_state["btc_sold_matched"] = btc_sold_matched
            st.session_state["usd_sold_matched"] = usd_sold_matched

            # Filtra solo le transazioni spuntate per le metriche "selezionate"
            kraken_relevant = edited[edited["include"]].copy()


            kraken_relevant = edited[edited["include"]].copy()

            if kraken_relevant.empty:
                st.info("Nessuna transazione selezionata (nessuna checkbox spuntata).")
            else:
                st.caption(
                    f"Transazioni selezionate: {len(kraken_relevant)} | "
                    f"BTC totali selezionati: {kraken_relevant['btc_sold'].sum():.6f}"
                )

            # Tabella DESMO → Kraken (on-chain)
            st.markdown("---")
            with st.expander("📤 Mostra le transazioni DESMO ➜ Kraken (deposit address)"):
                st.caption(
                    f"Mittente: `{DESMO_WALLET_ADDRESS}` → Destinatario: `{KRAKEN_DEPOSIT_ADDRESS}`"
                )
                if outgoing_kraken is None or outgoing_kraken.empty:
                    st.info(
                        "Nessuna transazione in uscita trovata da DESMO verso l'indirizzo Kraken "
                        "o impossibile recuperare i dati (API Blockstream)."
                    )
                else:
                    st.dataframe(
                        outgoing_kraken[
                            ["time", "recipient", "amount_btc", "txid"]
                        ].rename(
                            columns={
                                "time": "Time",
                                "recipient": "Recipient address",
                                "amount_btc": "Amount [BTC]",
                                "txid": "TxID",
                            }
                        ),
                        use_container_width=True,
                    )
                    # Avviso per eventuali outgoing senza match perfetto
                    if outgoing_kraken is not None and not outgoing_kraken.empty:
                        # ricostruisco unmatched per messaggio
                        unmatched_mask = ~outgoing_kraken["amount_btc"].round(8).isin(
                            kraken_filtered["btc_sold"].round(8)
                        )
                        if unmatched_mask.any():
                            st.warning(
                                "Alcune transazioni DESMO ➜ Kraken non hanno un match esatto per importo nei trade Kraken. "
                                "Controlla manualmente i casi sospetti."
                            )

        # ----- 2) Metriche per le SOLE transazioni selezionate -----
        st.markdown("### 📊 Metriche per le vendite selezionate")

        if kraken_relevant is not None and not kraken_relevant.empty:
            sel_btc_sold = kraken_relevant["btc_sold"].sum()
            sel_usd_proceeds = kraken_relevant["usd_proceeds"].sum()
            sel_avg_price = sel_usd_proceeds / sel_btc_sold if sel_btc_sold > 0 else None

            # Memorizza il prezzo medio selezionato per la sidebar (modalità "Prezzo medio di vendita")
            if sel_avg_price and sel_avg_price > 0:
                st.session_state["selected_sell_avg_price"] = float(sel_avg_price)

            # Salviamo anche i quantitativi matched per usarli in Overview & Trends
            st.session_state["btc_sold_matched"] = float(sel_btc_sold)
            st.session_state["usd_sold_matched"] = float(sel_usd_proceeds)

            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                st.metric("BTC venduti (selezionati)", f"{sel_btc_sold:.6f}")
            with col_sel2:
                st.metric("USD incassati (selezionati)", f"{sel_usd_proceeds:,.2f}")
            with col_sel3:
                if sel_avg_price:
                    st.metric("Prezzo medio selezionato [USD/BTC]", f"{sel_avg_price:,.2f}")
                else:
                    st.metric("Prezzo medio selezionato [USD/BTC]", "N/A")
        else:
            sel_btc_sold = 0.0
            sel_usd_proceeds = 0.0
            sel_avg_price = None
            st.info("Nessuna transazione selezionata: impossibile calcolare le metriche di competenza.")

        # ----- 3) Collegamento BTC minati → BTC venduti + saldo wallet -----
        st.markdown("### 🔗 Collegamento BTC minati → BTC venduti")

        if btc_mined_desmo > 0:
            st.write(
                f"BTC minati da DESMO (Antpool, periodo selezionato): **{btc_mined_desmo:.6f} BTC**"
            )

            default_sold_from_mining = float(min(sel_btc_sold, btc_mined_desmo)) if sel_btc_sold > 0 else float(
                min(total_btc_sold, btc_mined_desmo)
            )

            btc_theoretical_balance = btc_mined_desmo - sel_btc_sold

            wallet_balance_btc = get_wallet_balance_btc(DESMO_WALLET_ADDRESS)

            colw1, colw2, colw3,colw4 = st.columns(4)
            with colw1:
                st.metric("BTC minati (Antpool)", f"{btc_mined_desmo:.6f}")
            with colw2:
                st.metric("BTC venduti attribuiti al mining", f"{sel_btc_sold:.6f}")
            if wallet_balance_btc is not None:
                with colw3:
                    st.metric("Saldo on-chain wallet [BTC]", f"{wallet_balance_btc:.6f}")
            else:
                with colw3:
                    st.metric("Saldo on-chain wallet [BTC]", "N/A")

            if wallet_balance_btc is not None:
                delta_btc = wallet_balance_btc - btc_theoretical_balance
                with colw4:
                    st.metric(
                        "Differenza (wallet - teorico) [BTC]",
                        f"{delta_btc:+.6f}",
                    )

                if abs(delta_btc) < 0.0001:
                    st.success("✅ Differenza trascurabile: mining, vendite selezionate e saldo wallet sono coerenti.")
                else:
                    st.warning(
                        "⚠️ C'è una differenza tra saldo on-chain e saldo teorico. "
                        "Potrebbero esserci depositi/prelievi extra o trade non coperti dai CSV."
                    )
            else:
                st.error("❌ Impossibile recuperare il saldo on-chain del wallet (API Blockstream non disponibile).")
        else:
            st.info("Nessun dato Antpool nel periodo selezionato: impossibile collegare BTC minati e venduti.")

        # ----- 4) Grafico: prezzo BTC + punti di vendita -----
        st.markdown("### 📉 Vendite BTC sul grafico del prezzo")

        fig_trades = go.Figure()

        if btc_price_df is not None and not btc_price_df.empty:
            fig_trades.add_scatter(
                x=btc_price_df["date"],
                y=btc_price_df["btc_price_usd"],
                mode="lines",
                name="BTC price (CSV)",
            )
        else:
            fig_trades.add_scatter(
                x=kraken_filtered["time"],
                y=kraken_filtered["price"],
                mode="lines",
                name="BTC price (from trades)",
            )

        fig_trades.add_scatter(
            x=kraken_filtered["time"],
            y=kraken_filtered["price"],
            mode="markers",
            name="Vendite BTC (Kraken)",
            marker=dict(size=8),
            text=[
                f"BTC venduti: {b:.6f}<br>USD: {u:,.2f}"
                for b, u in zip(kraken_filtered["btc_sold"], kraken_filtered["usd_proceeds"])
            ],
            hovertemplate="%{x}<br>Prezzo: %{y:,.2f} USD/BTC<br>%{text}<extra></extra>",
        )

        fig_trades.update_layout(
            xaxis_title="Data / Ora",
            yaxis_title="Prezzo BTC [USD]",
            hovermode="x unified",
        )

        st.plotly_chart(fig_trades, use_container_width=True)

        # ----- 5) Tabella riepilogo trade -----
        st.markdown("### 📄 Trade Kraken (filtrati)")

        st.dataframe(
            kraken_filtered[
                [
                    "time",
                    "btc_sold",
                    "price",
                    "usd_proceeds",
                    "fee",
                ]
            ].rename(
                columns={
                    "time": "Time",
                    "btc_sold": "BTC sold",
                    "price": "Price [USD/BTC]",
                    "usd_proceeds": "Proceeds [USD]",
                    "fee": "Fee [USD]",
                }
            ),
            use_container_width=True,
        )

        # ----- 6) Box a scomparsa con metriche GLOBALI totali Kraken -----
        with st.expander("💰 Vendite BTC totali Kraken (tutte le transazioni)", expanded=False):
            colg1, colg2, colg3 = st.columns(3)
            with colg1:
                st.metric("BTC totali venduti (Kraken)", f"{total_btc_sold:.6f}")
            with colg2:
                st.metric("USD totali incassati", f"{total_usd_proceeds:,.2f}")
            with colg3:
                if avg_sell_price:
                    st.metric("Prezzo medio di vendita [USD/BTC]", f"{avg_sell_price:,.2f}")
                else:
                    st.metric("Prezzo medio di vendita [USD/BTC]", "N/A")


# -----------------------------
# TAB 1 – OVERVIEW & TRENDS
# -----------------------------
with tab1:
    st.header("📌 Overview & Trends")

    # Serve almeno Synota + Antpool per avere un quadro completo
    if synota_filtered is None or synota_filtered.empty or antpool_filtered is None or antpool_filtered.empty:
        st.info("Servono almeno i CSV Synota + Antpool per vedere l'overview generale.")
    else:
        # =========================
        # 0) COSTRUZIONE DATASET GIORNALIERO COMBINATO
        # =========================
        combined_ov = pd.merge(
            synota_filtered,
            antpool_filtered,
            on="date",
            how="inner",
            suffixes=("_synota", "_antpool"),
        )

        if combined_ov.empty:
            st.warning("Nessun giorno in comune tra Synota e Antpool nel range selezionato.")
        else:
            # Prezzo BTC giornaliero (CSV storico + fallback a btc_price_used)
            if btc_price_df is not None and not btc_price_df.empty:
                combined_ov = combined_ov.merge(btc_price_df, on="date", how="left")
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_ov["btc_price_usd"] = combined_ov["btc_price_usd"].fillna(fallback_price)
            else:
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_ov["btc_price_usd"] = fallback_price

            # Ricavi DESMO da mining (self-mining Antpool principale)
            combined_ov["earnings_usd"] = combined_ov["total_earnings_btc"] * combined_ov["btc_price_usd"]

            # =========================
            # HOSTING: energia & ricavi per giorno
            # =========================
            if hosting_filtered is not None and not hosting_filtered.empty:
                ASIC_TH = 200.0
                ASIC_KW = 3.5
                HOURS_PER_DAY = 24.0

                hf_ov = hosting_filtered.copy()
                hf_ov["asic_equivalent"] = hf_ov["daily_hashrate_ths"] / ASIC_TH
                hf_ov["energy_kwh"] = hf_ov["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
                hf_ov["hosting_revenue_usd"] = hf_ov["energy_kwh"] * hosting_price_per_kwh

                hosting_daily_ov = hf_ov.groupby("date", as_index=False).agg(
                    {
                        "energy_kwh": "sum",
                        "hosting_revenue_usd": "sum",
                        "total_earnings_btc": "sum",
                    }
                ).rename(
                    columns={
                        "energy_kwh": "hosting_energy_kwh",
                        "total_earnings_btc": "hosting_btc_mined",
                    }
                )

                combined_ov = combined_ov.merge(hosting_daily_ov, on="date", how="left")
                combined_ov["hosting_energy_kwh"] = combined_ov["hosting_energy_kwh"].fillna(0.0)
                combined_ov["hosting_revenue_usd"] = combined_ov["hosting_revenue_usd"].fillna(0.0)
                combined_ov["hosting_btc_mined"] = combined_ov["hosting_btc_mined"].fillna(0.0)
            else:
                combined_ov["hosting_energy_kwh"] = 0.0
                combined_ov["hosting_revenue_usd"] = 0.0
                combined_ov["hosting_btc_mined"] = 0.0

            # Energia DESMO = totale Synota - hosting
            combined_ov["energy_total_kwh"] = combined_ov["energy_mwh"] * 1000.0
            combined_ov["energy_desmo_kwh"] = combined_ov["energy_total_kwh"] - combined_ov["hosting_energy_kwh"]
            combined_ov["energy_desmo_kwh"] = combined_ov["energy_desmo_kwh"].clip(lower=0.0)

            # Profitto giornaliero DESMO (mining + hosting - costi energia)
            combined_ov["daily_profit_usd"] = (
                combined_ov["earnings_usd"] + combined_ov["hosting_revenue_usd"] - combined_ov["invoice_amount_usd"]
            )

            combined_ov = combined_ov.sort_values("date")

            # =========================
            # 1) PARAMETRO: PERIODO DI CONFRONTO (BOX A SCOMPARSA)
            # =========================
            st.markdown("#### 🔍 Periodo di confronto per le variazioni %")

            with st.expander("Imposta periodo di confronto (opzionale)", expanded=False):
                compare_mode = st.radio(
                    "Periodo per il confronto delle %:",
                    [
                        "Tutti i dati (nessun confronto)",
                        "Ultimi 7 giorni vs 7 precedenti",
                        "Ultimi 30 giorni vs 30 precedenti",
                    ],
                )

            last_date = combined_ov["date"].max()

            # df_main = periodo “corrente” su cui mostriamo i valori
            # df_compare = periodo precedente di confronto (stessa durata), se esiste
            df_main = combined_ov.copy()
            df_compare = None
            compare_label = None

            if compare_mode != "Tutti i dati (nessun confronto)":
                if "7" in compare_mode:
                    days = 7
                    compare_label = "settimana precedente"
                else:
                    days = 30
                    compare_label = "mese precedente"

                current_start = last_date - pd.Timedelta(days=days - 1)
                prev_start = last_date - pd.Timedelta(days=days * 2 - 1)
                prev_end = last_date - pd.Timedelta(days=days)

                df_main = combined_ov[(combined_ov["date"] >= current_start) & (combined_ov["date"] <= last_date)]
                df_compare = combined_ov[(combined_ov["date"] >= prev_start) & (combined_ov["date"] <= prev_end)]

                # Se il periodo corrente è vuoto (pochi dati), fallback a tutti i dati senza confronto
                if df_main.empty or df_main["date"].nunique() < 2:
                    df_main = combined_ov.copy()
                    df_compare = None
                    compare_label = None

            def compute_delta(base_value: float | None, current_value: float | None) -> str | None:
                if base_value is None or current_value is None:
                    return None
                if base_value == 0:
                    return None
                change = (current_value - base_value) / abs(base_value) * 100.0
                return f"{change:+.1f}%"

            # =========================
            # 3) MINING
            # =========================
            st.markdown("### ⛏️ Mining (DESMO)")

            avg_hashrate_main = df_main["daily_hashrate_ths"].mean()
            avg_hashrate_compare = (
                df_compare["daily_hashrate_ths"].mean()
                if df_compare is not None and not df_compare.empty
                else None
            )
            hash_delta = compute_delta(avg_hashrate_compare, avg_hashrate_main) if df_compare is not None else None

            installed_hashrate_main = df_main["daily_hashrate_ths"].max() if not df_main.empty else 0.0
            uptime_main = (
                avg_hashrate_main / installed_hashrate_main * 100.0 if installed_hashrate_main > 0 else None
            )

            avg_btc_day_main = df_main["total_earnings_btc"].mean()
            avg_btc_day_compare = (
                df_compare["total_earnings_btc"].mean()
                if df_compare is not None and not df_compare.empty
                else None
            )
            btc_day_delta = compute_delta(avg_btc_day_compare, avg_btc_day_main) if df_compare is not None else None

            avg_usd_day_main = df_main["earnings_usd"].mean()
            avg_usd_day_compare = (
                df_compare["earnings_usd"].mean()
                if df_compare is not None and not df_compare.empty
                else None
            )
            usd_day_delta = compute_delta(avg_usd_day_compare, avg_usd_day_main) if df_compare is not None else None

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(
                    "Hashrate medio giornaliero [TH/s]",
                    f"{avg_hashrate_main:,.2f}" if avg_hashrate_main is not None else "N/A",
                    delta=hash_delta if hash_delta is not None and compare_label else None,
                    help=(
                        "Hashrate medio giornaliero DESMO nel periodo corrente. "
                        f"Delta vs {compare_label} se impostato."
                        if compare_label
                        else "Hashrate medio giornaliero DESMO nel periodo corrente."
                    ),
                )
            with col_m2:
                st.metric(
                    "Uptime stimato DESMO",
                    f"{uptime_main:,.1f}%" if uptime_main is not None else "N/A",
                    help="Stimato come hashrate medio / hashrate massimo nel periodo corrente.",
                )
            with col_m3:
                st.metric(
                    "BTC minati al giorno [BTC]",
                    f"{avg_btc_day_main:.6f}" if avg_btc_day_main is not None else "N/A",
                    delta=btc_day_delta if btc_day_delta is not None and compare_label else None,
                    help=(
                        "Media giornaliera BTC minati da DESMO nel periodo corrente. "
                        f"Delta vs {compare_label} se impostato."
                        if compare_label
                        else "Media giornaliera BTC minati da DESMO nel periodo corrente."
                    ),
                )
            with col_m4:
                st.metric(
                    "Ricavi giornalieri da mining [USD]",
                    f"{avg_usd_day_main:,.2f}" if avg_usd_day_main is not None else "N/A",
                    delta=usd_day_delta if usd_day_delta is not None and compare_label else None,
                    help=(
                        "Ricavi medi giornalieri da mining (solo Antpool DESMO) nel periodo corrente. "
                        f"Delta vs {compare_label} se impostato."
                        if compare_label
                        else "Ricavi medi giornalieri da mining (solo Antpool DESMO) nel periodo corrente."
                    ),
                )

           

            st.markdown("---")

            # =========================
            # 4) ENERGIA
            # =========================
            st.markdown("### ⚡ Energia")

            # Costo medio giornaliero (prima metrica richiesta)
            avg_daily_energy_cost_main = df_main["invoice_amount_usd"].mean()
            avg_daily_energy_cost_compare = (
                df_compare["invoice_amount_usd"].mean()
                if df_compare is not None and not df_compare.empty
                else None
            )
            daily_cost_delta = (
                compute_delta(avg_daily_energy_cost_compare, avg_daily_energy_cost_main)
                if df_compare is not None
                else None
            )

            # Costo medio [USD/MWh]
            total_mwh_main = df_main["energy_mwh"].sum()
            total_cost_main = df_main["invoice_amount_usd"].sum()
            cost_per_mwh_main = total_cost_main / total_mwh_main if total_mwh_main > 0 else None

            def avg_cost_over_window(df: pd.DataFrame, days: int) -> float | None:
                if df.empty:
                    return None
                last_d = df["date"].max()
                start_d = last_d - pd.Timedelta(days=days - 1)
                df_sub = df[df["date"] >= start_d]
                sub_mwh = df_sub["energy_mwh"].sum()
                sub_cost = df_sub["invoice_amount_usd"].sum()
                return sub_cost / sub_mwh if sub_mwh > 0 else None

            cost_7d_main = avg_cost_over_window(df_main, 7)
            cost_30d_main = avg_cost_over_window(df_main, 30)

            # Per le % di 7/30 usiamo comunque il totale come base
            delta_7_vs_total = compute_delta(cost_per_mwh_main, cost_7d_main)
            delta_30_vs_total = compute_delta(cost_per_mwh_main, cost_30d_main)

            col_e0, col_e1 = st.columns(2)
            with col_e0:
                st.metric(
                    "Costo medio giornaliero energia [USD]",
                    f"{avg_daily_energy_cost_main:,.2f}" if avg_daily_energy_cost_main is not None else "N/A",
                    delta=daily_cost_delta if daily_cost_delta is not None and compare_label else None,
                    help=(
                        "Media dei costi Synota (Invoice Amount) per giorno nel periodo corrente. "
                        f"Delta vs {compare_label} se impostato."
                        if compare_label
                        else "Media dei costi Synota (Invoice Amount) per giorno nel periodo corrente."
                    ),
                )

            with col_e1:
                st.metric(
                    "Costo medio energia [USD/MWh] (totale)",
                    f"{cost_per_mwh_main:,.2f}" if cost_per_mwh_main is not None else "N/A",
                    help="Costo medio energia Synota in USD/MWh nel periodo corrente.",
                )


            st.markdown("---")

            # =========================
            # 5) HOSTING
            # =========================
            st.markdown("### 🤝 Hosting")

            if hosting_filtered is not None and not hosting_filtered.empty:
                avg_hosting_day_main = df_main["hosting_revenue_usd"].mean()
                avg_hosting_day_compare = (
                    df_compare["hosting_revenue_usd"].mean()
                    if df_compare is not None and not df_compare.empty
                    else None
                )
                hosting_day_delta = (
                    compute_delta(avg_hosting_day_compare, avg_hosting_day_main)
                    if df_compare is not None
                    else None
                )

                df_host_sa = hosting_filtered.copy()
                client_configs = {
                    "Gilga": {"asic_count": 20, "th_per_asic": 200.0},
                    "Soci": {"asic_count": 15, "th_per_asic": 200.0},
                }

                uptime_clients_text = []
                for key, cfg in client_configs.items():
                    asic_count = cfg["asic_count"]
                    th_per_asic = cfg["th_per_asic"]
                    theoretical_hashrate = asic_count * th_per_asic

                    mask_client = df_host_sa["client"].str.contains(key, case=False, na=False)
                    df_c = df_host_sa[mask_client]

                    if not df_c.empty and theoretical_hashrate > 0:
                        avg_hashrate_c = df_c["daily_hashrate_ths"].mean()
                        uptime_ratio_c = avg_hashrate_c / theoretical_hashrate
                        uptime_pct_c = max(0.0, min(1.0, uptime_ratio_c)) * 100.0
                        uptime_clients_text.append(f"{key}: {uptime_pct_c:,.1f}%")
                    else:
                        uptime_clients_text.append(f"{key}: N/A")

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.metric(
                        "Ricavi hosting medi giornalieri [USD]",
                        f"{avg_hosting_day_main:,.2f}" if avg_hosting_day_main is not None else "N/A",
                        delta=hosting_day_delta if hosting_day_delta is not None and compare_label else None,
                        help=(
                            "Media dei ricavi hosting (energia fatturata ai clienti) nel periodo corrente. "
                            f"Delta vs {compare_label} se impostato."
                            if compare_label
                            else "Media dei ricavi hosting (energia fatturata ai clienti) nel periodo corrente."
                        ),
                    )
                with col_h2:
                    st.write("**Uptime clienti**")
                    st.write(" • " + " | ".join(uptime_clients_text))
            else:
                st.info("Nessun dato hosting disponibile nel periodo selezionato.")

            st.markdown("---")

            # =========================
            # 6) METRICHE BTC (vendite & wallet)
            # =========================
            st.markdown("### ₿ Metriche BTC")

            # Valori matched salvati dalla tab Kraken
            btc_sold_total = float(st.session_state.get("btc_sold_matched", 0.0))
            usd_sold_total = float(st.session_state.get("usd_sold_matched", 0.0))

            if btc_sold_total > 0:
                avg_sell_price_total = usd_sold_total / btc_sold_total
            else:
                avg_sell_price_total = None

            # Saldo wallet DESMO on-chain
            wallet_balance_btc = get_wallet_balance_btc(DESMO_WALLET_ADDRESS)

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric(
                    "BTC venduti (Kraken, auto-matched)",
                    f"{btc_sold_total:.6f}",
                    help="Somma dei BTC venduti selezionati e auto-matched nella tab '💰 Vendite BTC'.",
                )
            with col_b2:
                st.metric(
                    "Prezzo medio di vendita [USD/BTC]",
                    f"{avg_sell_price_total:,.2f}" if avg_sell_price_total is not None else "N/A",
                    help="Prezzo medio ponderato dei BTC venduti (solo matched selezionati).",
                )
            with col_b3:
                st.metric(
                    "BTC su wallet DESMO",
                    f"{wallet_balance_btc:.6f}" if wallet_balance_btc is not None else "N/A",
                    help="Saldo on-chain corrente del wallet principale DESMO.",
                )

            st.markdown("---")

 # =========================
            # 2) IMPORTANT DATA
            # =========================
            st.markdown("### 🔥 Important data")

            total_mwh_main = df_main["energy_mwh"].sum()
            total_cost_main = df_main["invoice_amount_usd"].sum()
            total_btc_desmo_main = df_main["total_earnings_btc"].sum()
            total_rev_mining_main = df_main["earnings_usd"].sum()
            total_rev_hosting_main = df_main["hosting_revenue_usd"].sum()
            total_rev_all_main = total_rev_mining_main + total_rev_hosting_main
            total_profit_main = total_rev_all_main - total_cost_main

            # Profitto medio giornaliero
            avg_daily_profit_main = df_main["daily_profit_usd"].mean()
            avg_daily_profit_compare = (
                df_compare["daily_profit_usd"].mean() if df_compare is not None and not df_compare.empty else None
            )
            profit_delta = compute_delta(avg_daily_profit_compare, avg_daily_profit_main) if df_compare is not None else None

            # Costo per minare 1 BTC (DESMO)
            cost_per_btc_desmo_main = (
                total_cost_main / total_btc_desmo_main if total_btc_desmo_main > 0 else None
            )

            # Costo medio energia & break-even energia [USD/MWh]
            avg_rate_main = total_cost_main / total_mwh_main if total_mwh_main > 0 else None
            breakeven_rate_main = total_rev_all_main / total_mwh_main if total_mwh_main > 0 else None




            # =========================
            # Rapporto Watt/TH per break-even (DESMO)
            # =========================
            # 1) calcoliamo W/TH effettivo medio
            mask_eff = (df_main["daily_hashrate_ths"] > 0) & (df_main["energy_desmo_kwh"] > 0)
            df_eff = df_main.loc[mask_eff].copy()

            if not df_eff.empty:
                df_eff["kwh_per_th_per_day"] = df_eff["energy_desmo_kwh"] / df_eff["daily_hashrate_ths"]
                kwh_per_th_day_actual = df_eff["kwh_per_th_per_day"].mean()
                w_per_th_actual = kwh_per_th_day_actual * 1000.0 / 24.0  # kWh/day → W per TH
            else:
                w_per_th_actual = None

            # 2) stimiamo il costo energia imputabile a DESMO
            energy_desmo_kwh_total_main = df_main["energy_desmo_kwh"].sum()
            total_kwh_main = df_main["energy_mwh"].sum() * 1000.0
            if total_kwh_main > 0:
                avg_price_per_kwh_main = total_cost_main / total_kwh_main
            else:
                avg_price_per_kwh_main = None

            if avg_price_per_kwh_main is not None:
                cost_desmo_energy_main = energy_desmo_kwh_total_main * avg_price_per_kwh_main
            else:
                cost_desmo_energy_main = None

            # Prezzo BTC di break-even per DESMO (solo mining DESMO, solo costi elettrici DESMO)
            # Definizione: BTC_price * BTC_DESMO = costo_energia_DESMO
            # → BTC_price = cost_desmo_energy_main / total_btc_desmo_main
            if total_btc_desmo_main > 0 and cost_desmo_energy_main is not None:
                breakeven_btc_price_main = cost_desmo_energy_main / total_btc_desmo_main
            else:
                breakeven_btc_price_main = None

            # 3) break-even W/TH: riducendo i W/TH, il costo energia scala linearmente.
            # vogliamo: revenue_mining_main = costo_desmo_energy_scaled
            # ⇒ fattore f = revenue_mining_main / costo_desmo_energy_main
            # ⇒ W/TH_break_even = W/TH_actual * f
            if (
                w_per_th_actual is not None
                and cost_desmo_energy_main is not None
                and cost_desmo_energy_main > 0
            ):
                factor = total_rev_mining_main / cost_desmo_energy_main
                w_per_th_breakeven = w_per_th_actual * factor
            else:
                w_per_th_breakeven = None

            # ---------- layout Important data ----------
            col_imp1, col_imp2, col_imp3 = st.columns(3)
            with col_imp1:
                st.metric(
                    "Profitto medio giornaliero [USD]",
                    f"{avg_daily_profit_main:,.2f}" if avg_daily_profit_main is not None else "N/A",
                    delta=profit_delta if profit_delta is not None and compare_label else None,
                    help=(
                        "Media giornaliera di (ricavi mining + ricavi hosting - costi energia) "
                        f"nel periodo corrente. Delta vs {compare_label} se impostato."
                        if compare_label
                        else "Media giornaliera di (ricavi mining + ricavi hosting - costi energia) nel periodo corrente."
                    ),
                )

            with col_imp2:
                st.metric(
                    "Costo per minare 1 BTC [USD/BTC]",
                    f"{cost_per_btc_desmo_main:,.2f}" if cost_per_btc_desmo_main is not None else "N/A",
                    help="Costo totale energia nel periodo corrente diviso BTC minati da DESMO (Antpool principale).",
                )

            with col_imp3:
                st.metric(
                    "Profitto / perdita totale [USD]",
                    f"{total_profit_main:,.2f}",
                    help="Somma su tutto il periodo corrente di (ricavi mining + ricavi hosting - costi energia).",
                )

            col_imp4, col_imp5, col_imp6 = st.columns(3)
            with col_imp4:
                st.metric(
                    "Costo break-even energia [USD/MWh]",
                    f"{breakeven_rate_main:,.2f}" if breakeven_rate_main is not None else "N/A",
                    help="Costo medio USD/MWh a cui il profitto complessivo (mining + hosting) sarebbe ≈ 0 nel periodo corrente.",
                )

            with col_imp5:
                st.metric(
                    "Prezzo BTC di break-even [USD/BTC]",
                    f"{breakeven_btc_price_main:,.2f}" if breakeven_btc_price_main is not None else "N/A",
                    help=(
                            "Prezzo BTC per cui i soli ricavi del mining DESMO coprono il costo elettrico "
                            "imputato a DESMO (Synota - consumo stimato clienti), "
                            "risolvendo: BTC_price * BTC_DESMO = costo_energia_DESMO."
                        ),
                )

            with col_imp6:
                st.metric(
                    "Rapporto Watt/TH per break-even",
                    f"{w_per_th_breakeven:,.1f}" if w_per_th_breakeven is not None else "N/A",
                    help=(
                        "Efficienza elettrica (W per TH) che renderebbe i soli ricavi mining DESMO ≈ costi energia attribuiti a DESMO. "
                        "Calcolata scalando l'efficienza attuale in base al rapporto ricavi_mining / costi_energia_DESMO."
                    ),
                )

            st.markdown("---")

            # =========================
            # 7) MASSIVE DATA – FOTO DI INSIEME
            # =========================
            st.markdown("### 🧱 Massive data – Foto di insieme")

            # BTC DESMO (Antpool principale)
            btc_desmo_total = antpool_filtered["total_earnings_btc"].sum() if antpool_filtered is not None else 0.0

            # BTC CLIENTI (tutti i CSV hosting)
            if hosting_filtered is not None and not hosting_filtered.empty:
                btc_clients_total = hosting_filtered["total_earnings_btc"].sum()
            else:
                btc_clients_total = 0.0

            btc_all_total = btc_desmo_total + btc_clients_total

            # Valore in USD (usa daily price se disponibile, altrimenti btc_price_used)
            # DESMO
            if antpool_filtered is not None and not antpool_filtered.empty:
                df_tmp_desmo = antpool_filtered.copy()
                if btc_price_df is not None and not btc_price_df.empty:
                    df_tmp_desmo = df_tmp_desmo.merge(btc_price_df, on="date", how="left")
                    df_tmp_desmo["btc_price_usd"] = df_tmp_desmo["btc_price_usd"].fillna(
                        btc_price_used if btc_price_used else 0.0
                    )
                else:
                    df_tmp_desmo["btc_price_usd"] = btc_price_used if btc_price_used else 0.0
                df_tmp_desmo["usd_value"] = df_tmp_desmo["total_earnings_btc"] * df_tmp_desmo["btc_price_usd"]
                usd_desmo_total = df_tmp_desmo["usd_value"].sum()
            else:
                usd_desmo_total = 0.0

            # CLIENTI
            if hosting_filtered is not None and not hosting_filtered.empty:
                df_tmp_cli = hosting_filtered.copy()
                if btc_price_df is not None and not btc_price_df.empty:
                    df_tmp_cli = df_tmp_cli.merge(btc_price_df, on="date", how="left")
                    df_tmp_cli["btc_price_usd"] = df_tmp_cli["btc_price_usd"].fillna(
                        btc_price_used if btc_price_used else 0.0
                    )
                else:
                    df_tmp_cli["btc_price_usd"] = btc_price_used if btc_price_used else 0.0
                df_tmp_cli["usd_value"] = df_tmp_cli["total_earnings_btc"] * df_tmp_cli["btc_price_usd"]
                usd_clients_total = df_tmp_cli["usd_value"].sum()
            else:
                usd_clients_total = 0.0

            usd_all_total = usd_desmo_total + usd_clients_total

            # Energia & costi totali (Synota = DESMO + hosting)
            total_mwh_all = synota_filtered["energy_mwh"].sum() if synota_filtered is not None else 0.0
            total_cost_all = synota_filtered["invoice_amount_usd"].sum() if synota_filtered is not None else 0.0

            # Energia DESMO vs totale dal dataset combinato
            total_energy_desmo_kwh_all = combined_ov["energy_desmo_kwh"].sum()
            total_energy_kwh_all = combined_ov["energy_total_kwh"].sum()

            if total_energy_kwh_all > 0:
                # ripartizione del costo in base alla quota di kWh usati da DESMO
                cost_desmo_only = total_cost_all * (total_energy_desmo_kwh_all / total_energy_kwh_all)
            else:
                cost_desmo_only = 0.0

            # Ricavi hosting totali (DESMO)
            total_hosting_revenue_all = combined_ov["hosting_revenue_usd"].sum()

            # Profitto / perdita DESMO complessivo = mining DESMO - costi DESMO + hosting
            desmo_global_profit = (usd_desmo_total - cost_desmo_only) + total_hosting_revenue_all

            # --- layout ---
            # Riga 1: BTC totali + valore totale
            col_mass1, col_mass2, col_mass3 = st.columns(3)
            with col_mass1:
                st.metric(
                    "BTC minati in totale (DESMO + clienti) [BTC]",
                    f"{btc_all_total:.6f}",
                )
            with col_mass2:
                st.metric(
                    "Valore BTC minati (stima) [USD]",
                    f"{usd_all_total:,.2f}",
                )
            with col_mass3:
                st.metric(
                    "Energia consumata totale [MWh] / costo [USD]",
                       f"{total_mwh_all:,.2f} MWh | {total_cost_all:,.2f} USD",
                )

            # Riga 2: energia totale (MWh + USD), BTC DESMO (BTC+USD), costo elettrico DESMO
            col_mass3, col_mass4, col_mass6  = st.columns(3)
            with col_mass3:
                st.metric(
                    "BTC DESMO minati [BTC / USD]",
                    f"{btc_desmo_total:.6f} BTC | {usd_desmo_total:,.2f} USD",
                )
            with col_mass4:
                st.metric(
                    "Costo elettrico DESMO [USD]",
                    f"{cost_desmo_only:,.2f}",
                )
            with col_mass6:
                st.metric(
                    "Ricavi da hosting [USD]",
                    f"{total_hosting_revenue_all:,.2f}",
                )

# -----------------------------
# TAB 2 – CHARTS OVERVIEW
# -----------------------------
with tab2:
    st.header("📊 Grafici overview & trends")

    # Serve almeno Synota + Antpool
    if synota_filtered is None or synota_filtered.empty or antpool_filtered is None or antpool_filtered.empty:
        st.info("Servono almeno i CSV Synota + Antpool per vedere i grafici di overview.")
    else:
        # =========================
        # 0) COSTRUZIONE DATASET GIORNALIERO COMBINATO (come in Overview & Trends)
        # =========================
        combined_ch = pd.merge(
            synota_filtered,
            antpool_filtered,
            on="date",
            how="inner",
            suffixes=("_synota", "_antpool"),
        )

        if combined_ch.empty:
            st.warning("Nessun giorno in comune tra Synota e Antpool nel range selezionato.")
        else:
            # Prezzo BTC giornaliero (CSV storico + fallback a btc_price_used)
            if btc_price_df is not None and not btc_price_df.empty:
                combined_ch = combined_ch.merge(btc_price_df, on="date", how="left")
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_ch["btc_price_usd"] = combined_ch["btc_price_usd"].fillna(fallback_price)
            else:
                fallback_price = btc_price_used if btc_price_used else 0.0
                combined_ch["btc_price_usd"] = fallback_price

            # Ricavi DESMO da mining (self-mining Antpool principale)
            combined_ch["earnings_usd"] = combined_ch["total_earnings_btc"] * combined_ch["btc_price_usd"]

            # =========================
            # HOSTING: energia & ricavi per giorno
            # =========================
            if hosting_filtered is not None and not hosting_filtered.empty:
                ASIC_TH = 200.0
                ASIC_KW = 3.5
                HOURS_PER_DAY = 24.0

                hf_ch = hosting_filtered.copy()
                hf_ch["asic_equivalent"] = hf_ch["daily_hashrate_ths"] / ASIC_TH
                hf_ch["energy_kwh"] = hf_ch["asic_equivalent"] * ASIC_KW * HOURS_PER_DAY
                hf_ch["hosting_revenue_usd"] = hf_ch["energy_kwh"] * hosting_price_per_kwh

                hosting_daily_ch = hf_ch.groupby("date", as_index=False).agg(
                    {
                        "energy_kwh": "sum",
                        "hosting_revenue_usd": "sum",
                        "total_earnings_btc": "sum",
                    }
                ).rename(
                    columns={
                        "energy_kwh": "hosting_energy_kwh",
                        "total_earnings_btc": "hosting_btc_mined",
                    }
                )

                combined_ch = combined_ch.merge(hosting_daily_ch, on="date", how="left")
                combined_ch["hosting_energy_kwh"] = combined_ch["hosting_energy_kwh"].fillna(0.0)
                combined_ch["hosting_revenue_usd"] = combined_ch["hosting_revenue_usd"].fillna(0.0)
                combined_ch["hosting_btc_mined"] = combined_ch["hosting_btc_mined"].fillna(0.0)
            else:
                combined_ch["hosting_energy_kwh"] = 0.0
                combined_ch["hosting_revenue_usd"] = 0.0
                combined_ch["hosting_btc_mined"] = 0.0

            # Energia DESMO = totale Synota - hosting
            combined_ch["energy_total_kwh"] = combined_ch["energy_mwh"] * 1000.0
            combined_ch["energy_desmo_kwh"] = combined_ch["energy_total_kwh"] - combined_ch["hosting_energy_kwh"]
            combined_ch["energy_desmo_kwh"] = combined_ch["energy_desmo_kwh"].clip(lower=0.0)

            # Profitto giornaliero DESMO (mining + hosting - costi energia)
            combined_ch["daily_profit_usd"] = (
                combined_ch["earnings_usd"] + combined_ch["hosting_revenue_usd"] - combined_ch["invoice_amount_usd"]
            )

            # Uptime giornaliero stimato (hashrate / max hashrate del periodo)
            installed_hashrate = combined_ch["daily_hashrate_ths"].max() if not combined_ch.empty else 0.0
            if installed_hashrate > 0:
                combined_ch["uptime_pct"] = combined_ch["daily_hashrate_ths"] / installed_hashrate * 100.0
            else:
                combined_ch["uptime_pct"] = 0.0

            combined_ch = combined_ch.sort_values("date")

            # =========================
            # 1) PROFITTO GIORNALIERO (barre rosse/verde)
            # =========================
            st.markdown("### 💰 Profitto giornaliero (DESMO)")

            df_p = combined_ch.copy()
            df_p["profit_positive"] = df_p["daily_profit_usd"].apply(lambda x: x if x > 0 else None)
            df_p["profit_negative"] = df_p["daily_profit_usd"].apply(lambda x: x if x < 0 else None)

            fig_p = go.Figure()

            # Barre positive → verde
            fig_p.add_bar(
                x=df_p["date"],
                y=df_p["profit_positive"],
                name="Profitto",
                marker_color="green",
            )

            # Barre negative → rosso
            fig_p.add_bar(
                x=df_p["date"],
                y=df_p["profit_negative"],
                name="Perdita",
                marker_color="red",
            )

            # Media mobile 7 giorni (linea)
            df_p["profit_roll7"] = df_p["daily_profit_usd"].rolling(7).mean()
            fig_p.add_scatter(
                x=df_p["date"],
                y=df_p["profit_roll7"],
                mode="lines",
                name="Media mobile 7g",
                line=dict(color="orange", width=3),
            )

            fig_p.update_layout(
                xaxis_title="Data",
                yaxis_title="Profitto / Perdita [USD]",
                hovermode="x unified",
                barmode="overlay",  # sovrappone le barre
            )

            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("---")


            # =========================
            # 2) HASHRATE & UPTIME
            # =========================
            st.markdown("### ⛏️ Hashrate & uptime DESMO")

            fig_h = go.Figure()
            fig_h.add_bar(
                x=combined_ch["date"],
                y=combined_ch["daily_hashrate_ths"],
                name="Hashrate [TH/s]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_h.add_scatter(
                x=combined_ch["date"],
                y=combined_ch["uptime_pct"],
                mode="lines+markers",
                name="Uptime [%]",
                yaxis="y2",
                line=dict(color="gold", width=2),       # giallo
            )

            fig_h.update_layout(
                xaxis_title="Data",
                yaxis=dict(title="Hashrate [TH/s]"),
                yaxis2=dict(
                    title="Uptime [%]",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_h, use_container_width=True)

            st.markdown("---")

            # =========================
            # 3) BTC MINATI & RICAVI MINING
            # =========================
            st.markdown("### ₿ BTC minati & ricavi da mining (DESMO)")

            fig_btc = go.Figure()
            fig_btc.add_bar(
                x=combined_ch["date"],
                y=combined_ch["total_earnings_btc"],
                name="BTC minati [BTC/day]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_btc.add_scatter(
                x=combined_ch["date"],
                y=combined_ch["earnings_usd"],
                mode="lines+markers",
                name="Ricavi mining [USD/day]",
                yaxis="y2",
                line=dict(color="gold", width=2),       # giallo
            )

            fig_btc.update_layout(
                xaxis_title="Data",
                yaxis=dict(title="BTC/day"),
                yaxis2=dict(
                    title="Ricavi mining [USD/day]",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_btc, use_container_width=True)

            st.markdown("---")

            # =========================
            # 4) ENERGIA: COSTO, MWh, TARIFFA
            # =========================
            st.markdown("### ⚡ Energia: MWh, costi e tariffa [USD/MWh]")

            fig_e = go.Figure()
            fig_e.add_bar(
                x=combined_ch["date"],
                y=combined_ch["energy_mwh"],
                name="Energia [MWh]",
                marker_color="rgba(55, 150, 100, 0.9)",  # blu
            )
            fig_e.add_bar(
                x=combined_ch["date"],
                y=combined_ch["invoice_amount_usd"],
                name="Costo energia [USD]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_e.add_scatter(
                x=combined_ch["date"],
                y=combined_ch["effective_rate_usd_per_mwh"],
                mode="lines+markers",
                name="Tariffa [USD/MWh]",
                yaxis="y2",
                line=dict(color="gold", width=2),
            )

            fig_e.update_layout(
                barmode="group",
                xaxis_title="Data",
                yaxis=dict(title="Energia [MWh] / Costo [USD]"),
                yaxis2=dict(
                    title="Tariffa [USD/MWh]",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_e, use_container_width=True)

            st.markdown("---")

            # =========================
            # 5) RICAVI HOSTING GIORNALIERI
            # =========================
            st.markdown("### 🤝 Ricavi hosting giornalieri [USD]")

            fig_host = go.Figure()
            fig_host.add_bar(
                x=combined_ch["date"],
                y=combined_ch["hosting_revenue_usd"],
                name="Ricavi hosting [USD/day]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_host.update_layout(
                xaxis_title="Data",
                yaxis_title="Ricavi hosting [USD/day]",
                hovermode="x unified",
            )
            st.plotly_chart(fig_host, use_container_width=True)

            st.markdown("---")

            # =========================
            # 6) COSTO MEDIO ENERGIA – TOTALE vs 7g vs 30g
            # =========================
            st.markdown("### 📐 Costo medio energia – totale vs 7g vs 30g")

            def avg_cost_over_days_ch(days: int) -> float | None:
                if combined_ch.empty:
                    return None
                last_d = combined_ch["date"].max()
                start_d = last_d - pd.Timedelta(days=days - 1)
                df_sub = combined_ch[combined_ch["date"] >= start_d]
                sub_mwh = df_sub["energy_mwh"].sum()
                sub_cost = df_sub["invoice_amount_usd"].sum()
                return sub_cost / sub_mwh if sub_mwh > 0 else None

            total_mwh_ch = combined_ch["energy_mwh"].sum()
            total_cost_ch = combined_ch["invoice_amount_usd"].sum()
            cost_total = total_cost_ch / total_mwh_ch if total_mwh_ch > 0 else None
            cost_7d = avg_cost_over_days_ch(7)
            cost_30d = avg_cost_over_days_ch(30)

            labels = []
            values = []
            if cost_total is not None:
                labels.append("Totale")
                values.append(cost_total)
            if cost_7d is not None:
                labels.append("Ultimi 7g")
                values.append(cost_7d)
            if cost_30d is not None:
                labels.append("Ultimi 30g")
                values.append(cost_30d)

            if labels:
                fig_ce = go.Figure()
                fig_ce.add_bar(
                    x=labels,
                    y=values,
                    name="Costo medio [USD/MWh]",
                    marker_color="rgba(0, 123, 255, 0.9)",  # blu
                )
                fig_ce.update_layout(
                    yaxis_title="Costo medio energia [USD/MWh]",
                    hovermode="x",
                )
                st.plotly_chart(fig_ce, use_container_width=True)

            st.markdown("---")

            # =========================
            # 7) DESMO vs CLIENTI – BTC & USD
            # =========================
            st.markdown("### 🧱 DESMO vs clienti – BTC & USD")

            # BTC DESMO (Antpool principale)
            btc_desmo_total_ch = antpool_filtered["total_earnings_btc"].sum() if antpool_filtered is not None else 0.0

            # BTC CLIENTI
            if hosting_filtered is not None and not hosting_filtered.empty:
                btc_clients_total_ch = hosting_filtered["total_earnings_btc"].sum()
            else:
                btc_clients_total_ch = 0.0

            # Valore in USD DESMO
            if antpool_filtered is not None and not antpool_filtered.empty:
                df_desmo_val = antpool_filtered.copy()
                if btc_price_df is not None and not btc_price_df.empty:
                    df_desmo_val = df_desmo_val.merge(btc_price_df, on="date", how="left")
                    df_desmo_val["btc_price_usd"] = df_desmo_val["btc_price_usd"].fillna(
                        btc_price_used if btc_price_used else 0.0
                    )
                else:
                    df_desmo_val["btc_price_usd"] = btc_price_used if btc_price_used else 0.0
                df_desmo_val["usd_value"] = df_desmo_val["total_earnings_btc"] * df_desmo_val["btc_price_usd"]
                usd_desmo_total_ch = df_desmo_val["usd_value"].sum()
            else:
                usd_desmo_total_ch = 0.0

            # Valore in USD CLIENTI
            if hosting_filtered is not None and not hosting_filtered.empty:
                df_cli_val = hosting_filtered.copy()
                if btc_price_df is not None and not btc_price_df.empty:
                    df_cli_val = df_cli_val.merge(btc_price_df, on="date", how="left")
                    df_cli_val["btc_price_usd"] = df_cli_val["btc_price_usd"].fillna(
                        btc_price_used if btc_price_used else 0.0
                    )
                else:
                    df_cli_val["btc_price_usd"] = btc_price_used if btc_price_used else 0.0
                df_cli_val["usd_value"] = df_cli_val["total_earnings_btc"] * df_cli_val["btc_price_usd"]
                usd_clients_total_ch = df_cli_val["usd_value"].sum()
            else:
                usd_clients_total_ch = 0.0

            # Grafico BTC DESMO vs CLIENTI
            fig_btc_split = go.Figure()
            fig_btc_split.add_bar(
                x=["DESMO", "Clienti"],
                y=[btc_desmo_total_ch, btc_clients_total_ch],
                name="BTC minati [BTC]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_btc_split.update_layout(
                yaxis_title="BTC minati [BTC]",
                hovermode="x",
            )
            st.plotly_chart(fig_btc_split, use_container_width=True)

            # Grafico valore USD DESMO vs CLIENTI
            fig_usd_split = go.Figure()
            fig_usd_split.add_bar(
                x=["DESMO", "Clienti"],
                y=[usd_desmo_total_ch, usd_clients_total_ch],
                name="Valore BTC minati [USD]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_usd_split.update_layout(
                yaxis_title="Valore BTC minati [USD]",
                hovermode="x",
            )
            st.plotly_chart(fig_usd_split, use_container_width=True)

            st.markdown("---")

            # =========================
            # 8) BTC DESMO: MINATI vs VENDUTI vs WALLET
            # =========================
            st.markdown("### 🔗 BTC DESMO – minati, venduti (matched) e su wallet")

            # BTC venduti (matched) calcolati nella tab Kraken e salvati in session_state
            btc_sold_auto = float(st.session_state.get("btc_sold_matched", 0.0))

            # Saldo on-chain del wallet DESMO
            wallet_balance_btc_ch = get_wallet_balance_btc(DESMO_WALLET_ADDRESS)
            
            values_btc = [
                btc_desmo_total_ch,
                btc_sold_auto,
                wallet_balance_btc_ch if wallet_balance_btc_ch is not None else 0.0,
            ]
            labels_btc = ["BTC minati DESMO", "BTC venduti (matched)", "BTC su wallet DESMO"]

            fig_btc_flow = go.Figure()
            fig_btc_flow.add_bar(
                x=labels_btc,
                y=values_btc,
                name="BTC [BTC]",
                marker_color="rgba(0, 123, 255, 0.9)",  # blu
            )
            fig_btc_flow.update_layout(
                yaxis_title="BTC [BTC]",
                hovermode="x",
            )
            st.plotly_chart(fig_btc_flow, use_container_width=True)
