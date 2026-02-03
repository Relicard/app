import streamlit as st # type: ignore 
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
from datetime import datetime, timedelta  # type: ignore
import requests # type: ignore 
import calendar


# -----------------------------
# CONFIGURAZIONE BASE STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="DESMO Power & Mining Dashboard",
    layout="wide",
)

if "selected_sell_avg_price" not in st.session_state:
    st.session_state["selected_sell_avg_price"] = None


# -----------------------------
# FUNZIONI DI UTILITY
# -----------------------------
def parse_synota_csv(file) -> pd.DataFrame:
    """Parsa il CSV Synota con i campi: Settlement Start, Energy Delivered, Invoice Amount, Effective Rate."""
    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["Settlement Start"], format="%m/%d/%Y")

    def parse_energy(s):
        if isinstance(s, str):
            s = s.replace("kWh", "").replace(",", "").strip()
            return float(s) if s else None
        return s

    df["energy_kwh"] = df["Energy Delivered"].apply(parse_energy)
    df["energy_mwh"] = df["energy_kwh"] / 1000.0

    def parse_money(s):
        if isinstance(s, str):
            s = s.replace("$", "").replace(",", "").strip()
            return float(s) if s else None
        return s

    df["invoice_amount_usd"] = df["Invoice Amount"].apply(parse_money)

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

    df = df.sort_values("date")
    return df


import csv
import io

def parse_chase_activity_csv(file) -> pd.DataFrame:
    """
    Parsa l'estratto conto Chase 'Activity' (CSV).
    Output standardizzato:
      - date (datetime normalizzato a giorno)
      - description (str)
      - amount (float, positivo per CREDIT e negativo per DEBIT)
      - direction ("DEBIT"/"CREDIT")
      - raw_type (colonna Type se presente)
    """
    # Streamlit uploader -> file-like. Lo leggiamo come testo.
    raw = file.getvalue()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)

    f = io.StringIO(text)
    reader = csv.reader(f)
    rows = list(reader)
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = rows[1:]

    # Chase spesso mette più colonne del header (colonne vuote in coda).
    max_len = max(len(r) for r in data_rows) if data_rows else len(header)
    if len(header) < max_len:
        header = header + [f"extra_{i}" for i in range(max_len - len(header))]

    # normalizza righe a max_len
    fixed = []
    for r in data_rows:
        if len(r) < max_len:
            r = r + [""] * (max_len - len(r))
        fixed.append(r[:max_len])

    df = pd.DataFrame(fixed, columns=header)

    # Colonne attese (possono variare leggermente)
    # Tipico: Details, Posting Date, Description, Amount, Type, Balance, Check or Slip #
    if "Posting Date" not in df.columns or "Amount" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["Posting Date"], format="%m/%d/%Y", errors="coerce").dt.normalize()
    df["description"] = df["Description"].astype(str).str.strip()

    # Amount -> float
    df["amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(",", ""), errors="coerce")

    # Details indica DEBIT/CREDIT
    if "Details" in df.columns:
        df["direction"] = df["Details"].astype(str).str.strip().str.upper()
    else:
        df["direction"] = np.where(df["amount"] < 0, "DEBIT", "CREDIT")

    df["raw_type"] = df["Type"].astype(str).str.strip() if "Type" in df.columns else ""

    # pulizia base
    df = df.dropna(subset=["date", "amount"]).copy()
    df = df.sort_values("date")

    # Standardizziamo: DEBIT negativo, CREDIT positivo
    # (in questo CSV sembra già così, ma lo rendiamo certo)
    df.loc[df["direction"].str.contains("DEBIT", na=False), "amount"] = -df.loc[
        df["direction"].str.contains("DEBIT", na=False), "amount"
    ].abs()
    df.loc[df["direction"].str.contains("CREDIT", na=False), "amount"] = df.loc[
        df["direction"].str.contains("CREDIT", na=False), "amount"
    ].abs()

    return df[["date", "description", "amount", "direction", "raw_type"]]


def parse_prometheus_usage(file) -> pd.DataFrame:
    """
    Parsa il CSV di Prometheus (Hourly Energy Usage and Cost).
    Restituisce un dataframe giornaliero con:
      - date (datetime)
      - energy_kwh
      - energy_mwh
      - invoice_amount_usd
      - effective_rate_usd_per_mwh
    """

    df0 = pd.read_csv(file)

    header = df0.iloc[1]
    df = df0.iloc[2:].copy()
    df.columns = header.values

    df = df[df["Date"].notna()].copy()

    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")

    def parse_num(x):
        if isinstance(x, (int, float)):
            return float(x)
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        # rimuove separatore migliaia e converte la virgola in punto
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    # Carico medio in MW per l'ora (WEST)
    df["avg_load_mw"] = df["Avg Load (MW)"].map(parse_num)

    # Costo energia per ora (prima colonna "Energy Cost for Hour ($)" = WEST)
    # NB: nel CSV questa colonna è la quinta (indice 4)
    df["hourly_cost_usd"] = df.iloc[:, 4].map(parse_num)

    # Ogni riga è un'ora, quindi MWh = MW * 1h
    df["energy_mwh_hour"] = df["avg_load_mw"]

    # Aggregazione giornaliera
    daily = df.groupby("date", as_index=False).agg(
        energy_mwh=("energy_mwh_hour", "sum"),
        invoice_amount_usd=("hourly_cost_usd", "sum"),
    )

    # kWh e tariffa media giornaliera
    daily["energy_kwh"] = daily["energy_mwh"] * 1000.0
    daily["effective_rate_usd_per_mwh"] = daily.apply(
        lambda r: (r["invoice_amount_usd"] / r["energy_mwh"])
        if r["energy_mwh"] and r["energy_mwh"] > 0
        else None,
        axis=1,
    )

    # Ordina e restituisce con lo schema compatibile con Synota
    daily = daily[
        [
            "date",
            "energy_kwh",
            "energy_mwh",
            "invoice_amount_usd",
            "effective_rate_usd_per_mwh",
        ]
    ].sort_values("date")

    return daily



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


def parse_ercot_price_csv(file) -> pd.DataFrame:
    """
    Parsa il CSV dei prezzi ERCOT per LZ_WEST.
    Si aspetta almeno colonne:
    - interval_start_local  (timestamp locale, es. 2025-11-10T00:00:00-06:00)
    - lmp                   (prezzo in USD/MWh)
    - location              (filtriamo LZ_WEST)
    """
    df = pd.read_csv(file)

    # Filtra LZ_WEST se la colonna esiste
    if "location" in df.columns:
        df = df[df["location"] == "LZ_WEST"].copy()

    # Scegliamo la colonna timestamp principale
    if "interval_start_local" in df.columns:
        df["date"] = pd.to_datetime(df["interval_start_local"], errors="coerce")
    elif "interval_start_utc" in df.columns:
        df["date"] = pd.to_datetime(df["interval_start_utc"], errors="coerce")
    else:
        return pd.DataFrame()  # niente timestamp → niente grafico

    # ⛔ QUI era il problema: date con timezone
    # ➜ le rendiamo tutte "naive" (senza tz) così sono compatibili con Synota/Antpool
    from pandas.api.types import is_datetime64tz_dtype #type: ignore

    if is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert(None)

    # Prezzo LMP → colonna numerica
    if "lmp" in df.columns:
        df["lmp_usd_mwh"] = pd.to_numeric(df["lmp"], errors="coerce")
    else:
        return pd.DataFrame()

    # Pulisci righe non valide e ordina
    df = df.dropna(subset=["date", "lmp_usd_mwh"])
    df = df.sort_values("date")

    return df




@st.cache_data(show_spinner=False)
def parse_prometheus_rtm_prices_excel(file) -> pd.DataFrame:
    """Parsa l'Excel 'RTM Price Extract' (Prometheus) con prezzi RTM per LoadZone/HUB.

    Struttura attesa (prima riga = header logico):
      Date | LoadZone | Settlement | Intervals | Sum | Min | Max | Avg | 1..100 (prezzi 15-min)

    Note:
    - Intervals può essere 92/96/100 a seconda del giorno (DST). Le colonne 1..100 sono sempre presenti.
    - I prezzi sono in USD/MWh.
    """
    df0 = pd.read_excel(file, sheet_name=0)

    if df0.empty:
        return pd.DataFrame()

    # La prima riga contiene i nomi colonna reali
    header = df0.iloc[0].tolist()
    df = df0.iloc[1:].copy()
    df.columns = header

    if "Date" not in df.columns:
        return pd.DataFrame()

    # Data (normalizzata a giorno)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()

    # Normalizza testo
    for c in ["LoadZone", "Settlement"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()

    # Colonne intervalli: 1..100
    interval_cols = [
        c for c in df.columns
        if isinstance(c, (int, float)) and float(c).is_integer()
    ]
    # Rinominamo in int (1..100) per semplicità
    rename_map = {c: int(c) for c in interval_cols}
    df = df.rename(columns=rename_map)

    # Converte numerici principali
    numeric_cols = []
    for c in ["Intervals", "Sum", "Min", "Max", "Avg"]:
        if c in df.columns:
            numeric_cols.append(c)
    numeric_cols += [int(c) for c in interval_cols]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Default Intervals se mancante
    if "Intervals" not in df.columns:
        df["Intervals"] = 96

    keep = ["date", "LoadZone", "Settlement", "Intervals", "Avg", "Min", "Max"]
    keep = [c for c in keep if c in df.columns] + sorted([c for c in df.columns if isinstance(c, int)])

    return df[keep].sort_values("date")


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




def parse_gridmatic_csv(file) -> pd.DataFrame:
    """Parsa un CSV Gridmatic con almeno le colonne: month, value_usd.

    - Accetta month in formati tipici (es. '2025-11-01', '2025-11', '11/2025', ecc.)
    - Normalizza month al primo giorno del mese (timestamp)
    - Rimuove colonne inutili (es. 'Unnamed: 0') se presenti
    - Aggrega eventuali mesi duplicati sommando value_usd
    """
    try:
        df = pd.read_csv(file)

        # Drop tipiche colonne indice
        drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if "month" not in df.columns or "value_usd" not in df.columns:
            return pd.DataFrame(columns=["month", "value_usd"])

        out = df[["month", "value_usd"]].copy()

        out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        out["value_usd"] = pd.to_numeric(out["value_usd"], errors="coerce")

        out = out.dropna(subset=["month", "value_usd"])
        if out.empty:
            return pd.DataFrame(columns=["month", "value_usd"])

        out = (
            out.groupby("month", as_index=False)["value_usd"]
            .sum()
            .sort_values("month")
            .reset_index(drop=True)
        )

        return out

    except Exception:
        return pd.DataFrame(columns=["month", "value_usd"])


# -----------------------------
# GRIDMATIC – ADJUSTMENT LAYER
# -----------------------------
def _month_shift(year: int, month: int, delta_months: int) -> tuple[int, int]:
    """Shift a (year, month) by delta_months, returning (year, month)."""
    m = month + delta_months
    y = year
    while m <= 0:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return y, m


def build_gridmatic_daily_adjustments(entries: pd.DataFrame) -> pd.DataFrame:
    """Build a daily adjustment series from Gridmatic monthly entries.

    Business rule:
      - Each (month, value_usd) applies from the 25th of the previous month (inclusive)
        to the 25th of the selected month (exclusive).
      - The value is spread evenly across the days in that window:
          daily = value_usd / number_of_days_in_window
      - Multiple months can overlap; adjustments are summed per day.
    """
    if entries is None or entries.empty:
        return pd.DataFrame(columns=["date", "gridmatic_adj_usd"])

    df = entries.copy()

    # Normalize columns
    if "month" not in df.columns or "value_usd" not in df.columns:
        return pd.DataFrame(columns=["date", "gridmatic_adj_usd"])

    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.normalize()
    df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")

    df = df.dropna(subset=["month", "value_usd"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "gridmatic_adj_usd"])

    rows = []
    for _, r in df.iterrows():
        month_dt: pd.Timestamp = r["month"]
        value = float(r["value_usd"])

        y = int(month_dt.year)
        m = int(month_dt.month)

        py, pm = _month_shift(y, m, -1)

        start = datetime(py, pm, 25)
        end = datetime(y, m, 25)  # exclusive

        days = (end - start).days
        if days <= 0:
            continue

        daily = value / days

        # Build daily date range [start, end)
        for d in pd.date_range(start=start, end=end - timedelta(days=1), freq="D"):
            rows.append({"date": pd.to_datetime(d).normalize(), "gridmatic_adj_usd": daily})

    if not rows:
        return pd.DataFrame(columns=["date", "gridmatic_adj_usd"])

    adj = pd.DataFrame(rows)
    adj = adj.groupby("date", as_index=False)["gridmatic_adj_usd"].sum()
    return adj.sort_values("date")


def apply_gridmatic_adjustments(energy_df: pd.DataFrame, gridmatic_entries: pd.DataFrame) -> pd.DataFrame:
    """Apply Gridmatic daily adjustments to an energy dataframe (Synota/Prometheus-compatible).

    - Preserves original invoice_amount_usd in 'invoice_amount_usd_original' (if present)
    - Writes adjusted cost back into 'invoice_amount_usd'
    - Recomputes effective_rate_usd_per_mwh if possible
    """
    if energy_df is None or energy_df.empty:
        return energy_df

    if "invoice_amount_usd" not in energy_df.columns:
        return energy_df

    adj = build_gridmatic_daily_adjustments(gridmatic_entries)
    if adj.empty:
        # Still ensure original column exists (for debugging)
        if "invoice_amount_usd_original" not in energy_df.columns:
            out = energy_df.copy()
            out["invoice_amount_usd_original"] = out["invoice_amount_usd"]
            out["gridmatic_adj_usd"] = 0.0
            return out
        return energy_df

    out = energy_df.copy()
    if "invoice_amount_usd_original" not in out.columns:
        out["invoice_amount_usd_original"] = out["invoice_amount_usd"]

    # Re-base to the original invoice before applying adjustments (avoids double counting on reruns)
    out["invoice_amount_usd"] = out["invoice_amount_usd_original"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.merge(adj, on="date", how="left")
    out["gridmatic_adj_usd"] = out["gridmatic_adj_usd"].fillna(0.0)

    # Adjust invoice
    out["invoice_amount_usd"] = out["invoice_amount_usd"].astype(float) + out["gridmatic_adj_usd"].astype(float)

    # Recompute effective rate where possible
    if "energy_mwh" in out.columns:
        out["effective_rate_usd_per_mwh"] = out.apply(
            lambda r: (r["invoice_amount_usd"] / r["energy_mwh"])
            if pd.notna(r.get("invoice_amount_usd")) and pd.notna(r.get("energy_mwh")) and r["energy_mwh"] and r["energy_mwh"] > 0
            else r.get("effective_rate_usd_per_mwh"),
            axis=1,
        )

    return out.sort_values("date")





# -----------------------------
# SIDEBAR: CARICAMENTO & FILTRI
# -----------------------------
st.sidebar.title("⚙️ Configurazione")

# 1. CSV principali
st.sidebar.subheader("1. Carica i CSV principali")

# Sorgente principale per i dati di ENERGIA
data_source = st.sidebar.radio(
    "Fonte dati energia",
    options=["Synota", "Prometheus"],
    index=0,
    help="Scegli se usare i dati giornalieri Synota (energia + costi) oppure Prometheus (solo energia).",
)

# Uploader per i file
synota_file = st.sidebar.file_uploader("Synota CSV (energia/costi)", type=["csv"], key="synota")
prometheus_file = st.sidebar.file_uploader(
    "Prometheus CSV (Hourly Energy Usage & Cost)",
    type=["csv"],
    key="prometheus",
)
antpool_file = st.sidebar.file_uploader("Antpool CSV (hashrate/ricavi)", type=["csv"], key="antpool")

# Dataframe base
synota_df = None
prometheus_df = None
antpool_df = None

if synota_file is not None:
    synota_df = parse_synota_csv(synota_file)

if prometheus_file is not None:
    prometheus_df = parse_prometheus_usage(prometheus_file)

if antpool_file is not None:
    antpool_df = parse_antpool_csv(antpool_file)

# Dataframe di energia effettivamente usato nel resto dell’app
energia_source_label = data_source  # "Synota" oppure "Prometheus"
if data_source == "Synota":
    energia_df = synota_df
else:
    energia_df = prometheus_df

# Alias per compatibilità: da qui in giù il codice continua a usare synota_df / synota_filtered
synota_df = energia_df



st.sidebar.subheader("2. Vendite BTC (Kraken)")

kraken_file = st.sidebar.file_uploader(
    "CSV vendite BTC (esportazione Kraken)",
    type=["csv"],
    key="kraken_trades",
)

kraken_trades_df = None
if kraken_file is not None:
    kraken_trades_df = parse_kraken_trades_csv(kraken_file)

# 3. Prezzi ERCOT LZ_WEST (storico 5 minuti)
st.sidebar.subheader("3. Prezzi ERCOT (LZ_WEST)")

# 👉 Inizializzazione per evitare NameError
ercot_df = None

ercot_file = st.sidebar.file_uploader(
    "CSV prezzi ERCOT LZ_WEST (5 min)",
    type=["csv"],
    key="ercot_prices",
)

if ercot_file is not None:
    ercot_df = parse_ercot_price_csv(ercot_file)
    

st.sidebar.subheader("8. Estratto conto (riconciliazione)")

bank_file = st.sidebar.file_uploader(
    "Chase Activity CSV (bank statement)",
    type=["csv"],
    key="chase_activity",
)

bank_df = None
if bank_file is not None:
    bank_df = parse_chase_activity_csv(bank_file)
    if bank_df is not None and not bank_df.empty:
        st.sidebar.success(
            f"Estratto conto caricato: {bank_df['date'].min().date()} → {bank_df['date'].max().date()}"
        )
    else:
        st.sidebar.error("CSV Chase non valido o vuoto.")


# 3b. Prezzi RTM (Prometheus extract - Excel)
st.sidebar.subheader("3b. Prezzi RTM (Prometheus extract)")

rtm_prices_df = None
rtm_prices_file = st.sidebar.file_uploader(
    "Prometheus RTM Price Extract (xlsx)",
    type=["xlsx"],
    key="prometheus_rtm_prices_xlsx",
)

if rtm_prices_file is not None:
    rtm_prices_df = parse_prometheus_rtm_prices_excel(rtm_prices_file)


st.sidebar.subheader("4. Prezzo Bitcoin")


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
st.sidebar.subheader("5. CSV prezzo BTC (opzionale)")

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
st.sidebar.subheader("6. Hosting clients")

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





# 7. Gridmatic (adjustment su costo elettrico)
st.sidebar.subheader("7. Gridmatic")

st.sidebar.caption(
    "Inserisci uno o più aggiustamenti mensili (USD) che modificano il costo elettrico finale. "
    # "Regola: il valore del mese selezionato vale dal 25 del mese precedente (incluso) al 25 del mese (escluso) "
    # "ed è spalmato uniformemente sui giorni del periodo. Valori negativi diminuiscono il costo, positivi lo aumentano."
)

# --- Import CSV (opzionale) ---
gridmatic_csv = st.sidebar.file_uploader(
    "Carica CSV Gridmatic (colonne: month, value_usd)",
    type=["csv"],
    key="gridmatic_csv",
)

if "gridmatic_entries" not in st.session_state:
    # Default: una riga vuota sul mese corrente
    st.session_state["gridmatic_entries"] = pd.DataFrame(
        [{"month": datetime.today().replace(day=1), "value_usd": 0.0}]
    )

# Se l'utente carica un CSV, può importare (sovrascrivere o aggiungere)
if gridmatic_csv is not None:
    imported = parse_gridmatic_csv(gridmatic_csv)

    if imported.empty:
        st.sidebar.error("CSV Gridmatic non valido. Servono colonne 'month' e 'value_usd'.")
    else:
        st.sidebar.success(f"CSV Gridmatic letto: {len(imported)} righe")
        with st.sidebar.expander("Anteprima CSV Gridmatic", expanded=False):
            st.dataframe(imported, use_container_width=True)

        c1, c2 = st.sidebar.columns(2)
        if c1.button("Sovrascrivi da CSV", key="gridmatic_overwrite_btn"):
            st.session_state["gridmatic_entries"] = imported.copy()

        if c2.button("Aggiungi da CSV", key="gridmatic_append_btn"):
            base = st.session_state["gridmatic_entries"].copy()
            combined = pd.concat([base, imported], ignore_index=True)

            # Normalizza mese al primo giorno e somma eventuali duplicati
            combined["month"] = pd.to_datetime(combined["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            combined["value_usd"] = pd.to_numeric(combined["value_usd"], errors="coerce")
            combined = combined.dropna(subset=["month", "value_usd"])
            if not combined.empty:
                combined = (
                    combined.groupby("month", as_index=False)["value_usd"]
                    .sum()
                    .sort_values("month")
                    .reset_index(drop=True)
                )
            st.session_state["gridmatic_entries"] = combined

# --- Editor manuale (sempre disponibile) ---
gridmatic_entries = st.sidebar.data_editor(
    st.session_state["gridmatic_entries"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "month": st.column_config.DateColumn("Mese (seleziona un giorno del mese)", format="YYYY-MM-DD"),
        "value_usd": st.column_config.NumberColumn("Valore [USD] (+/-)", step=100.0, format="%.2f"),
    },
    key="gridmatic_entries_editor",
)

# Salva in sessione (persistenza) + normalizzazione
gridmatic_entries = gridmatic_entries.copy()
gridmatic_entries["month"] = pd.to_datetime(gridmatic_entries["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
gridmatic_entries["value_usd"] = pd.to_numeric(gridmatic_entries["value_usd"], errors="coerce")
gridmatic_entries = gridmatic_entries.dropna(subset=["month", "value_usd"])
if not gridmatic_entries.empty:
    # Se per errore l'utente inserisce due righe stesso mese, le sommiamo
    gridmatic_entries = (
        gridmatic_entries.groupby("month", as_index=False)["value_usd"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

st.session_state["gridmatic_entries"] = gridmatic_entries

# Applichiamo gli aggiustamenti al dataframe energia (Synota/Prometheus) se disponibile
if synota_df is not None and not synota_df.empty:
    synota_df = apply_gridmatic_adjustments(synota_df, gridmatic_entries)


# 6. Range date globale (Synota / Antpool / ERCOT)
if synota_df is not None or antpool_df is not None or ercot_df is not None or rtm_prices_df is not None:
    all_dates = []
    if synota_df is not None:
        all_dates.extend(list(synota_df["date"]))
    if antpool_df is not None:
        all_dates.extend(list(antpool_df["date"]))
    if ercot_df is not None:
        all_dates.extend(list(ercot_df["date"]))
    if rtm_prices_df is not None:
        all_dates.extend(list(rtm_prices_df["date"]))

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    st.sidebar.subheader("9. Filtro timeframe")

    # 👇 Preset rapidi
    preset = st.sidebar.radio(
        "Seleziona periodo rapido",
        [
            "Tutti i dati",
            "Ultima data disponibile",
            "Giorno specifico",
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

    elif preset == "Giorno specifico":
        one_day = st.sidebar.date_input(
            "Seleziona il giorno",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(one_day, tuple):
            one_day = one_day[0]
        start_date = one_day
        end_date = one_day

    elif preset == "Ultimi 7 giorni":
        start_date = max(min_date, max_date - timedelta(days=6))
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
if synota_df is None and antpool_df is None and hosting_df_all is None and ercot_df is None and rtm_prices_df is None:
    st.info("👈 Carica almeno un CSV nella sidebar per iniziare.")
    st.stop()


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

    if ercot_df is not None:
        ercot_filtered = filter_by_date(ercot_df, start_date, end_date)
    else:
        ercot_filtered = None

    if rtm_prices_df is not None:
        rtm_prices_filtered = filter_by_date(rtm_prices_df, start_date, end_date)
    else:
        rtm_prices_filtered = None

    if prometheus_df is not None:
        prometheus_filtered = filter_by_date(prometheus_df, start_date, end_date)
    else:
        prometheus_filtered = None
else:
    synota_filtered = synota_df
    antpool_filtered = antpool_df
    hosting_filtered = hosting_df_all
    kraken_filtered = kraken_trades_df
    ercot_filtered = ercot_df
    rtm_prices_filtered = rtm_prices_df
    prometheus_filtered = prometheus_df




# -----------------------------
# TABS PRINCIPALI
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
    [
        "📌 Overview & Trends",
        "📊 Charts",
        "⚡ Energia (Synota/Prometheus)",
        "⛏️ Mining (Antpool)",
        "🤝 Hosting",
        "🧠 Smart Analytics",
        "📈 Metriche globali",
        "🔗 Combined View",
        "💰 Vendite BTC",
        "⚡ Prezzi ERCOT LZ_WEST",
        "⚡ Prometheus (dettaglio)",
        "⚡ RTM Prices (Zones/Hubs)",
        "🧾 Riconciliazione pagamento elettrico",
    ]
)

# -----------------------------
# TAB 3 – ENERGIA / SYNOTA / PROMETHEUS
# -----------------------------
with tab3:
    if energia_source_label == "Synota":
        st.header("⚡ Energia & Costi (Synota)")
    else:
        st.header("⚡ Energia & Costi (Prometheus)")

    if synota_filtered is None or synota_filtered.empty:
        st.warning("Nessun dato energia nel range selezionato per la fonte scelta.")
    else:
        col1, col2, col3 = st.columns(3)
        total_mwh = synota_filtered["energy_mwh"].sum()

        with col1:
            st.metric("Energia totale [MWh]", f"{total_mwh:,.2f}")

        # Se ci sono i costi (Synota o Prometheus) li mostriamo
        if (
            "invoice_amount_usd" in synota_filtered.columns
            and synota_filtered["invoice_amount_usd"].notna().any()
        ):
            total_cost = synota_filtered["invoice_amount_usd"].sum()
            avg_rate = total_cost / total_mwh if total_mwh > 0 else None

            with col2:
                st.metric("Costo totale [USD]", f"{total_cost:,.2f}")
            with col3:
                st.metric(
                    "Tariffa media [USD/MWh]",
                    f"{avg_rate:,.2f}" if avg_rate is not None else "N/A",
                )
        else:
            with col2:
                st.metric("Costo totale [USD]", "N/A")
            with col3:
                st.metric("Tariffa media [USD/MWh]", "N/A")

        st.markdown("### 📉 Grafico giornaliero")

        # Selezione campi da visualizzare
        show_energy = st.checkbox(
            "Mostra energia (MWh)", value=True, key="syn_energy"
        )

        # Verifichiamo se esistono i campi di costo/tariffa
        has_invoice = (
            "invoice_amount_usd" in synota_filtered.columns
            and synota_filtered["invoice_amount_usd"].notna().any()
        )
        has_rate = (
            "effective_rate_usd_per_mwh" in synota_filtered.columns
            and synota_filtered["effective_rate_usd_per_mwh"].notna().any()
        )

        show_invoice = False
        show_rate = False

        if has_invoice:
            show_invoice = st.checkbox(
                "Mostra Invoice Amount [USD]", value=True, key="syn_invoice"
            )
        if has_rate:
            show_rate = st.checkbox(
                "Mostra Effective Rate [USD/MWh]", value=True, key="syn_rate"
            )

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

        if energia_source_label != "Synota":
            st.info("Questo confronto è disponibile solo quando la fonte energia è Synota (dati di costo presenti).")
        else:
            synota_rel = synota_filtered.copy()

            # Base = media
            base_energy = synota_rel["energy_mwh"].mean()
            base_cost = synota_rel["invoice_amount_usd"].mean()

            # Evita divisioni per zero / NaN
            if base_energy and pd.notna(base_cost) and base_cost != 0:
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


        st.markdown("### 📄 Dati energia (filtrati)")

        if energia_source_label == "Synota":
            df_to_show = synota_filtered[
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
            )
        else:
            df_to_show = synota_filtered[
                [
                    "date",
                    "energy_mwh",
                ]
            ].rename(
                columns={
                    "date": "Date",
                    "energy_mwh": "Energy [MWh]",
                }
            )

        st.dataframe(df_to_show, use_container_width=True)

        
        # -----------------------------
        # SCOMPOSIZIONE COSTI CONTRATTO SYNOTA / ONCOR
        # -----------------------------
        st.markdown("### 🧮 Scomposizione costi contratto Synota / Oncor (stima)")

        total_mwh = synota_filtered["energy_mwh"].sum()
        num_days = synota_filtered["date"].dt.normalize().nunique()
        total_cost = synota_filtered["invoice_amount_usd"].sum()
        avg_rate = total_cost / total_mwh if total_mwh > 0 else None

        if total_mwh <= 0 or num_days == 0 or total_cost <= 0:
            st.info("Impossibile calcolare la scomposizione: energia totale, numero giorni o invoice = 0.")
        else:
            st.caption(
                "Parametri contratto (puoi modificarli per simulare). "
                "I valori sono applicati ai soli giorni/record attualmente selezionati nel filtro."
            )

            colp1, colp2, colp3 = st.columns(3)
            with colp1:
                billed_kw = st.number_input(
                    "Potenza fatturata (kW, 4CP / capacità sito)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=1100.0,
                    step=50.0,
                    key="syn_cp_kw",
                )
            with colp2:
                tdsp_peak_rate = st.number_input(
                    "TDSP Peak Charge [$/kW-day]",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.3724,
                    step=0.01,
                    key="syn_tdsp_peak",
                )
            with colp3:
                tdsp_flat_rate = st.number_input(
                    "TDSP Flat Charge [$/day]",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.13,
                    step=0.1,
                    key="syn_tdsp_flat",
                )

            # --- Parametri contratto fissi per MWh ---
            adder = 4.10                      # $/MWh
            ercot_admin = 0.63                # $/MWh
            ancillary = 0.03                  # $/MWh
            securitization = 0.59             # $/MWh

            # TDSP usage è espresso in $/kWh → convertiamo in $/MWh
            tdsp_usage_per_kwh = 0.00055      # $/kWh
            tdsp_usage_per_mwh = tdsp_usage_per_kwh * 1000.0  # = 0.55 $/MWh

            # Tasse (in forma decimale)
            tax_state = 0.0625
            tax_local = 0.015
            tax_gross = 0.01997
            tax_pub = 0.001667
            total_tax_rate = tax_state + tax_local + tax_gross + tax_pub

            # Componenti "variabili" dirette per MWh
            variable_components_per_mwh = {
                "Adder commerciale": adder,
                "ERCOT System Admin": ercot_admin,
                "Ancillary Service": ancillary,
                "Securitization": securitization,
                "TDSP Usage": tdsp_usage_per_mwh,
            }

            # --- Componenti giornalieri: TDSP Flat + TDSP Peak (4CP stimato) ---
            tdsp_flat_total = tdsp_flat_rate * num_days
            tdsp_peak_total = tdsp_peak_rate * billed_kw * num_days

            tdsp_flat_per_mwh = tdsp_flat_total / total_mwh
            tdsp_peak_per_mwh = tdsp_peak_total / total_mwh

            # Costruzione tabella MICRO (dettaglio componenti)
            rows = []
            subtotal_pre_tax_per_mwh = 0.0
            subtotal_pre_tax_total = 0.0

            for name, rate_per_mwh in variable_components_per_mwh.items():
                total_usd = rate_per_mwh * total_mwh
                rows.append(
                    {
                        "Categoria": name,
                        "Costo totale [USD]": total_usd,
                        "Costo medio [USD/MWh]": rate_per_mwh,
                    }
                )
                subtotal_pre_tax_per_mwh += rate_per_mwh
                subtotal_pre_tax_total += total_usd

            # TDSP Flat
            rows.append(
                {
                    "Categoria": "TDSP Flat",
                    "Costo totale [USD]": tdsp_flat_total,
                    "Costo medio [USD/MWh]": tdsp_flat_per_mwh,
                }
            )
            # TDSP Peak
            rows.append(
                {
                    "Categoria": "TDSP Peak (4CP)",
                    "Costo totale [USD]": tdsp_peak_total,
                    "Costo medio [USD/MWh]": tdsp_peak_per_mwh,
                }
            )

            subtotal_pre_tax_per_mwh += tdsp_flat_per_mwh + tdsp_peak_per_mwh
            subtotal_pre_tax_total += tdsp_flat_total + tdsp_peak_total

            # --- Tasse applicate alle componenti sopra ---
            taxes_total = subtotal_pre_tax_total * total_tax_rate
            taxes_per_mwh = subtotal_pre_tax_per_mwh * total_tax_rate

            rows.append(
                {
                    "Categoria": "Tasse totali",
                    "Costo totale [USD]": taxes_total,
                    "Costo medio [USD/MWh]": taxes_per_mwh,
                }
            )

            estimated_total_per_mwh = subtotal_pre_tax_per_mwh * (1.0 + total_tax_rate)
            estimated_total_total = subtotal_pre_tax_total * (1.0 + total_tax_rate)

            rows.append(
                {
                    "Categoria": "Sub-Totale add-on",
                    "Costo totale [USD]": estimated_total_total,
                    "Costo medio [USD/MWh]": estimated_total_per_mwh,
                }
            )

            # --- Residuo: lo interpretiamo come "Energia ERCOT (residua)" ---
            ercot_residual_total = total_cost - estimated_total_total
            # se per qualche motivo viene negativo, lo lasciamo comunque per vedere che stiamo sovrastimando
            ercot_residual_per_mwh = ercot_residual_total / total_mwh

            rows.append(
                {
                    "Categoria": "Energia LMP ERCOT LZ_WEST",
                    "Costo totale [USD]": ercot_residual_total,
                    "Costo medio [USD/MWh]": ercot_residual_per_mwh,
                }
            )

            # Confronto con la fattura reale Synota del periodo
            rows.append(
                {
                    "Categoria": "Invoice (Synota)",
                    "Costo totale [USD]": total_cost,
                    "Costo medio [USD/MWh]": avg_rate,
                }
            )

            summary_df = pd.DataFrame(rows)

            # Peso % di ogni riga sulla invoice reale
            summary_df["Peso % su invoice"] = (
                summary_df["Costo totale [USD]"] / total_cost * 100.0
            )

            st.markdown("#### Micro-aree di costo (dettaglio)")
            st.dataframe(
                summary_df.style.format(
                    {
                        "Costo totale [USD]": "{:,.2f}",
                        "Costo medio [USD/MWh]": "{:,.2f}",
                        "Peso % su invoice": "{:,.1f}%",
                    }
                ),
                use_container_width=True,
            )

            # --- MACRO AREE: add-on vs ERCOT & residuo energia ---
            st.markdown("#### Macro-aree di costo (add-on vs prezzo ERCOT)")

            retail_addon_per_mwh = adder + ercot_admin + ancillary + securitization
            retail_total = retail_addon_per_mwh * total_mwh

            tdsp_addon_per_mwh = tdsp_usage_per_mwh + tdsp_flat_per_mwh + tdsp_peak_per_mwh
            tdsp_total = tdsp_addon_per_mwh * total_mwh

            tax_addon_per_mwh = taxes_per_mwh
            tax_total = tax_addon_per_mwh * total_mwh

            total_addon_per_mwh = estimated_total_per_mwh
            total_addon_total = total_addon_per_mwh * total_mwh

            invoice_per_mwh = avg_rate

            macro_rows = []

            macro_rows.append(
                {
                    "Macro area": "Componenti retail (Adder + Admin + Ancillary + Securitization)",
                    "Costo totale [USD]": retail_total,
                    "Costo medio [USD/MWh]": retail_addon_per_mwh,
                }
            )

            macro_rows.append(
                {
                    "Macro area": "TDSP (Usage + Flat + Peak)",
                    "Costo totale [USD]": tdsp_total,
                    "Costo medio [USD/MWh]": tdsp_addon_per_mwh,
                }
            )

            macro_rows.append(
                {
                    "Macro area": "Tasse su energia + TDSP",
                    "Costo totale [USD]": tax_total,
                    "Costo medio [USD/MWh]": tax_addon_per_mwh,
                }
            )

            macro_rows.append(
                {
                    "Macro area": "Totale add-on (no energia ERCOT)",
                    "Costo totale [USD]": total_addon_total,
                    "Costo medio [USD/MWh]": total_addon_per_mwh,
                }
            )

            macro_rows.append(
                {
                    "Macro area": "Energia ERCOT LZ_WEST (residua)",
                    "Costo totale [USD]": ercot_residual_total,
                    "Costo medio [USD/MWh]": ercot_residual_per_mwh,
                }
            )

            macro_rows.append(
                {
                    "Macro area": "Totale fattura (Synota)",
                    "Costo totale [USD]": total_cost,
                    "Costo medio [USD/MWh]": invoice_per_mwh,
                }
            )

            macro_df = pd.DataFrame(macro_rows)

            macro_df["Peso % su invoice"] = (
                macro_df["Costo totale [USD]"] / total_cost * 100.0
            )

            st.dataframe(
                macro_df.style.format(
                    {
                        "Costo totale [USD]": "{:,.2f}",
                        "Costo medio [USD/MWh]": "{:,.2f}",
                        "Peso % su invoice": "{:,.1f}%",
                    }
                ),
                use_container_width=True,
            )

            st.caption(
                "La riga **Energia ERCOT LZ_WEST (residua)** è calcolata come differenza tra la "
                "fattura reale Synota e la somma stimata di add-on retail/TDSP/tasse. "
                "Dividendo questo residuo per i MWh consumati otteniamo un **prezzo medio implicito ERCOT [USD/MWh]** "
                "per il periodo selezionato, utile per confrontarlo con i prezzi LZ_WEST dello storico ERCOT."
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

            # 🔹 Barre: ricavi (mining + hosting) vs costo energia
            if show_rev:
                # Ricavi mining – parte bassa della colonna "Ricavi"
                fig3.add_bar(
                    x=combined["date"],
                    y=combined["earnings_usd"],
                    name="Ricavi mining [USD]",
                    offsetgroup="ricavi",       # gruppo "ricavi" (colonna di sinistra)
                    legendgroup="ricavi",
                    marker_color="green",
                )

                # Ricavi hosting – parte alta della stessa colonna "Ricavi"
                fig3.add_bar(
                    x=combined["date"],
                    y=combined["hosting_revenue_usd"],
                    name="Ricavi hosting [USD]",
                    offsetgroup="ricavi",       # stesso gruppo → si stacka sulla colonna "ricavi"
                    legendgroup="ricavi",
                    marker_color="blue",
                )

            if show_cost:
                # Costo energia – colonna a sé affiancata ai ricavi
                fig3.add_bar(
                    x=combined["date"],
                    y=combined["invoice_amount_usd"],
                    name="Costo energia [USD]",
                    offsetgroup="costo",        # gruppo "costo" → seconda colonna
                    legendgroup="costo",
                    marker_color="red",
                )

            # 🔹 Linea: tariffa e BTC/day sull’asse destro
            if show_rate_mwh:
                fig3.add_scatter(
                    x=combined["date"],
                    y=combined["effective_rate_usd_per_mwh"],
                    name="Tariffa [USD/MWh]",
                    mode="lines+markers",
                    yaxis="y2",
                    marker_color="white",
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
                # relative/stack: mining + hosting si stackano,
                # ma il gruppo "ricavi" rimane affiancato al gruppo "costo"
                barmode="relative",
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

            # 👇 Assicuriamoci che daily_profit_usd sia numerica
            df_p["daily_profit_usd"] = pd.to_numeric(df_p["daily_profit_usd"], errors="coerce")

            df_p["profit_positive"] = df_p["daily_profit_usd"].apply(
                lambda x: x if pd.notna(x) and x > 0 else None
            )
            df_p["profit_negative"] = df_p["daily_profit_usd"].apply(
                lambda x: x if pd.notna(x) and x < 0 else None
            )

            # 👇 Rolling safe: se tutta NaN, rolling restituisce comunque NaN (nessun DataError)
            df_p["profit_roll7"] = df_p["daily_profit_usd"].rolling(7, min_periods=1).mean()
            df_p["profit_roll30"] = df_p["daily_profit_usd"].rolling(30, min_periods=1).mean()

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


# -----------------------------
# TAB 10 – PREZZI ERCOT LZ_WEST
# -----------------------------
with tab10:
    st.header("⚡ Prezzi ERCOT LZ_WEST – storico a 5 minuti")

    if ercot_filtered is None or ercot_filtered.empty:
        st.info("Carica il CSV dei prezzi ERCOT LZ_WEST nella sidebar per vedere il grafico.")
    else:
        df_erc = ercot_filtered.copy().sort_values("date")

        col1, col2 = st.columns(2)

        # Numero di giorni coperti dai dati filtrati
        num_days = df_erc["date"].dt.normalize().nunique()

        with col1:
            st.metric("Numero giorni coperti", f"{num_days}")

        with col2:
            st.metric(
                "Prezzo medio [USD/MWh]",
                f"{df_erc['lmp_usd_mwh'].mean():.2f}"
            )

        st.markdown("### Andamento LMP LZ_WEST (5 min)")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_erc["date"],
                y=df_erc["lmp_usd_mwh"],
                mode="lines",
                name="LMP LZ_WEST [USD/MWh]",
            )
        )

        fig.update_layout(
            xaxis_title="Data / ora",
            yaxis_title="Prezzo [USD/MWh]",
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="Ultime 24h", step="day", stepmode="backward"),
                        dict(count=7, label="Ultimi 7g", step="day", stepmode="backward"),
                        dict(step="all", label="Tutto"),
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")
        st.markdown("### 🧪 Simulatore cut-off prezzo ERCOT")

        st.write(
            "Imposta un prezzo soglia [USD/MWh]. "
            "Sopra questa soglia le macchine vengono spente; "
            "calcoliamo il prezzo medio quando sono accese e l'uptime risultante."
        )

        cutoff_price = st.number_input(
            "Prezzo soglia di spegnimento [USD/MWh]",
            min_value=0.0,
            max_value=10000.0,
            value=80.0,
            step=1.0,
        )

        # Intervalli in cui le macchine sarebbero ACCESE (prezzo <= soglia)
        df_on = df_erc[df_erc["lmp_usd_mwh"] <= cutoff_price]

        total_intervals = len(df_erc)
        on_intervals = len(df_on)

        if total_intervals == 0:
            st.warning("Non ci sono intervalli disponibili nel periodo selezionato.")
        elif on_intervals == 0:
            st.warning(
                "Con questa soglia di prezzo non ci sarebbe nessun intervallo in cui le macchine restano accese."
            )
        else:
            uptime_ratio = on_intervals / total_intervals
            uptime_pct = uptime_ratio * 100.0

            # Ogni intervallo è di 5 minuti → ore equivalenti
            minutes_per_interval = 5
            total_minutes_on = on_intervals * minutes_per_interval
            hours_on = total_minutes_on / 60.0
            days_on = hours_on / 24.0

            avg_price_on = df_on["lmp_usd_mwh"].mean()

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric(
                    "Uptime stimato [%]",
                    f"{uptime_pct:.1f}%",
                    help="Percentuale di tempo in cui il prezzo è ≤ soglia."
                )

            with col_b:
                st.metric(
                    "Prezzo medio quando ON [USD/MWh]",
                    f"{avg_price_on:.2f}",
                    help="Media LMP solo negli intervalli in cui le macchine restano accese."
                )

            with col_c:
                st.metric(
                    "Tempo ON (equivalente)",
                    f"{hours_on:.1f} h (~{days_on:.2f} gg)",
                    help="Ore/giorni equivalenti in cui le macchine sarebbero accese nel periodo selezionato."
                )

            # Piccolo riepilogo testuale
            st.caption(
                f"Nel periodo selezionato, con soglia **{cutoff_price:.2f} USD/MWh**, "
                f"le macchine sarebbero accese in **{on_intervals}** intervalli "
                f"su **{total_intervals}** totali."
            )

        # -------------------------------------------------
        # SIMULATORE INVERSO: TARGET "PREZZO MEDIO QUANDO ON"
        # -------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧪 Simulatore inverso: target prezzo medio quando ON")

        st.write(
            "Imposta un **prezzo medio obiettivo** quando le macchine sono ON. "
            "Calcoliamo quale soglia prezzo di spegnimento (cut-off) lo approssima meglio "
            "e che uptime ne risulta."
        )

        # Costruisco una tabella per soglie intere (1 USD/MWh) → media prezzo ON & uptime
        total_intervals_all = len(df_erc)
        minutes_per_interval = 5

        min_price = float(df_erc["lmp_usd_mwh"].min())
        max_price = float(df_erc["lmp_usd_mwh"].max())

        # Per sicurezza, limitiamo l'intervallo a qualcosa di sensato
        min_thr = int(min_price)
        max_thr = int(max_price)

        thresholds = list(range(min_thr, max_thr + 1))
        stats = []

        for thr in thresholds:
            df_on_thr = df_erc[df_erc["lmp_usd_mwh"] <= thr]
            on_int_thr = len(df_on_thr)

            if on_int_thr == 0:
                # Niente intervalli ON → saltiamo questa soglia
                continue

            avg_on_thr = df_on_thr["lmp_usd_mwh"].mean()
            uptime_thr = on_int_thr / total_intervals_all * 100.0
            hours_on_thr = on_int_thr * minutes_per_interval / 60.0

            stats.append(
                {
                    "threshold": thr,
                    "avg_price_on": avg_on_thr,
                    "uptime_pct": uptime_thr,
                    "hours_on": hours_on_thr,
                }
            )

        if len(stats) == 0:
            st.warning("Impossibile calcolare il simulatore inverso: nessun intervallo ON disponibile.")
        else:
            stats_df = pd.DataFrame(stats)

            min_avg_possible = float(stats_df["avg_price_on"].min())
            max_avg_possible = float(stats_df["avg_price_on"].max())
            default_target = float(stats_df["avg_price_on"].median())

            target_avg = st.number_input(
                "Target 'Prezzo medio quando ON' [USD/MWh]",
                min_value=round(min_avg_possible, 2),
                max_value=round(max_avg_possible, 2),
                value=round(default_target, 2),
                step=1.0,
            )

            # Trova la soglia che produce una media più vicina al target
            closest_idx = (stats_df["avg_price_on"] - target_avg).abs().argmin()
            closest_row = stats_df.iloc[closest_idx]

            best_thr = closest_row["threshold"]
            best_avg = closest_row["avg_price_on"]
            best_uptime = closest_row["uptime_pct"]
            best_hours = closest_row["hours_on"]
            best_days = best_hours / 24.0

            col_inv_a, col_inv_b, col_inv_c, col_inv_d = st.columns(4)
            with col_inv_a:
                st.metric(
                    "Soglia equivalente [USD/MWh]",
                    f"{best_thr:.0f}",
                    help="Cut-off prezzo che approssima il target medio ON."
                )
            with col_inv_b:
                st.metric(
                    "Prezzo medio risultante [USD/MWh]",
                    f"{best_avg:.2f}",
                    help="Prezzo medio effettivo con quella soglia."
                )
            with col_inv_c:
                st.metric(
                    "Uptime stimato [%]",
                    f"{best_uptime:.1f}%",
                    help="Percentuale di intervalli ON con quella soglia."
                )
            with col_inv_d:
                st.metric(
                    "Tempo ON equivalente",
                    f"{best_hours:.1f} h (~{best_days:.2f} gg)",
                    help="Tempo complessivo ON nel periodo selezionato."
                )

            st.caption(
                f"Il target di **{target_avg:.2f} USD/MWh** corrisponde circa a una soglia di "
                f"**{best_thr:.0f} USD/MWh**, con uptime ≈ **{best_uptime:.1f}%** "
                f"nel periodo attuale."
            )

            st.markdown("### 📈 Sensibilità: effetto del cut-off sul prezzo medio e sull'uptime")

            st.write(
                "Per ogni soglia di spegnimento (step 1 USD/MWh) mostriamo come cambiano "
                "il **prezzo medio quando ON** e l'**uptime**. "
                "Questo fa vedere, ad esempio, che passare da 95→90 USD non ha lo stesso effetto che 50→45 USD."
            )

            fig_cut = go.Figure()

            # Linea: prezzo medio quando ON
            fig_cut.add_trace(
                go.Scatter(
                    x=stats_df["threshold"],
                    y=stats_df["avg_price_on"],
                    mode="lines+markers",
                    name="Prezzo medio quando ON [USD/MWh]",
                )
            )

            # Secondo asse: uptime %
            fig_cut.add_trace(
                go.Scatter(
                    x=stats_df["threshold"],
                    y=stats_df["uptime_pct"],
                    mode="lines",
                    name="Uptime [%]",
                    yaxis="y2",
                )
            )

            fig_cut.update_layout(
                xaxis_title="Soglia prezzo di spegnimento [USD/MWh]",
                yaxis=dict(
                    title="Prezzo medio quando ON [USD/MWh]",
                ),
                yaxis2=dict(
                    title="Uptime [%]",
                    overlaying="y",
                    side="right",
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig_cut, use_container_width=True)

            with st.expander("Tabella dettagli per soglia (step 1 USD/MWh)"):
                st.dataframe(
                    stats_df.rename(
                        columns={
                            "threshold": "Soglia [USD/MWh]",
                            "avg_price_on": "Prezzo medio ON [USD/MWh]",
                            "uptime_pct": "Uptime [%]",
                            "hours_on": "Ore ON equivalenti",
                        }
                    ).style.format(
                        {
                            "Prezzo medio ON [USD/MWh]": "{:,.2f}",
                            "Uptime [%]": "{:,.1f}",
                            "Ore ON equivalenti": "{:,.1f}",
                        }
                    ),
                    use_container_width=True,
                )

# -----------------------------
# TAB 11 – ENERGIA PROMETHEUS (DETTAGLIO)
# -----------------------------
with tab11:
    st.header("⚡ Energia – Prometheus (daily usage)")

    if prometheus_filtered is None or prometheus_filtered.empty:
        st.info("Carica il file Prometheus nella sidebar e seleziona un timeframe valido.")
    else:
        dfp = prometheus_filtered.copy().sort_values("date")

        # Metriche base
        total_mwh = dfp["energy_mwh"].sum()
        num_days = dfp["date"].dt.normalize().nunique()
        avg_daily_mwh = total_mwh / num_days if num_days > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Energia totale [MWh]", f"{total_mwh:,.2f}")
        with col2:
            st.metric("Numero giorni", num_days)
        with col3:
            st.metric("Media giornaliera [MWh]", f"{avg_daily_mwh:,.2f}")

        st.markdown("### 📊 Energia giornaliera (Prometheus)")

        fig_p = go.Figure()
        fig_p.add_bar(
            x=dfp["date"],
            y=dfp["energy_mwh"],
            name="Energia [MWh]",
        )
        fig_p.update_layout(
            xaxis_title="Data",
            yaxis_title="Energia [MWh]",
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # -----------------------------------
        # INCROCIO CON ANTPPOOL: RICAVI / MWh
        # -----------------------------------
        st.markdown("### 💰 Ricavi mining per MWh (Prometheus vs Antpool)")

        if antpool_filtered is None or antpool_filtered.empty:
            st.info("Carica anche il CSV Antpool per vedere il confronto ricavi / MWh.")
        else:
            # Copia locale di Antpool
            ant_df = antpool_filtered.copy()

            # Applica prezzo BTC (stessa logica del tab Antpool)
            if btc_price_df is not None and not btc_price_df.empty:
                ant_df = ant_df.merge(btc_price_df, on="date", how="left")
                fallback_price = btc_price_used if btc_price_used else 0.0
                ant_df["btc_price_usd"] = ant_df["btc_price_usd"].fillna(fallback_price)
            else:
                fallback_price = btc_price_used if btc_price_used else 0.0
                ant_df["btc_price_usd"] = fallback_price

            # Ricavi in USD per giorno
            ant_df["earnings_usd"] = (
                ant_df["total_earnings_btc"] * ant_df["btc_price_usd"]
            )

            # Merge con Prometheus (energia MWh)
            merged = pd.merge(
                dfp[["date", "energy_mwh"]],
                ant_df[
                    [
                        "date",
                        "earnings_usd",
                        "total_earnings_btc",
                    ]
                ],
                on="date",
                how="inner",
            )

            if merged.empty:
                st.info("Nessun giorno in comune tra Prometheus e Antpool nel periodo selezionato.")
            else:
                merged["usd_per_mwh"] = merged["earnings_usd"] / merged["energy_mwh"]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Ricavo medio [USD/MWh]",
                        f"{merged['usd_per_mwh'].mean():.2f}",
                    )
                with col_b:
                    st.metric(
                        "Giorni con dati comuni",
                        len(merged),
                    )


                fig_rev = go.Figure()
                fig_rev.add_bar(
                    x=merged["date"],
                    y=merged["usd_per_mwh"],
                    name="Ricavi mining [USD/MWh]",
                )
                fig_rev.update_layout(
                    xaxis_title="Data",
                    yaxis_title="Ricavi mining [USD/MWh]",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_rev, use_container_width=True)

                st.markdown("#### 📄 Dettaglio giornaliero")
                st.dataframe(
                    merged[
                        [
                            "date",
                            "energy_mwh",
                            "earnings_usd",
                            "total_earnings_btc",
                            "usd_per_mwh",
                        ]
                    ].rename(
                        columns={
                            "date": "Date",
                            "energy_mwh": "Energia [MWh]",
                            "earnings_usd": "Ricavi mining [USD]",
                            "total_earnings_btc": "Ricavi mining [BTC]",
                            "usd_per_mwh": "Ricavi [USD/MWh]",
                        }
                    ).style.format(
                        {
                            "Energia [MWh]": "{:,.2f}",
                            "Ricavi mining [USD]": "{:,.2f}",
                            "Ricavi mining [BTC]": "{:,.6f}",
                            "Ricavi [USD/MWh]": "{:,.2f}",
                        }
                    ),
                    use_container_width=True,
                )



# -----------------------------
# TAB 12 – PREZZI RTM (PROMETHEUS EXTRACT)
# -----------------------------
with tab12:
    st.header("⚡ RTM Prices – Zones/Hubs (Prometheus extract)")

    if rtm_prices_filtered is None or rtm_prices_filtered.empty:
        st.info("Carica l'Excel RTM Price Extract nella sidebar e seleziona un timeframe valido.")
    else:
        df_rtm = rtm_prices_filtered.copy().sort_values("date")

        # Selezione zona/settlement
        df_rtm["pair"] = df_rtm["LoadZone"].astype(str) + " " + df_rtm["Settlement"].astype(str)
        pairs = sorted(df_rtm["pair"].dropna().unique().tolist())

        selected_pair = st.selectbox(
            "Seleziona LoadZone / Settlement",
            options=pairs,
            index=pairs.index("WEST LZ") if "WEST LZ" in pairs else 0,
        )

        lz, sett = selected_pair.split(" ", 1)
        dfp = df_rtm[(df_rtm["LoadZone"] == lz) & (df_rtm["Settlement"] == sett)].copy()

        if dfp.empty:
            st.warning("Nessun record per la selezione effettuata nel range scelto.")
            st.stop()

        # -----------------------------
        # SEZIONE 1: Stima costo usando AVG giornaliero
        # -----------------------------
        st.markdown("### 1) 💡 Stima costo usando la **media giornaliera (Avg)**")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            kw_on = st.number_input(
                "Consumo medio quando ON [kW] (costante)",
                min_value=0.0,
                value=1000.0,
                step=10.0,
            )
        with col_b:
            st.caption("Energia ON giornaliera")
            st.code(f"{kw_on:.1f} kW × 24 h / 1000 = {kw_on*24/1000:.3f} MWh/giorno")
        with col_c:
            st.caption("Nota")
            st.write("I prezzi sono in **USD/MWh**.")

        dfp["energy_mwh_on"] = (kw_on * 24.0) / 1000.0
        dfp["cost_usd_on_avg"] = dfp["Avg"] * dfp["energy_mwh_on"]

        total_days = int(dfp["date"].dt.normalize().nunique())
        total_energy = float(dfp["energy_mwh_on"].sum())
        total_cost = float(dfp["cost_usd_on_avg"].sum())
        avg_price_weighted = (total_cost / total_energy) if total_energy > 0 else np.nan

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Giorni", f"{total_days}")
        with m2:
            st.metric("Energia totale [MWh]", f"{total_energy:,.2f}")
        with m3:
            st.metric("Costo totale [USD]", f"{total_cost:,.2f}")
        with m4:
            st.metric("Prezzo medio ponderato [USD/MWh]", f"{avg_price_weighted:,.2f}" if np.isfinite(avg_price_weighted) else "N/A")

        # Tabelle daily / monthly
        df_daily = dfp[["date", "Avg", "energy_mwh_on", "cost_usd_on_avg"]].copy()
        df_daily = df_daily.rename(
            columns={
                "Avg": "avg_usd_mwh",
                "energy_mwh_on": "energy_mwh",
                "cost_usd_on_avg": "cost_usd",
            }
        )
        df_daily["effective_usd_mwh"] = df_daily["cost_usd"] / df_daily["energy_mwh"]

        df_daily["month"] = df_daily["date"].dt.to_period("M").dt.to_timestamp()
        df_month = (
            df_daily.groupby("month", as_index=False)
            .agg(
                cost_usd=("cost_usd", "sum"),
                energy_mwh=("energy_mwh", "sum"),
                avg_usd_mwh=("avg_usd_mwh", "mean"),
            )
            .sort_values("month")
        )
        df_month["effective_usd_mwh"] = df_month["cost_usd"] / df_month["energy_mwh"]
        df_month["days"] = df_daily.groupby("month")["date"].nunique().values

        # Grafici
        st.markdown("#### 📈 Grafici (Avg giornaliero)")
        c1, c2 = st.columns(2)

        with c1:
            fig_cost_d = go.Figure()
            fig_cost_d.add_bar(x=df_daily["date"], y=df_daily["cost_usd"], name="Costo giornaliero [USD]")
            fig_cost_d.update_layout(xaxis_title="Data", yaxis_title="Costo [USD]", hovermode="x unified")
            st.plotly_chart(fig_cost_d, use_container_width=True)

        with c2:
            fig_price_d = go.Figure()
            fig_price_d.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["avg_usd_mwh"], mode="lines", name="Avg [USD/MWh]"))
            fig_price_d.update_layout(xaxis_title="Data", yaxis_title="Prezzo [USD/MWh]", hovermode="x unified")
            st.plotly_chart(fig_price_d, use_container_width=True)

        fig_month = go.Figure()
        fig_month.add_bar(x=df_month["month"], y=df_month["cost_usd"], name="Costo mensile [USD]")
        fig_month.update_layout(xaxis_title="Mese", yaxis_title="Costo [USD]", hovermode="x unified")
        st.plotly_chart(fig_month, use_container_width=True)

        with st.expander("📋 Tabelle (Avg giornaliero)"):
            st.markdown("**Daily**")
            st.dataframe(
                df_daily.sort_values("date").style.format(
                    {
                        "avg_usd_mwh": "{:,.2f}",
                        "energy_mwh": "{:,.3f}",
                        "cost_usd": "{:,.2f}",
                        "effective_usd_mwh": "{:,.2f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("**Monthly**")
            st.dataframe(
                df_month.sort_values("month").style.format(
                    {
                        "cost_usd": "{:,.2f}",
                        "energy_mwh": "{:,.2f}",
                        "avg_usd_mwh": "{:,.2f}",
                        "effective_usd_mwh": "{:,.2f}",
                    }
                ),
                use_container_width=True,
            )

        st.markdown("---")

        # -----------------------------
        # SEZIONE 2: Curtailment con prezzi 15-min (1..100)
        # -----------------------------
        st.markdown("### 2) 🧪 Simulatore spegnimento ASIC (prezzi **15-min**)")

        st.write(
            "Usiamo i prezzi 15-min (colonne 1..100). "
            "Per ogni intervallo: se **prezzo > soglia** → OFF (sleep); altrimenti ON. "
            "Il file usa **Intervals** per gestire giorni da 92/96/100 intervalli (DST)."
        )

        colx1, colx2, colx3 = st.columns(3)
        with colx1:
            thr = st.number_input(
                "Soglia spegnimento [USD/MWh] (OFF se > soglia)",
                min_value=0.0,
                value=100.0,
                step=1.0,
            )
        with colx2:
            kw_sleep = st.number_input(
                "Consumo stimato a SLEEP [kW]",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )
        with colx3:
            st.caption("Consumo ON")
            st.write(f"Usiamo **{kw_on:.1f} kW** (lo stesso sopra).")

        # Colonne intervalli disponibili
        interval_cols = [c for c in dfp.columns if isinstance(c, int)]
        interval_cols = [c for c in interval_cols if 1 <= c <= 100]
        interval_cols = sorted(interval_cols)

        prices = dfp[interval_cols].to_numpy(dtype=float)
        intervals = dfp["Intervals"].fillna(96).astype(int).to_numpy()

        # mask valid per DST (es: 92/96/100)
        idxs = np.arange(1, len(interval_cols) + 1)
        valid_mask = idxs[None, :] <= intervals[:, None]

        on_mask = (prices <= thr) & valid_mask

        kw_matrix = np.where(on_mask, kw_on, kw_sleep)
        mwh_matrix = (kw_matrix * 0.25) / 1000.0  # 15 min = 0.25h

        # zero out invalid intervals
        mwh_matrix = np.where(valid_mask, mwh_matrix, 0.0)
        cost_matrix = np.where(valid_mask, prices * mwh_matrix, 0.0)

        df_daily2 = pd.DataFrame(
            {
                "date": dfp["date"].values,
                "cost_usd": cost_matrix.sum(axis=1),
                "energy_mwh": mwh_matrix.sum(axis=1),
                "uptime_pct": (on_mask.sum(axis=1) / valid_mask.sum(axis=1)) * 100.0,
                "hours_on": on_mask.sum(axis=1) * 0.25,
            }
        )
        df_daily2["effective_usd_mwh"] = df_daily2["cost_usd"] / df_daily2["energy_mwh"]
        df_daily2["month"] = pd.to_datetime(df_daily2["date"]).dt.to_period("M").dt.to_timestamp()

        total_cost2 = float(df_daily2["cost_usd"].sum())
        total_energy2 = float(df_daily2["energy_mwh"].sum())
        avg_eff2 = (total_cost2 / total_energy2) if total_energy2 > 0 else np.nan
        avg_uptime2 = float(df_daily2["uptime_pct"].mean())

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            st.metric("Costo totale [USD]", f"{total_cost2:,.2f}")
        with cc2:
            st.metric("Energia totale [MWh]", f"{total_energy2:,.2f}")
        with cc3:
            st.metric("Prezzo medio ponderato [USD/MWh]", f"{avg_eff2:,.2f}" if np.isfinite(avg_eff2) else "N/A")
        with cc4:
            st.metric("Uptime medio [%]", f"{avg_uptime2:,.1f}")

        df_month2 = (
            df_daily2.groupby("month", as_index=False)
            .agg(
                cost_usd=("cost_usd", "sum"),
                energy_mwh=("energy_mwh", "sum"),
                uptime_pct=("uptime_pct", "mean"),
                hours_on=("hours_on", "sum"),
            )
            .sort_values("month")
        )
        df_month2["effective_usd_mwh"] = df_month2["cost_usd"] / df_month2["energy_mwh"]

        st.markdown("#### 📈 Grafici (15-min con cut-off)")
        g1, g2 = st.columns(2)

        with g1:
            fig_d2 = go.Figure()
            fig_d2.add_bar(x=df_daily2["date"], y=df_daily2["cost_usd"], name="Costo giornaliero [USD]")
            fig_d2.update_layout(xaxis_title="Data", yaxis_title="Costo [USD]", hovermode="x unified")
            st.plotly_chart(fig_d2, use_container_width=True)

        with g2:
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(x=df_daily2["date"], y=df_daily2["effective_usd_mwh"], mode="lines", name="Prezzo effettivo [USD/MWh]"))
            fig_eff.add_trace(go.Scatter(x=df_daily2["date"], y=df_daily2["uptime_pct"], mode="lines", name="Uptime [%]", yaxis="y2"))
            fig_eff.update_layout(
                xaxis_title="Data",
                yaxis=dict(title="Prezzo [USD/MWh]"),
                yaxis2=dict(title="Uptime [%]", overlaying="y", side="right"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        fig_m2 = go.Figure()
        fig_m2.add_bar(x=df_month2["month"], y=df_month2["cost_usd"], name="Costo mensile [USD]")
        fig_m2.update_layout(xaxis_title="Mese", yaxis_title="Costo [USD]", hovermode="x unified")
        st.plotly_chart(fig_m2, use_container_width=True)

        with st.expander("📋 Tabelle (15-min con cut-off)"):
            st.markdown("**Daily**")
            st.dataframe(
                df_daily2.sort_values("date").style.format(
                    {
                        "cost_usd": "{:,.2f}",
                        "energy_mwh": "{:,.3f}",
                        "effective_usd_mwh": "{:,.2f}",
                        "uptime_pct": "{:,.1f}",
                        "hours_on": "{:,.1f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("**Monthly**")
            st.dataframe(
                df_month2.sort_values("month").style.format(
                    {
                        "cost_usd": "{:,.2f}",
                        "energy_mwh": "{:,.2f}",
                        "effective_usd_mwh": "{:,.2f}",
                        "uptime_pct": "{:,.1f}",
                        "hours_on": "{:,.1f}",
                    }
                ),
                use_container_width=True,
            )

        st.markdown("---")
        # -----------------------------
        # SEZIONE 3: Forecast futuro (storico RTM 15-min + add-on calcolato da picco)
        # -----------------------------
        st.markdown("### 3) 🔮 Forecast prezzi futuri (storico **15-min** + add-on)")

        if rtm_prices_df is None or rtm_prices_df.empty:
            st.info("Carica l'Excel RTM Price Extract (xlsx) nella sidebar per usare il forecast.")
        else:
            # Usiamo il DF completo (non filtrato dal date range globale),
            # così possiamo stimare mesi futuri usando lo storico disponibile.
            df_all = rtm_prices_df.copy().sort_values("date")

            # Manteniamo la stessa selezione LoadZone/Settlement della tab
            df_all["pair"] = df_all["LoadZone"].astype(str) + " " + df_all["Settlement"].astype(str)
            df_all = df_all[df_all["pair"] == selected_pair].copy()

            if df_all.empty:
                st.warning("Nessun dato storico disponibile per la coppia selezionata (LoadZone/Settlement).")
                st.stop()

           # -----------------------------
            # 3A) Add-on + modalità Uptime/Cut-off
            # -----------------------------
            st.markdown("#### 3A) 🧱 Add-on + Uptime / Cut-off (ON/OFF)")

            cA1, cA2, cA3, cA4 = st.columns(4)
            with cA1:
                peak_kwh_month = st.number_input(
                    "kWh di picco del mese (es. 720)",
                    min_value=0.0,
                    value=720.0,
                    step=1.0,
                )
            with cA2:
                peak_multiplier_usd = st.number_input(
                    "Valore da moltiplicare (es. 11$)",
                    min_value=0.0,
                    value=11.0,
                    step=0.5,
                )
            with cA3:
                days_assumed = st.number_input(
                    "Giorni mese (assunzione)",
                    min_value=1,
                    value=30,
                    step=1,
                )
            with cA4:
                st.caption("Ore/giorno")
                st.code("24")

            total_usd_peak = peak_kwh_month * peak_multiplier_usd
            st.metric("Totale USD (picco×moltiplicatore)", f"{total_usd_peak:,.2f}")

            mode = st.radio(
                "Modalità controllo (equivalenti):",
                ["Cut-off soglia [USD/MWh]", "Uptime medio [%]"],
                horizontal=True,
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                off_threshold_usd_mwh = st.number_input(
                    "Soglia spegnimento [USD/MWh] (OFF se > soglia)",
                    min_value=0.0,
                    value=80.0,
                    step=5.0,
                    disabled=(mode != "Cut-off soglia [USD/MWh]"),
                )
            with col_m2:
                uptime_pct_input = st.number_input(
                    "Uptime medio [%]",
                    min_value=0.0,
                    max_value=100.0,
                    value=90.0,
                    step=1.0,
                    disabled=(mode != "Uptime medio [%]"),
                    help="Interpreto: ON per la percentuale di intervalli più economici (cut-off implicito).",
                )

            # -----------------------------
            # 3B) Mese target + storico coerente
            # -----------------------------
            st.markdown("#### 3B) 🗓️ Mese target + storico coerente")

            target_date = st.date_input(
                "Seleziona un giorno del mese futuro (useremo solo mese/anno)",
                value=datetime(datetime.now().year + 1, 1, 1).date(),
            )
            target_month = target_date.month
            target_year = target_date.year

            df_all["year"] = df_all["date"].dt.year
            df_all["month"] = df_all["date"].dt.month

            years_available = sorted(df_all.loc[df_all["month"] == target_month, "year"].dropna().unique().tolist())
            if not years_available:
                st.warning("Non ci sono anni disponibili nello storico per il mese selezionato.")
                st.stop()

            # Evita anni futuri se per caso presenti
            years_available = [y for y in years_available if y < target_year] or years_available

            max_years = len(years_available)
            n_years = st.slider(
                "Quanti anni storici usare (stesso mese)?",
                min_value=1,
                max_value=max_years,
                value=min(3, max_years),
            )

            years_used = sorted(years_available[-n_years:])

            st.write(f"**Mese target:** {target_year}-{target_month:02d}")
            st.write(f"**Anni usati per lo storico:** {', '.join(map(str, years_used))}")

            # -----------------------------
            # 3C) Calcolo stima “ON only” + uptime + add-on uptime-aware
            # -----------------------------
            st.markdown("#### 3C) 📊 Stima (solo intervalli ON) + uptime + add-on")

            interval_cols = [c for c in df_all.columns if isinstance(c, int)]
            interval_cols = sorted([c for c in interval_cols if 1 <= c <= 100])
            if not interval_cols:
                st.warning("Non trovo le colonne 15-min (1..100). Controlla il parsing dell'xlsx.")
                st.stop()

            rows = []
            for y in years_used:
                dfx = df_all[(df_all["year"] == y) & (df_all["month"] == target_month)].copy()
                if dfx.empty:
                    continue

                prices = dfx[interval_cols].to_numpy(dtype=float)
                intervals = dfx["Intervals"].fillna(96).astype(int).to_numpy()

                idxs = np.arange(1, len(interval_cols) + 1)
                valid_mask = idxs[None, :] <= intervals[:, None]

                valid_prices = prices[valid_mask]
                valid_prices = valid_prices[np.isfinite(valid_prices)]

                if valid_prices.size == 0:
                    continue

                # Determina ON-mask in 2 modi equivalenti
                if mode == "Cut-off soglia [USD/MWh]":
                    threshold_used = float(off_threshold_usd_mwh)
                    on_prices = valid_prices[valid_prices <= threshold_used]
                    uptime_frac = float(on_prices.size / valid_prices.size) if valid_prices.size > 0 else 0.0
                else:
                    uptime_frac = float(uptime_pct_input) / 100.0
                    uptime_frac = max(0.0, min(1.0, uptime_frac))
                    # ON = prendo la quota più economica -> cut-off implicito = percentile uptime%
                    threshold_used = float(np.quantile(valid_prices, uptime_frac)) if uptime_frac > 0 else 0.0
                    on_prices = valid_prices[valid_prices <= threshold_used]

                    # (Per robustezza: se tanti valori uguali al cut-off possono sballare l’esatto %, ma va bene)
                    uptime_frac = float(on_prices.size / valid_prices.size) if valid_prices.size > 0 else uptime_frac

                hist_avg_all = float(valid_prices.mean())
                hist_avg_on = float(on_prices.mean()) if on_prices.size > 0 else np.nan

                # kWh presunti "effettivi" = base * uptime
                presumed_kwh_eff = peak_kwh_month * float(days_assumed) * 24.0 * uptime_frac
                addon_usd_per_kwh = (total_usd_peak / presumed_kwh_eff) if presumed_kwh_eff > 0 else 0.0
                addon_usd_per_mwh = addon_usd_per_kwh * 1000.0

                est_allin_on = (hist_avg_on + addon_usd_per_mwh) if np.isfinite(hist_avg_on) else np.nan

                rows.append(
                    {
                        "year": y,
                        "uptime_pct": uptime_frac * 100.0,
                        "cutoff_used_usd_mwh": threshold_used,
                        "hist_avg_all_usd_mwh": hist_avg_all,
                        "hist_avg_on_usd_mwh": hist_avg_on,
                        "addon_usd_mwh": addon_usd_per_mwh,
                        "est_allin_on_usd_mwh": est_allin_on,
                        "days_in_month_data": int(dfx["date"].dt.normalize().nunique()),
                    }
                )

            if not rows:
                st.warning("Nessun dato calcolabile per gli anni selezionati.")
                st.stop()

            forecast_df = pd.DataFrame(rows).sort_values("year")

            # -----------------------------
            # Trend YoY: "Prezzo di tendenza" (su storico del mese)
            # -----------------------------
            # Usiamo il prezzo storico ON-only come base trend (coerente col cut-off/uptime)
            trend_base_col = "hist_avg_on_usd_mwh"

            trend_df = forecast_df[["year", trend_base_col]].dropna().sort_values("year").copy()
            trend_df["yoy"] = trend_df[trend_base_col].pct_change()  # (P_t / P_{t-1}) - 1

            # Media YoY sugli ultimi anni disponibili (dentro years_used)
            yoy_values = trend_df["yoy"].dropna()

            trend_yoy_avg = float(yoy_values.mean()) if len(yoy_values) > 0 else 0.0

            # Ultimo prezzo disponibile (es. 2025)
            last_year = int(trend_df["year"].max())
            last_price = float(trend_df.loc[trend_df["year"] == last_year, trend_base_col].iloc[0])

            trend_price = last_price * (1.0 + trend_yoy_avg)

            # Trend ALL-IN (opzionale ma molto utile): aggiungo anche l'add-on medio
            addon_avg = float(forecast_df["addon_usd_mwh"].mean())
            trend_price_allin = trend_price + addon_avg


            if mode == "Uptime medio [%]":
                avg_cutoff = float(forecast_df["cutoff_used_usd_mwh"].mean())
                p10_cutoff = float(forecast_df["cutoff_used_usd_mwh"].quantile(0.10))
                p90_cutoff = float(forecast_df["cutoff_used_usd_mwh"].quantile(0.90))

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Cut-off implicito medio [USD/MWh]", f"{avg_cutoff:,.2f}")
                with c2:
                    st.metric("Cut-off implicito P10 [USD/MWh]", f"{p10_cutoff:,.2f}")
                with c3:
                    st.metric("Cut-off implicito P90 [USD/MWh]", f"{p90_cutoff:,.2f}")

                st.caption("Cut-off implicito = percentile dei prezzi RTM 15-min coerente con l’uptime inserito.")

            # Metriche riassuntive
            mean_uptime = float(forecast_df["uptime_pct"].mean())
            mean_on = float(forecast_df["hist_avg_on_usd_mwh"].mean())
            mean_allin = float(forecast_df["est_allin_on_usd_mwh"].mean())
            min_allin = float(forecast_df["est_allin_on_usd_mwh"].min())
            max_allin = float(forecast_df["est_allin_on_usd_mwh"].max())

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Uptime medio [%]", f"{mean_uptime:,.1f}")
            with m2:
                st.metric("Storico ON avg [USD/MWh]", f"{mean_on:,.2f}")
            with m3:
                st.metric("ALL-IN ON avg [USD/MWh]", f"{mean_allin:,.2f}")
            with m4:
                st.metric("ALL-IN min [USD/MWh]", f"{min_allin:,.2f}")
            with m5:
                st.metric("ALL-IN max [USD/MWh]", f"{max_allin:,.2f}")


            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("Trend YoY medio [%]", f"{trend_yoy_avg*100:,.2f}%")
            with t2:
                st.metric(f"Prezzo di tendenza (ON) [{last_year}→{target_year}] [USD/MWh]", f"{trend_price:,.2f}")
            with t3:
                st.metric(f"Tendenza ALL-IN (ON) [USD/MWh]", f"{trend_price_allin:,.2f}")


            with st.expander("📋 Dettaglio forecast (per anno)"):
                st.dataframe(
                    forecast_df.rename(
                        columns={
                            "year": "Year",
                            "uptime_pct": "Uptime [%]",
                            "cutoff_used_usd_mwh": "Cut-off used [USD/MWh]",
                            "hist_avg_all_usd_mwh": "Historical avg ALL (15-min) [USD/MWh]",
                            "hist_avg_on_usd_mwh": "Historical avg ON only [USD/MWh]",
                            "addon_usd_mwh": "Add-on (uptime-aware) [USD/MWh]",
                            "est_allin_on_usd_mwh": "Estimated ALL-IN (ON) [USD/MWh]",
                            "days_in_month_data": "Days covered",
                        }
                    ).style.format(
                        {
                            "Uptime [%]": "{:,.1f}",
                            "Cut-off used [USD/MWh]": "{:,.2f}",
                            "Historical avg ALL (15-min) [USD/MWh]": "{:,.2f}",
                            "Historical avg ON only [USD/MWh]": "{:,.2f}",
                            "Add-on (uptime-aware) [USD/MWh]": "{:,.2f}",
                            "Estimated ALL-IN (ON) [USD/MWh]": "{:,.2f}",
                        }
                    ),
                    use_container_width=True,
                )


            # -----------------------------
            # 3D) 📈 Grafico forecast
            # -----------------------------
            st.markdown("#### 3D) 📈 Grafico forecast")

            fig_fore = go.Figure()

            fig_fore.add_bar(
                x=forecast_df["year"],
                y=forecast_df["hist_avg_on_usd_mwh"],
                name="Storico ON only [USD/MWh]",
            )

            fig_fore.add_scatter(
                x=forecast_df["year"],
                y=forecast_df["est_allin_on_usd_mwh"],
                mode="lines+markers",
                name="ALL-IN (ON) [USD/MWh]",
            )

            # Punto Trend
            fig_fore.add_scatter(
                x=[target_year],
                y=[trend_price_allin],
                mode="markers+text",
                text=["Trend"],
                textposition="top center",
                name="Trend (ALL-IN) – proiezione",
            )

            fig_fore.update_layout(
                xaxis_title="Anno",
                yaxis_title="Prezzo [USD/MWh]",
                hovermode="x unified",
            )

            st.plotly_chart(fig_fore, use_container_width=True)

            st.markdown("---")
            st.markdown("### 3E) 🗓️ Timeline mensile (T0 al centro) – All-in ON (storico / stimato / tendenza)")

            # Quanti mesi a sinistra/destra
            colw1, colw2 = st.columns(2)
            with colw1:
                months_back = st.slider("Mesi precedenti (a sinistra)", 1, 24, 6)
            with colw2:
                months_fwd = st.slider("Mesi futuri (a destra)", 1, 24, 6)

            # Helper: lista mesi centrata su T0 (mese corrente)
            today = datetime.now()
            t0_y, t0_m = today.year, today.month

            def add_months(y, m, delta):
                # delta può essere negativo/positivo
                m2 = m + delta
                y2 = y + (m2 - 1) // 12
                m2 = (m2 - 1) % 12 + 1
                return y2, m2

            months_axis = []
            for d in range(-months_back, months_fwd + 1):
                yy, mm = add_months(t0_y, t0_m, d)
                months_axis.append((yy, mm))

            def month_label(yy, mm):
                return f"{yy}-{mm:02d}"

            # Helper: calcola ON-only avg per un dataset di prezzi 15-min (array) + uptime_frac secondo mode
            def compute_on_stats(valid_prices: np.ndarray, mode: str, off_thr: float, uptime_pct: float):
                valid_prices = valid_prices[np.isfinite(valid_prices)]
                if valid_prices.size == 0:
                    return np.nan, 0.0, np.nan  # (avg_on, uptime_frac, cutoff_used)

                if mode == "Cut-off soglia [USD/MWh]":
                    cutoff_used = float(off_thr)
                    on_prices = valid_prices[valid_prices <= cutoff_used]
                    uptime_frac = float(on_prices.size / valid_prices.size)
                else:
                    uptime_frac = max(0.0, min(1.0, float(uptime_pct) / 100.0))
                    cutoff_used = float(np.quantile(valid_prices, uptime_frac)) if uptime_frac > 0 else 0.0
                    on_prices = valid_prices[valid_prices <= cutoff_used]
                    uptime_frac = float(on_prices.size / valid_prices.size) if valid_prices.size > 0 else uptime_frac

                avg_on = float(on_prices.mean()) if on_prices.size > 0 else np.nan
                return avg_on, uptime_frac, cutoff_used

            # Helper: add-on uptime-aware (USD/MWh) usando giorni REALI del mese
            def addon_usd_mwh_for_month(yy, mm, uptime_frac):
                days_real = calendar.monthrange(yy, mm)[1]
                presumed_kwh_eff = peak_kwh_month * float(days_real) * 24.0 * float(uptime_frac)
                total_usd_peak_local = peak_kwh_month * peak_multiplier_usd
                addon_usd_per_kwh = (total_usd_peak_local / presumed_kwh_eff) if presumed_kwh_eff > 0 else 0.0
                return addon_usd_per_kwh * 1000.0

            # Prepariamo dati base (serve interval_cols)
            interval_cols = [c for c in df_all.columns if isinstance(c, int)]
            interval_cols = sorted([c for c in interval_cols if 1 <= c <= 100])
            if not interval_cols:
                st.warning("Non trovo le colonne 15-min (1..100).")
                st.stop()

            # Anni disponibili per ogni mese (per stimato/tendenza)
            df_all["year"] = df_all["date"].dt.year
            df_all["month"] = df_all["date"].dt.month

            def monthly_valid_prices_for_year(yy, mm):
                dfx = df_all[(df_all["year"] == yy) & (df_all["month"] == mm)].copy()
                if dfx.empty:
                    return np.array([], dtype=float)

                prices = dfx[interval_cols].to_numpy(dtype=float)
                intervals = dfx["Intervals"].fillna(96).astype(int).to_numpy()

                idxs = np.arange(1, len(interval_cols) + 1)
                valid_mask = idxs[None, :] <= intervals[:, None]

                valid_prices = prices[valid_mask]
                valid_prices = valid_prices[np.isfinite(valid_prices)]
                return valid_prices

            # 1) Costruiamo serie "reale" per mesi passati (se dati presenti)
            # 2) Serie "stimato" per mesi futuri (media storico ON-only sugli ultimi N anni disponibili)
            # 3) Serie "tendenza" per mesi futuri (proiezione YoY sul mese)

            actual_allin = []
            est_allin = []
            trend_allin = []

            for (yy, mm) in months_axis:
                is_past_or_t0 = (yy < t0_y) or (yy == t0_y and mm <= t0_m)
                is_future = (yy > t0_y) or (yy == t0_y and mm > t0_m)

                # ----- ACTUAL (solo se abbiamo dati per quel mese+anno)
                if is_past_or_t0:
                    vp = monthly_valid_prices_for_year(yy, mm)
                    if vp.size > 0:
                        avg_on, uptime_frac, _ = compute_on_stats(vp, mode, off_threshold_usd_mwh, uptime_pct_input)
                        add_on = addon_usd_mwh_for_month(yy, mm, uptime_frac)
                        actual_allin.append(avg_on + add_on if np.isfinite(avg_on) else np.nan)
                    else:
                        actual_allin.append(np.nan)
                else:
                    actual_allin.append(np.nan)

                # ----- ESTIMATED FUTURE (media storico sugli ultimi N anni disponibili per quel mese)
                if is_future:
                    years_av = sorted(df_all.loc[df_all["month"] == mm, "year"].dropna().unique().tolist())
                    # escludi anno futuro
                    years_av = [y for y in years_av if y < yy] or years_av
                    if len(years_av) == 0:
                        est_allin.append(np.nan)
                        trend_allin.append(np.nan)
                        continue

                    years_used_local = years_av[-min(n_years, len(years_av)) :]

                    rows_tmp = []
                    yoy_base = []  # per tendenza
                    for yhist in years_used_local:
                        vp_hist = monthly_valid_prices_for_year(yhist, mm)
                        if vp_hist.size == 0:
                            continue
                        avg_on, uptime_frac, _ = compute_on_stats(vp_hist, mode, off_threshold_usd_mwh, uptime_pct_input)
                        add_on = addon_usd_mwh_for_month(yy, mm, uptime_frac)  # add-on riferito al mese FUTURO (giorni reali futuri)
                        allin = avg_on + add_on if np.isfinite(avg_on) else np.nan
                        rows_tmp.append(allin)
                        yoy_base.append(avg_on)  # trend sul solo RTM ON-only (poi aggiungo add-on medio)

                    est_allin.append(float(np.nanmean(rows_tmp)) if len(rows_tmp) > 0 else np.nan)

                    # ----- TREND FUTURE (YoY medio del mese applicato all'ultimo anno disponibile)
                    # Calcolo YoY sul RTM ON-only (senza add-on), poi aggiungo add-on medio.
                    if len(yoy_base) >= 2:
                        # ricostruisco una serie ordinata per anno
                        tmp = []
                        for yhist in years_used_local:
                            vp_hist = monthly_valid_prices_for_year(yhist, mm)
                            if vp_hist.size == 0:
                                continue
                            avg_on, _, _ = compute_on_stats(vp_hist, mode, off_threshold_usd_mwh, uptime_pct_input)
                            tmp.append((yhist, avg_on))
                        tmp = [(a, b) for a, b in tmp if np.isfinite(b)]
                        tmp.sort(key=lambda x: x[0])

                        if len(tmp) >= 2:
                            vals = [b for _, b in tmp]
                            yrs = [a for a, _ in tmp]
                            yoy = [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals)) if vals[i - 1] != 0]
                            yoy_avg = float(np.mean(yoy)) if len(yoy) > 0 else 0.0

                            last_y = yrs[-1]
                            last_p = vals[-1]

                            # proiezione composta se yy è oltre last_y
                            k = max(1, yy - last_y)
                            projected = last_p * ((1.0 + yoy_avg) ** k)

                            # add-on: uso add-on medio tra anni usati (uptime-aware calcolato su ciascun anno, ma riferito al mese futuro)
                            add_on_list = []
                            for yhist in years_used_local:
                                vp_hist = monthly_valid_prices_for_year(yhist, mm)
                                if vp_hist.size == 0:
                                    continue
                                _, uptime_frac, _ = compute_on_stats(vp_hist, mode, off_threshold_usd_mwh, uptime_pct_input)
                                add_on_list.append(addon_usd_mwh_for_month(yy, mm, uptime_frac))
                            add_on_avg = float(np.mean(add_on_list)) if len(add_on_list) > 0 else 0.0

                            trend_allin.append(projected + add_on_avg)
                        else:
                            trend_allin.append(np.nan)
                    else:
                        trend_allin.append(np.nan)
                else:
                    est_allin.append(np.nan)
                    trend_allin.append(np.nan)

            # X axis labels (con evidenza T0)
            x_labels = [month_label(yy, mm) for (yy, mm) in months_axis]

            # -----------------------------
            # Grafico 1: storico reale + FUTURO stimato
            # -----------------------------
            st.markdown("#### 📊 1) Timeline – Futuro stimato (media storico + add-on)")

            fig1 = go.Figure()
            fig1.add_bar(x=x_labels, y=actual_allin, name="Reale (se disponibile) – ALL-IN ON")
            fig1.add_bar(x=x_labels, y=est_allin, name="Futuro stimato – ALL-IN ON")

            # Linea verticale T0 (mese corrente)
            t0_label = month_label(t0_y, t0_m)
            fig1.add_vline(x=t0_label, line_width=2, line_dash="dash")

            fig1.update_layout(
                barmode="group",
                xaxis_title="Mese (T0 al centro)",
                yaxis_title="Prezzo [USD/MWh]",
                hovermode="x unified",
            )
            st.plotly_chart(fig1, use_container_width=True)

            # -----------------------------
            # Grafico 2: storico reale + FUTURO in tendenza
            # -----------------------------
            st.markdown("#### 📈 2) Timeline – Futuro in tendenza (YoY mese + add-on)")

            fig2 = go.Figure()
            fig2.add_bar(x=x_labels, y=actual_allin, name="Reale (se disponibile) – ALL-IN ON")
            fig2.add_bar(x=x_labels, y=trend_allin, name="Futuro tendenza – ALL-IN ON")

            fig2.add_vline(x=t0_label, line_width=2, line_dash="dash")

            fig2.update_layout(
                barmode="group",
                xaxis_title="Mese (T0 al centro)",
                yaxis_title="Prezzo [USD/MWh]",
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)



def _find_subset_sum(amounts_cents, target_cents, tol_cents, max_items=10):
    """
    Cerca una combinazione (subset) di amounts_cents che somma ~ target_cents (entro tol_cents).
    DFS limitato: max_items e pruning semplice.
    Ritorna lista di indici oppure None.
    """
    # ordina per importo decrescente per migliorare pruning
    indexed = list(enumerate(amounts_cents))
    indexed.sort(key=lambda x: x[1], reverse=True)

    best = None

    def dfs(i, chosen, s):
        nonlocal best
        if best is not None:
            return  # troviamo la prima combinazione valida e stop (ok per riconciliazione pratica)

        if abs(s - target_cents) <= tol_cents:
            best = chosen[:]
            return

        if i >= len(indexed):
            return

        if len(chosen) >= max_items:
            return

        # pruning: se già sopra target + tol e importi tutti positivi
        if s > target_cents + tol_cents:
            return

        idx, val = indexed[i]

        # scegli
        chosen.append(idx)
        dfs(i + 1, chosen, s + val)
        chosen.pop()

        # non scegliere
        dfs(i + 1, chosen, s)

    dfs(0, [], 0)
    return best


with tab13:
    st.header("🧾 Riconciliazione pagamento elettrico (Synota ↔ Bank)")

    if synota_df is None or synota_df.empty or "invoice_amount_usd" not in synota_df.columns:
        st.warning("Carica un **Synota CSV** con la colonna **Invoice Amount** per usare questa tab.")
        st.stop()

    if bank_df is None or bank_df.empty:
        st.info("Carica in sidebar il **Chase Activity CSV** per avviare la riconciliazione.")
        st.stop()

    st.caption("Logica: prendo gli addebiti bancari con 'SYNOTA' e li confronto con i costi dal CSV Synota.")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        date_window_days = st.number_input("Finestra date ± giorni", min_value=0, value=7, step=1)
    with colB:
        tolerance_usd = st.number_input("Tolleranza match [USD]", min_value=0.0, value=1.00, step=0.25)
    with colC:
        allow_sum_match = st.checkbox("Permetti match per somma (1 bank ↔ N Synota)", value=True)
    with colD:
        max_items_sum = st.number_input("Max righe Synota in un match-somma", min_value=2, value=10, step=1)

    # --- filtri sorgenti ---
    syn_cost = synota_df[["date", "invoice_amount_usd"]].dropna().copy()
    syn_cost["date"] = pd.to_datetime(syn_cost["date"]).dt.normalize()
    syn_cost["synota_amount"] = syn_cost["invoice_amount_usd"].astype(float).abs()
    syn_cost = syn_cost.sort_values("date").reset_index(drop=True)

    bank = bank_df.copy()
    bank["date"] = pd.to_datetime(bank["date"]).dt.normalize()
    bank["bank_amount"] = bank["amount"].astype(float).abs()

    # Solo addebiti Synota (DEBIT) – filtro testo
    bank_synota = bank[
        bank["description"].str.contains("SYNOTA", case=False, na=False)
    ].copy()

    # opzionale: prendi solo DEBIT
    bank_synota = bank_synota[bank_synota["amount"] < 0].copy()
    bank_synota = bank_synota.sort_values("date").reset_index(drop=True)

    if bank_synota.empty:
        st.warning("Nessun movimento in banca contenente 'SYNOTA' (DEBIT). Controlla il CSV.")
        st.stop()

    tol_cents = int(round(tolerance_usd * 100))

    # --- matching 1:1 prima ---
    syn_used = set()
    matches = []

    for bi, brow in bank_synota.iterrows():
        bdate = brow["date"]
        bamt_c = int(round(brow["bank_amount"] * 100))

        start = bdate - pd.Timedelta(days=int(date_window_days))
        end = bdate + pd.Timedelta(days=int(date_window_days))

        candidates = syn_cost[
            (syn_cost["date"] >= start) &
            (syn_cost["date"] <= end) &
            (~syn_cost.index.isin(list(syn_used)))
        ].copy()

        if candidates.empty:
            continue

        candidates["syn_cents"] = (candidates["synota_amount"] * 100).round().astype(int)
        candidates["diff_cents"] = (candidates["syn_cents"] - bamt_c).abs()

        # prendi candidato più vicino entro tolleranza
        ok = candidates[candidates["diff_cents"] <= tol_cents].copy()
        if not ok.empty:
            ok["date_diff"] = (ok["date"] - bdate).abs()
            ok = ok.sort_values(["diff_cents", "date_diff"]).head(1)
            si = int(ok.index[0])

            syn_used.add(si)
            matches.append({
                "bank_date": bdate,
                "bank_desc": brow["description"],
                "bank_amount_usd": float(brow["bank_amount"]),
                "synota_dates": [syn_cost.loc[si, "date"]],
                "synota_amount_usd": float(syn_cost.loc[si, "synota_amount"]),
                "delta_usd": float(syn_cost.loc[si, "synota_amount"] - float(brow["bank_amount"])),
                "match_type": "1:1"
            })

    # --- matching per somma (1 bank ↔ N synota) ---
    if allow_sum_match:
        unmatched_bank = []
        matched_bank_keys = {(m["bank_date"], m["bank_amount_usd"]) for m in matches}
        for bi, brow in bank_synota.iterrows():
            key = (brow["date"], float(brow["bank_amount"]))
            if key not in matched_bank_keys:
                unmatched_bank.append((bi, brow))

        for bi, brow in unmatched_bank:
            bdate = brow["date"]
            bamt = float(brow["bank_amount"])
            target_c = int(round(bamt * 100))

            start = bdate - pd.Timedelta(days=int(date_window_days))
            end = bdate + pd.Timedelta(days=int(date_window_days))

            pool = syn_cost[
                (syn_cost["date"] >= start) &
                (syn_cost["date"] <= end) &
                (~syn_cost.index.isin(list(syn_used)))
            ].copy()

            if pool.empty:
                continue

            pool["syn_cents"] = (pool["synota_amount"] * 100).round().astype(int)
            amounts = pool["syn_cents"].tolist()

            subset = _find_subset_sum(amounts, target_c, tol_cents, max_items=int(max_items_sum))
            if subset is None:
                continue

            # subset contiene indici rispetto a pool (posizioni), convertiamo a indici reali syn_cost
            chosen_pool = pool.iloc[subset].copy()
            chosen_idx = chosen_pool.index.tolist()

            for si in chosen_idx:
                syn_used.add(int(si))

            syn_sum = chosen_pool["synota_amount"].sum()
            delta = syn_sum - bamt

            matches.append({
                "bank_date": bdate,
                "bank_desc": brow["description"],
                "bank_amount_usd": bamt,
                "synota_dates": chosen_pool["date"].tolist(),
                "synota_amount_usd": float(syn_sum),
                "delta_usd": float(delta),
                "match_type": f"1:{len(chosen_pool)}"
            })

    # --- output tables ---
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(["bank_date", "bank_amount_usd"]).reset_index(drop=True)

        # pretty synota_dates
        matches_df["synota_dates"] = matches_df["synota_dates"].apply(
            lambda lst: ", ".join([pd.to_datetime(x).date().isoformat() for x in lst]) if isinstance(lst, list) else ""
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Movimenti banca SYNOTA", len(bank_synota))
        with col2:
            st.metric("Match trovati", len(matches_df))
        with col3:
            st.metric("Totale delta assoluto [USD]", f"{matches_df['delta_usd'].abs().sum():,.2f}")

        st.markdown("### ✅ Match trovati")
        st.dataframe(
            matches_df.rename(columns={
                "bank_date": "Bank date",
                "bank_desc": "Bank description",
                "bank_amount_usd": "Bank amount [USD]",
                "synota_dates": "Synota date(s)",
                "synota_amount_usd": "Synota sum [USD]",
                "delta_usd": "Delta (Synota - Bank) [USD]",
                "match_type": "Match type",
            }).style.format({
                "Bank amount [USD]": "{:,.2f}",
                "Synota sum [USD]": "{:,.2f}",
                "Delta (Synota - Bank) [USD]": "{:+,.2f}",
            }),
            use_container_width=True
        )
    else:
        st.warning("Nessun match trovato con i parametri attuali. Prova ad aumentare la finestra date o la tolleranza.")

    # --- Unmatched lists ---
    matched_bank_idx = set()
    if not matches_df.empty:
        # ricostruisco chi è matchato (approssimazione: bank_date+amount)
        keys = set(zip(matches_df["bank_date"], matches_df["bank_amount_usd"]))
        for i, r in bank_synota.iterrows():
            if (r["date"], float(r["bank_amount"])) in keys:
                matched_bank_idx.add(i)

    bank_unmatched = bank_synota.drop(index=list(matched_bank_idx), errors="ignore").copy()
    syn_unmatched = syn_cost.drop(index=list(syn_used), errors="ignore").copy()

    st.markdown("### ⚠️ Non riconciliati (Bank → SYNOTA)")
    st.dataframe(
        bank_unmatched[["date", "description", "amount"]].rename(columns={
            "date": "Bank date",
            "description": "Bank description",
            "amount": "Bank amount (signed)",
        }),
        use_container_width=True
    )

    st.markdown("### ⚠️ Non riconciliati (Synota → Bank)")
    st.dataframe(
        syn_unmatched[["date", "synota_amount"]].rename(columns={
            "date": "Synota date",
            "synota_amount": "Synota invoice [USD]",
        }).style.format({"Synota invoice [USD]": "{:,.2f}"}),
        use_container_width=True
    )
