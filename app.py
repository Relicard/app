# DESMO Bitcoin Mining Optimizer 

from __future__ import annotations

import math, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta, timezone

import requests, pandas as pd, numpy as np # type: ignore
from dateutil.relativedelta import relativedelta  # type: ignore
import streamlit as st # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
from threading import RLock
from AntPool import antpool as antpool_client

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCENARIOS_FILE = DATA_DIR / "public_scenarios.json"   # tutti gli scenari condivisi
LOCK = RLock()

def _read_all() -> dict:
    if not SCENARIOS_FILE.exists():
        return {"classica": [], "prossimi_step": [], "hosting": []}
    try:
        with LOCK, SCENARIOS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"classica": [], "prossimi_step": [], "hosting": []}

def _write_all(payload: dict) -> None:
    with LOCK, SCENARIOS_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def save_public_scenario(kind: str, payload: dict) -> None:
    """kind ∈ {'classica','prossimi_step','hosting'}"""
    data = _read_all()
    data.setdefault(kind, [])
    data[kind].append(payload)
    _write_all(data)

def delete_public_scenario(kind: str, index: int) -> bool:
    """Elimina lo scenario pubblico di tipo `kind` all'indice `index`.
    Ritorna True se eliminato, False se indice fuori range."""
    data = _read_all()
    arr = data.get(kind, [])
    if 0 <= index < len(arr):
        del arr[index]
        data[kind] = arr
        _write_all(data)
        return True
    return False

def clear_public_scenarios(kind: str) -> None:
    """Cancella tutti gli scenari pubblici di quel tipo."""
    data = _read_all()
    data[kind] = []
    _write_all(data)

def list_public_scenarios(kind: str) -> list[dict]:
    data = _read_all()
    return data.get(kind, [])

def get_secret(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default

try:
    import gridstatus as gs  # type: ignore
except Exception:
    HAS_GRIDSTATUS = False


ANTPOOL_USER_ID = get_secret("ANTPOOL_USER_ID")
ANTPOOL_API_KEY = get_secret("ANTPOOL_API_KEY")
ANTPOOL_API_SECRET = get_secret("ANTPOOL_API_SECRET")



try:
    import pulp  # type: ignore
    HAS_PULP = True
except Exception:
    HAS_PULP = False

ERCOT_FILE = DATA_DIR / "ercot_prices.json"

def _read_ercot_history() -> list[dict]:
    if not ERCOT_FILE.exists():
        return []
    try:
        with ERCOT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_ercot_history(data: list[dict]) -> None:
    with ERCOT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_ercot_price(location: str, price: float) -> None:
    history = _read_ercot_history()
    history.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "location": location,
        "price_usd_per_kwh": price,
    })
    _write_ercot_history(history)

def list_ercot_prices() -> pd.DataFrame:
    data = _read_ercot_history()
    return pd.DataFrame(data)

def fetch_ercot_last_24h(api_key: str, location: str) -> pd.DataFrame:
    """
    Ultime 24 ore di LMP per settlement point ERCOT.
    Converte $/MWh -> $/kWh dividendo per 1000.
    """
    url = "https://api.gridstatus.io/v1/datasets/ercot_lmp_by_settlement_point/query"
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)

    params = {
        "filter_column": "location",
        "filter_value": location,
        # >>> parametri corretti per l'intervallo temporale <<<
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "order": "asc",
        "page_size": 1000,   # bastano per 24h a 5-min (≈288 righe)
    }
    headers = {
        "x-api-key": api_key,              # consigliato da Grid Status
        "User-Agent": "DESMO-Optimizer/1.1"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        rows = r.json().get("data", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # timestamp del dataset
        if "interval_start_utc" in df.columns:
            df["timestamp"] = pd.to_datetime(df["interval_start_utc"], utc=True)
        elif "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        else:
            # fallback: prova a indovinare
            ts_col = next((c for c in df.columns if "time" in c or "utc" in c), None)
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True) if ts_col else pd.NaT

        # $/MWh -> $/kWh
        df["price_usd_per_kwh"] = pd.to_numeric(df["lmp"], errors="coerce") / 1000.0
        df["location"] = location
        return df[["timestamp", "price_usd_per_kwh", "location"]].sort_values("timestamp")
    except Exception as e:
        st.error(f"Errore fetch_ercot_last_24h: {e}")
        return pd.DataFrame()







# -----------------------------
# Config
# -----------------------------

st.set_page_config(page_title="DESMO Mining Optimizer", layout="wide")

USERNAME = st.secrets.get("USERNAME", os.getenv("USERNAME"))
PASSWORD = st.secrets.get("PASSWORD", os.getenv("PASSWORD"))

# -----------------------------
# Simple Login
# -----------------------------
def require_login():
    st.session_state.setdefault("auth_ok", False)
    if not st.session_state["auth_ok"]:
        with st.form("login"):
            u = st.text_input("User")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Entra")
        if ok:
            if u == USERNAME and p == PASSWORD:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Credenziali errate")
                st.stop()
    if not st.session_state["auth_ok"]:
        st.stop()

require_login()

# -----------------------------
# API fetchers (cached)
# -----------------------------
HEADERS = {"User-Agent": "DESMO-Optimizer/1.1 (https://desmo.example)"}

@st.cache_data(ttl=300)
def fetch_btc_price_usd() -> float:
    """Spot price from Coinbase public API (no key needed)."""
    try:
        r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=10, headers=HEADERS)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])  # USD
    except Exception:
        # Fallback to CoinGecko simple price (public, rate-limited).
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin", "vs_currencies": "usd"},
                timeout=10,
                headers=HEADERS
            )
            r.raise_for_status()
            return float(r.json()["bitcoin"]["usd"])
        except Exception:
            return float("nan")

@st.cache_data(ttl=600)
def fetch_block_height() -> Optional[int]:
    try:
        r = requests.get("https://blockchain.info/q/getblockcount?cors=true", timeout=10, headers=HEADERS)
        r.raise_for_status()
        return int(r.text)
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_difficulty() -> Optional[float]:
    try:
        r = requests.get("https://blockchain.info/q/getdifficulty?cors=true", timeout=10, headers=HEADERS)
        r.raise_for_status()
        return float(r.text)
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_block_reward_btc() -> Optional[float]:
    """Current block subsidy in BTC (excludes fees)."""
    try:
        r = requests.get("https://blockchain.info/q/bcperblock?cors=true", timeout=10, headers=HEADERS)
        r.raise_for_status()
        return float(r.text)
    except Exception:
        # Compute from halving schedule if needed
        height = fetch_block_height()
        if height is None:
            return None
        halvings = height // 210_000
        reward = 50.0 / (2 ** halvings)
        return max(reward, 0.0)

# --- Average Fees 7d (mempool.space) ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_avg_fees_per_block_btc_7d_latest() -> Optional[float]:
    """
    Prende l'ULTIMO campione della serie 1w da mempool.space:
    GET /api/v1/mining/blocks/fees/1w
    Usa 'avgFees' (in sats) -> BTC (8 decimali).
    """
    try:
        url = "https://mempool.space/api/v1/mining/blocks/fees/1w"
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        last = data[-1]                       # ultimo campione (più recente)
        sats = last.get("avgFees", None)
        if sats is None:
            return None
        return float(sats) / 1e8              # sats -> BTC
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_avg_fees_last_n_blocks_btc(n: int = 1000) -> Optional[tuple[float, int, int]]:
    """
    Media delle fee per blocco (in BTC) sugli ultimi n blocchi.
    Usa: GET /api/v1/mining/reward-stats/:blockCount
    Ritorna (avg_btc_per_block, start_block, end_block) oppure None.
    """
    try:
        url = f"https://mempool.space/api/v1/mining/reward-stats/{n}"
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        total_fee_sats = int(data.get("totalFee", "0"))  # string -> int sats
        if total_fee_sats <= 0:
            return None
        avg_sats = total_fee_sats / float(n)
        avg_btc = avg_sats / 1e8
        start_block = int(data.get("startBlock", 0))
        end_block   = int(data.get("endBlock", 0))
        return avg_btc, start_block, end_block
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_antpool_overview() -> Optional[dict]:
    """
    Recupera l'overview di Antpool per l'account configurato.
    Ritorna il dict 'data' se ok, altrimenti None.

    Campi tipici:
      - hsLast10m : hash rate ultimi 10 minuti (stringa, in H/s)
      - hsLast1h  : hash rate ultima ora
      - hsLast1d  : hash rate ultimo giorno
    """
    if not (ANTPOOL_USER_ID and ANTPOOL_API_KEY and ANTPOOL_API_SECRET):
        return None

    try:
        client = antpool_client.AntPool(
            ANTPOOL_USER_ID,
            ANTPOOL_API_KEY,
            ANTPOOL_API_SECRET,
        )
        resp = client.get_overview()   # come da doc solidity-antpool

        # La libreria, da doc PyPI, ritorna qualcosa tipo:
        # {"code": 0, "message": "ok", "data": {...}}
        if not isinstance(resp, dict):
            return None

        if resp.get("code") != 0:
            # facoltativo: loggare il messaggio
            st.warning(f"Antpool API error: {resp.get('message')}")
            return None

        return resp.get("data", None)
    except Exception as e:
        st.error(f"Errore chiamando Antpool: {e}")
        return None

    
@st.cache_data(ttl=60)
def fetch_ercot_rtm_price_per_kwh_api(
    api_key: str,
    location: str = "PIONR_DJ_RN",
    dataset_id: str = "ercot_lmp_by_settlement_point",
) -> Optional[float]:
    """
    Legge l'ultimo LMP RTM ($/MWh) per 'location' e lo converte in $/kWh.
    """
    try:
        url = f"https://api.gridstatus.io/v1/datasets/{dataset_id}/query"
        params = {
            "api_key": api_key,          # <-- chiave in query (massima compatibilità)
            "time": "latest",
            "filter_column": "location",
            "filter_value": location,
            "order": "desc",
            "limit": 1,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            st.warning(f"Nessun dato ERCOT per location={location}.")
            return None
        price_mwh = float(rows[0]["lmp"])
        return price_mwh / 1000.0  # $/kWh
    except Exception as e:
        st.error(f"Errore ERCOT RTM: {e}")
        return None



# -----------------------------
# Mining math & halving
# -----------------------------

SECONDS_PER_BLOCK = 600.0
BLOCKS_PER_DAY = 24.0 * 3600.0 / SECONDS_PER_BLOCK  # ~144
SATOSHI = 1e-8

def difficulty_to_network_hashrate_ths(difficulty: float) -> float:
    """Convert difficulty to network hashrate in TH/s.
    H = D * 2^32 / T  (hashes per second)
    """
    if difficulty is None:
        return float("nan")
    hashes_per_sec = difficulty * (2**32) / SECONDS_PER_BLOCK
    return hashes_per_sec / 1e12  # TH/s

def days_in_month(dt: pd.Timestamp) -> int:
    return (dt + pd.offsets.MonthEnd(0)).day

def monthly_weight_after_date(month_start: pd.Timestamp, cutover: date) -> Tuple[int, int]:
    """Return (days_before, days_after) within this month relative to 'cutover' date."""
    dim = days_in_month(month_start)
    m_year, m_mon = month_start.year, month_start.month
    # Cutover inside this month?
    if cutover.year == m_year and cutover.month == m_mon:
        day_cut = cutover.day
        before = max(min(day_cut - 1, dim), 0)
        after = dim - before
        return before, after
    # Entirely before cutover
    if (m_year, m_mon) < (cutover.year, cutover.month):
        return dim, 0
    # Entirely after cutover
    return 0, dim

def monthly_block_reward_btc(month_start: pd.Timestamp, halving_date: date,
                             subsidy_before: float, subsidy_after: float) -> float:
    """Average subsidy for the month, weighted by days around halving."""
    before, after = monthly_weight_after_date(month_start, halving_date)
    dim = before + after
    if dim == 0:
        return subsidy_before
    return (before * subsidy_before + after * subsidy_after) / dim

@dataclass
class AsicModel:
    name: str
    hashrate_ths: float  # per unit
    power_kw: float      # at-the-wall, per unit
    unit_price_usd: float
    disponibility: str = "not-set"

# Catalogo (puoi modificarlo in UI)
DEFAULT_CATALOG: Dict[str, AsicModel] = {
    "Antminer S19 PRO 110T": AsicModel("Antminer S19 PRO 110T", hashrate_ths=110.0, power_kw=3.25, unit_price_usd=319.0, disponibility="now"),
    "Antminer S19 XP 141T": AsicModel("Antminer S19 XP 141T", hashrate_ths=141.0, power_kw=3.01, unit_price_usd=1100.0, disponibility="now"),
    "Bitmain S21 200T": AsicModel("Bitmain S21 200T", hashrate_ths=200.0, power_kw=3.50, unit_price_usd=2840.0, disponibility="now"),
    "Bitmain S21 Pro 220T": AsicModel("Bitmain S21 Pro 220T", hashrate_ths=220.0, power_kw=3.300, unit_price_usd=3432.0, disponibility="now"),
    "Bitmain S21+ 225T": AsicModel("Bitmain S21+ 225T", hashrate_ths=225.0, power_kw=3.7125, unit_price_usd=3217.5, disponibility="now"),
    "Bitmain S21 Pro 234T": AsicModel("Bitmain S21 Pro 234T", hashrate_ths=234.0, power_kw=3.510, unit_price_usd=3814.2, disponibility="now"),
    "Bitmain S21+ 235T": AsicModel("Bitmain S21+ 235T", hashrate_ths=235.0, power_kw=3.8775, unit_price_usd=3384.0, disponibility="now"),
    "Bitmain S21 Pro 245T": AsicModel("Bitmain S21 Pro 245T", hashrate_ths=245.0, power_kw=3.675, unit_price_usd=4091.5, disponibility="now"),
    "Bitmain S21XP 270T": AsicModel("Bitmain S21XP 270T", hashrate_ths=270.0, power_kw=3.645, unit_price_usd=5805.0, disponibility="now"),
    "Bitdeer A2 Pro Air 226TH": AsicModel("Bitdeer A2 Pro air 226TH", hashrate_ths=226.0, power_kw=3.729, unit_price_usd=3557.0, disponibility="November"),
    "Bitdeer A2 Pro Hydro 500TH": AsicModel("Bitdeer A2 Pro Hydro 500TH", hashrate_ths=500.0, power_kw=7.450, unit_price_usd=7500.0, disponibility="November"),
    "Bitmain T21": AsicModel("Bitmain T21", hashrate_ths=190.0, power_kw=3.310, unit_price_usd=1950.0, disponibility="Now"),
}

@dataclass
class Scenario:
    name: str
    # Fleet (units per model)
    fleet: Dict[str, int]
    # Power & site
    pue: float  
    uptime_pct: float 
    # Costs
    fixed_opex_month_usd: float
    variable_energy_usd_per_kwh: float 
    capex_asics_usd: float
    capex_container_usd: float
    capex_transformer_usd: float
    other_capex_usd: float
    # Financial assumptions
    btc_price_override: Optional[float] = None
    avg_fees_per_block_btc_override: Optional[float] = None
    monthly_network_growth_pct: float = 0.0
    btc_price_monthly_growth_pct: float = 0.0
    months_horizon: int = 36

    def total_hashrate_ths(self, catalog: Dict[str, AsicModel]) -> float:
        return float(sum(catalog[m].hashrate_ths * n for m, n in self.fleet.items() if m in catalog))

    def total_power_kw(self, catalog: Dict[str, AsicModel]) -> float:
        return float(sum(catalog[m].power_kw * n for m, n in self.fleet.items() if m in catalog))

    def total_capex_usd(self, catalog: Dict[str, AsicModel]) -> float:
        asic_capex = self.capex_asics_usd if self.capex_asics_usd > 0 else sum(
            catalog[m].unit_price_usd * n for m, n in self.fleet.items() if m in catalog
        )
        return asic_capex + self.capex_container_usd + self.capex_transformer_usd + self.other_capex_usd

@dataclass
class FutureStep:
    """A step that applies from month_offset onward.
    - fleet: ADDED to existing fleet at month_offset (capex applied as lump that month)
    - other fields: if provided (not None), OVERRIDE from that month forward
    """
    name: str
    scenario_name: str
    month_offset: int  # e.g., 4 means apply starting in 4th month from t0 (0-based)
    fleet: Dict[str, int]
    # Optional overrides (None = inherit previous)
    pue: Optional[float] = None
    uptime_pct: Optional[float] = None
    fixed_opex_month_usd: Optional[float] = None
    variable_energy_usd_per_kwh: Optional[float] = None
    capex_asics_usd: float = 0.0
    capex_container_usd: float = 0.0
    capex_transformer_usd: float = 0.0
    other_capex_usd: float = 0.0
    btc_price_override: Optional[float] = None
    avg_fees_per_block_btc_override: Optional[float] = None
    monthly_network_growth_pct: Optional[float] = None
    btc_price_monthly_growth_pct: Optional[float] = None

# -----------------------------
# Hosting dataclass
# -----------------------------
@dataclass
class HostingScenario:
    name: str
    # Fleet hosted (units per model) — questi sono gli ASIC dei clienti
    fleet: Dict[str, int]

    # Efficienza sito
    pue: float
    uptime_pct: float  # %

    # Costi fissi mensili (nostri)
    fixed_opex_month_usd: float

    # Prezzi energia
    our_energy_usd_per_kwh: float      # quanto ci costa 1 kWh (come "variable_energy_usd_per_kwh")
    hosting_sell_usd_per_kwh: float    # prezzo di vendita 1 kWh al cliente

    # Prezzi ASIC (acquisto vs vendita — il surplus è ricavo one-off al mese 0)
    # Se un modello non è in sale_price_overrides, si assume uguale al prezzo di acquisto (=> markup 0)
    sale_price_overrides: Dict[str, float]

    # CAPEX (nostri) per sito/infrastruttura (gli ASIC li compriamo e rivendiamo)
    capex_asics_usd: float             # 0 => calcolato da catalogo (costo acquisto)
    capex_container_usd: float
    capex_transformer_usd: float
    other_capex_usd: float

    # Assunzioni finanziarie
    btc_price_override: Optional[float] = None
    avg_fees_per_block_btc_override: Optional[float] = None
    monthly_network_growth_pct: float = 0.0
    btc_price_monthly_growth_pct: float = 0.0
    months_horizon: int = 36

    # Commissione sul mining del cliente (es. 10% → 10.0)
    commission_hashrate_pct: float = 0.0

    # Helper: totali fleet
    def total_hashrate_ths(self, catalog: Dict[str, AsicModel]) -> float:
        return float(sum(catalog[m].hashrate_ths * n for m, n in self.fleet.items() if m in catalog))

    def total_power_kw(self, catalog: Dict[str, AsicModel]) -> float:
        return float(sum(catalog[m].power_kw * n for m, n in self.fleet.items() if m in catalog))

    def total_capex_usd(self, catalog: Dict[str, AsicModel]) -> float:
        # Il CAPEX ASIC è il nostro costo di acquisto (che poi “rientra” come vendita),
        # ma ai fini PROFIT consideriamo solo il markup in “ricavi one-off”.
        asic_capex = self.capex_asics_usd if self.capex_asics_usd > 0 else sum(
            catalog[m].unit_price_usd * n for m, n in self.fleet.items() if m in catalog
        )
        return float(asic_capex + self.capex_container_usd + self.capex_transformer_usd + self.other_capex_usd)

def scenario_to_public_dict(scn: Scenario, catalog: dict, author: str = "anonymous") -> dict:
    return {
        "type": "classica",
        "name": scn.name,
        "fleet": scn.fleet,
        "pue": scn.pue,
        "uptime_pct": scn.uptime_pct,
        "fixed_opex_month_usd": scn.fixed_opex_month_usd,
        "variable_energy_usd_per_kwh": scn.variable_energy_usd_per_kwh,
        "capex_asics_usd": scn.capex_asics_usd,
        "capex_container_usd": scn.capex_container_usd,
        "capex_transformer_usd": scn.capex_transformer_usd,
        "other_capex_usd": scn.other_capex_usd,
        "btc_price_override": scn.btc_price_override,
        "avg_fees_per_block_btc_override": scn.avg_fees_per_block_btc_override,
        "monthly_network_growth_pct": scn.monthly_network_growth_pct,
        "btc_price_monthly_growth_pct": scn.btc_price_monthly_growth_pct,
        "months_horizon": scn.months_horizon,
        "author": author,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

def step_to_public_dict(step: FutureStep) -> dict:
    d = step.__dict__.copy()
    d["type"] = "step"
    d["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return d

def hosting_to_public_dict(hscn: HostingScenario, author: str = "anonymous") -> dict:
    return {
        "type": "hosting",
        "name": hscn.name,
        "fleet": hscn.fleet,
        "pue": hscn.pue,
        "uptime_pct": hscn.uptime_pct,
        "fixed_opex_month_usd": hscn.fixed_opex_month_usd,
        "our_energy_usd_per_kwh": hscn.our_energy_usd_per_kwh,
        "hosting_sell_usd_per_kwh": hscn.hosting_sell_usd_per_kwh,
        "sale_price_overrides": hscn.sale_price_overrides,
        "capex_asics_usd": hscn.capex_asics_usd,
        "capex_container_usd": hscn.capex_container_usd,
        "capex_transformer_usd": hscn.capex_transformer_usd,
        "other_capex_usd": hscn.other_capex_usd,
        "btc_price_override": hscn.btc_price_override,
        "avg_fees_per_block_btc_override": hscn.avg_fees_per_block_btc_override,
        "monthly_network_growth_pct": hscn.monthly_network_growth_pct,
        "btc_price_monthly_growth_pct": hscn.btc_price_monthly_growth_pct,
        "months_horizon": hscn.months_horizon,
        "commission_hashrate_pct": hscn.commission_hashrate_pct,
        "author": author,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# -----------------------------
# Expected production & cashflow
# -----------------------------

def expected_btc_per_day(miner_ths: float, network_ths: float, block_reward_btc: float, avg_fees_btc: float, uptime: float) -> float:
    if any(map(lambda x: x is None or (isinstance(x,(int,float)) and (np.isnan(x) or x<=0)),
               [miner_ths, network_ths, block_reward_btc])):
        return float("nan")
    fees = max(avg_fees_btc or 0.0, 0.0)
    return (miner_ths / network_ths) * BLOCKS_PER_DAY * (block_reward_btc + fees) * uptime

def hourly_energy_cost_usd(power_kw: float, price_curve_usd_per_kwh: np.ndarray, uptime: float, pue: float) -> float:
    # power_kw is IT load; PUE accounts for overhead
    effective_kw = power_kw * pue
    return float(effective_kw * np.sum(price_curve_usd_per_kwh) * uptime)

def make_month_index(horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(0),
                         periods=horizon, freq="MS")

def build_price_curve_for_month(hours: int, flat: float, uploaded: Optional[np.ndarray]) -> np.ndarray:
    if uploaded is None or len(uploaded) == 0:
        return np.full(hours, flat, dtype=float)
    reps = int(math.ceil(hours / len(uploaded)))
    return np.resize(np.tile(uploaded, reps), hours)


def simulate_scenario(
    scn: Scenario,
    catalog: Dict[str, AsicModel],
    network_ths_start: float,
    avg_fees_btc: float,
    btc_price_usd_start: float,
    flat_price_usd_per_kwh: float,
    price_curve_usd_per_kwh: Optional[np.ndarray],
    halving_date: date,
    subsidy_before: float,
    subsidy_after: float,
) -> pd.DataFrame:
    """Return a monthly dataframe with production, revenue, costs, cashflow, cumulative, ROI, etc.
       Halving-aware block subsidy per-month (weighted by days).
    """
    uptime = scn.uptime_pct / 100.0
    months = make_month_index(scn.months_horizon)

    # dynamic paths
    nh_ths = network_ths_start
    nh_growth = scn.monthly_network_growth_pct / 100.0
    price = btc_price_usd_start
    price_growth = scn.btc_price_monthly_growth_pct / 100.0
    fees = avg_fees_btc if (scn.avg_fees_per_block_btc_override is None) else scn.avg_fees_per_block_btc_override

    it_power_kw = scn.total_power_kw(catalog)
    fleet_ths = scn.total_hashrate_ths(catalog)
    total_capex = scn.total_capex_usd(catalog)

    rows = []
    for month_idx, month_start in enumerate(months):
        dim = days_in_month(month_start)
        hours = 24 * dim
        price_curve = build_price_curve_for_month(hours, scn.variable_energy_usd_per_kwh, price_curve_usd_per_kwh)

        # halving-aware subsidy for this month
        subsidy_m = monthly_block_reward_btc(month_start, halving_date, subsidy_before, subsidy_after)

        # Production (fleet at current month)
        btc_day = expected_btc_per_day(fleet_ths, nh_ths, subsidy_m, fees or 0.0, uptime)
        btc_month = btc_day * dim
        rev_usd = btc_month * price

        # Energy cost
        energy_cost = hourly_energy_cost_usd(it_power_kw, price_curve, uptime, scn.pue)

        # OPEX fixed
        opex_fixed = scn.fixed_opex_month_usd

        # EBITDA-ish
        ebitda = rev_usd - energy_cost - opex_fixed

        rows.append({
            "month": month_start.strftime("%Y-%m"),
            "date": month_start.date(),
            "network_ths": nh_ths,
            "btc_month": btc_month,
            "rev_usd": rev_usd,
            "energy_cost_usd": energy_cost,
            "fixed_opex_usd": opex_fixed,
            "ebitda_usd": ebitda,
            "btc_price_usd": price,
            "subsidy_btc": subsidy_m,
            "fleet_ths": fleet_ths,
            "it_power_kw": it_power_kw,
        })

        # Update dynamic variables for next month
        nh_ths *= (1.0 + nh_growth)
        price = (price * (1.0 + price_growth)) if (scn.btc_price_override is None) else scn.btc_price_override

    df = pd.DataFrame(rows)

    # CAPEX once at t0
    df["cashflow_usd"] = df["ebitda_usd"]
    if len(df) > 0:
        df.loc[df.index[0], "cashflow_usd"] = df.loc[df.index[0], "cashflow_usd"] - total_capex
    df["cum_cashflow_usd"] = df["cashflow_usd"].cumsum()

    # Payback month (first month with cumulative >= 0)
    payback_idx = df.index[df["cum_cashflow_usd"] >= 0]
    df.attrs["payback_months"] = int(payback_idx[0] + 1) if len(payback_idx) else None
    df.attrs["total_capex_usd"] = total_capex
    df.attrs["fleet_ths"] = fleet_ths
    df.attrs["it_power_kw"] = it_power_kw
    return df

def simulate_scenario_with_steps(
    scn: Scenario,
    steps: List[FutureStep],
    catalog: Dict[str, AsicModel],
    network_ths_start: float,
    avg_fees_btc: float,
    btc_price_usd_start: float,
    flat_price_usd_per_kwh_global: float,
    price_curve_usd_per_kwh: Optional[np.ndarray],
    halving_date: date,
    subsidy_before: float,
    subsidy_after: float,
) -> pd.DataFrame:
    months = make_month_index(scn.months_horizon)

    # State (start from base)
    curr_fleet = dict(scn.fleet)
    curr_pue = scn.pue
    curr_uptime = scn.uptime_pct / 100.0
    curr_fixed_opex = scn.fixed_opex_month_usd
    curr_var_price = scn.variable_energy_usd_per_kwh if scn.variable_energy_usd_per_kwh is not None else flat_price_usd_per_kwh_global
    curr_btc_override = scn.btc_price_override
    curr_fees = avg_fees_btc if (scn.avg_fees_per_block_btc_override is None) else scn.avg_fees_per_block_btc_override
    curr_nh_growth = scn.monthly_network_growth_pct / 100.0
    curr_price_growth = scn.btc_price_monthly_growth_pct / 100.0

    # Dynamics
    nh_ths = network_ths_start
    price = (btc_price_usd_start if curr_btc_override is None else curr_btc_override)

    # Helpers
    def fleet_power_ths(fleet: Dict[str,int]) -> Tuple[float,float]:
        ths = sum(catalog[m].hashrate_ths * n for m,n in fleet.items() if m in catalog)
        kw = sum(catalog[m].power_kw * n for m,n in fleet.items() if m in catalog)
        return float(ths), float(kw)

    base_asic_capex = scn.capex_asics_usd if scn.capex_asics_usd > 0 else sum(
        catalog[m].unit_price_usd * n for m,n in scn.fleet.items() if m in catalog
    )
    capex_lumps = {0: base_asic_capex + scn.capex_container_usd + scn.capex_transformer_usd + scn.other_capex_usd}

    # Normalize steps for this scenario, map by month_offset
    steps_for_scn = [s for s in steps if s.scenario_name == scn.name]
    steps_by_month: Dict[int, List[FutureStep]] = {}
    for s in steps_for_scn:
        steps_by_month.setdefault(int(s.month_offset), []).append(s)

    rows = []

    for month_idx, month_start in enumerate(months):
        # Apply steps at this month (add fleet, override params, record capex)
        if month_idx in steps_by_month:
            for s in steps_by_month[month_idx]:
                # Add fleet (+ capex ASIC)
                for m, n in s.fleet.items():
                    curr_fleet[m] = int(curr_fleet.get(m, 0) + int(n))
                add_asic_capex = s.capex_asics_usd if s.capex_asics_usd > 0 else sum(
                    catalog[m].unit_price_usd * n for m, n in s.fleet.items() if m in catalog
                )
                add_capex = add_asic_capex + s.capex_container_usd + s.capex_transformer_usd + s.other_capex_usd
                if add_capex != 0:
                    capex_lumps[month_idx] = capex_lumps.get(month_idx, 0.0) + add_capex

                # Overrides
                if s.pue is not None: curr_pue = float(s.pue)
                if s.uptime_pct is not None: curr_uptime = float(s.uptime_pct) / 100.0
                if s.fixed_opex_month_usd is not None: curr_fixed_opex = float(s.fixed_opex_month_usd)
                if s.variable_energy_usd_per_kwh is not None: curr_var_price = float(s.variable_energy_usd_per_kwh)
                if s.btc_price_override is not None: price = float(s.btc_price_override); curr_btc_override = float(s.btc_price_override)
                if s.avg_fees_per_block_btc_override is not None: curr_fees = float(s.avg_fees_per_block_btc_override)
                if s.monthly_network_growth_pct is not None: curr_nh_growth = float(s.monthly_network_growth_pct) / 100.0
                if s.btc_price_monthly_growth_pct is not None: curr_price_growth = float(s.btc_price_monthly_growth_pct) / 100.0

        # Current fleet power/hash
        fleet_ths, it_power_kw = fleet_power_ths(curr_fleet)

        # Prices & curves for this month
        dim = days_in_month(month_start)
        hours = 24 * dim
        price_curve = build_price_curve_for_month(hours, curr_var_price, price_curve_usd_per_kwh)
        subsidy_m = monthly_block_reward_btc(month_start, halving_date, subsidy_before, subsidy_after)

        # Production
        btc_day = expected_btc_per_day(fleet_ths, nh_ths, subsidy_m, curr_fees or 0.0, curr_uptime)
        btc_month = btc_day * dim
        rev_usd = btc_month * price

        # Energy + opex
        energy_cost = hourly_energy_cost_usd(it_power_kw, price_curve, curr_uptime, curr_pue)
        ebitda = rev_usd - energy_cost - curr_fixed_opex

        rows.append({
            "month": month_start.strftime("%Y-%m"),
            "date": month_start.date(),
            "network_ths": nh_ths,
            "btc_month": btc_month,
            "rev_usd": rev_usd,
            "energy_cost_usd": energy_cost,
            "fixed_opex_usd": curr_fixed_opex,
            "ebitda_usd": ebitda,
            "btc_price_usd": price,
            "subsidy_btc": subsidy_m,
            "fleet_ths": fleet_ths,
            "it_power_kw": it_power_kw,
        })

        # Update for next month
        nh_ths *= (1.0 + curr_nh_growth)
        price = (price * (1.0 + curr_price_growth)) if (curr_btc_override is None) else curr_btc_override

    df = pd.DataFrame(rows)

    # Cashflows: include CAPEX lumps at their months
    df["cashflow_usd"] = df["ebitda_usd"]
    for m_idx, capex in capex_lumps.items():
        if m_idx < len(df):
            df.loc[m_idx, "cashflow_usd"] = df.loc[m_idx, "cashflow_usd"] - capex
    df["cum_cashflow_usd"] = df["cashflow_usd"].cumsum()

    # Payback
    payback_idx = df.index[df["cum_cashflow_usd"] >= 0]
    df.attrs["payback_months"] = int(payback_idx[0] + 1) if len(payback_idx) else None
    df.attrs["total_capex_usd"] = sum(capex_lumps.values())
    # last known fleet/power
    df.attrs["fleet_ths"] = float(df.iloc[-1]["fleet_ths"]) if len(df) else 0.0
    df.attrs["it_power_kw"] = float(df.iloc[-1]["it_power_kw"]) if len(df) else 0.0
    return df

# -----------------------------
# Simulazione modalità Hosting
# -----------------------------
def simulate_hosting_scenario(
    scn: HostingScenario,
    catalog: Dict[str, AsicModel],
    network_ths_start: float,
    avg_fees_btc: float,
    btc_price_usd_start: float,
    price_curve_usd_per_kwh: Optional[np.ndarray],
    halving_date: date,
    subsidy_before: float,
    subsidy_after: float,
) -> pd.DataFrame:
    """
    Restituisce un DF mensile con:
      - Ricavo energia: hosting_sell * kWh
      - Costo energia: our_cost * kWh
      - Margine energia: (hosting_sell - our_cost) * kWh
      - Commissione mining: % su revenue mining cliente (in USD)
      - Markup ASIC (one-off al mese 0): sum( (sale_price - buy_price) * units )
      - EBITDA: margine energia + commissione - OPEX fissi
      - Cashflow: EBITDA - **solo Infra CAPEX** (al mese 0) + markup_oneoff (mese 0)
    NOTA: il costo d’acquisto ASIC NON entra nel cashflow (lo recuperiamo via vendita).
    """
    uptime = scn.uptime_pct / 100.0
    months = make_month_index(scn.months_horizon)

    # Traiettorie dinamiche
    nh_ths = float(network_ths_start)
    nh_growth = scn.monthly_network_growth_pct / 100.0
    price_usd = float(btc_price_usd_start if scn.btc_price_override is None else scn.btc_price_override)
    price_growth = scn.btc_price_monthly_growth_pct / 100.0
    fees_block = float(avg_fees_btc if scn.avg_fees_per_block_btc_override is None else scn.avg_fees_per_block_btc_override)

    # Fleet e potenza
    fleet_ths = scn.total_hashrate_ths(catalog)
    it_power_kw = scn.total_power_kw(catalog)

    # Markup ASIC one-off (al mese 0) e costo acquisto (solo informativo)
    def _buy_cost_unit(m: str) -> float:
        return float(catalog[m].unit_price_usd)

    asic_markup_usd = 0.0
    asic_buy_cost = 0.0
    for m, units in scn.fleet.items():
        if m in catalog and units > 0:
            buy = _buy_cost_unit(m)
            sell = float(scn.sale_price_overrides.get(m, buy))
            asic_buy_cost += buy * float(units)
            asic_markup_usd += max(sell - buy, 0.0) * float(units)
    asic_markup_usd = float(asic_markup_usd)
    asic_buy_cost = float(asic_buy_cost)  # non usato nel cashflow

    # --- CAPEX ---
    # Solo infrastruttura va nel cashflow (container/trafo/other)
    infra_capex = float(scn.capex_container_usd + scn.capex_transformer_usd + scn.other_capex_usd)

    rows = []
    for month_idx, month_start in enumerate(months):
        dim = days_in_month(month_start)
        hours = 24 * dim

        # Curve prezzi energia (se non fornita una curva, usa i flat)
        price_curve_our = build_price_curve_for_month(hours, scn.our_energy_usd_per_kwh, price_curve_usd_per_kwh)
        price_curve_sell = build_price_curve_for_month(hours, scn.hosting_sell_usd_per_kwh, price_curve_usd_per_kwh)

        # kWh consumati (IT × PUE × ore × uptime)
        effective_kw = it_power_kw * scn.pue
        kwh_month = effective_kw * float(np.sum(np.ones(hours))) * uptime  # = effective_kw * hours * uptime

        # Ricavi/Costi energetici (integrale delle curve * kW * uptime)
        rev_energy = float(effective_kw * float(np.sum(price_curve_sell)) * uptime)
        cost_energy = float(effective_kw * float(np.sum(price_curve_our)) * uptime)
        margin_energy = float(rev_energy - cost_energy)

        # Commissione su revenue mining del cliente
        subsidy_m = monthly_block_reward_btc(month_start, halving_date, subsidy_before, subsidy_after)
        btc_day_client = expected_btc_per_day(fleet_ths, nh_ths, subsidy_m, fees_block or 0.0, uptime)
        btc_month_client = btc_day_client * dim
        mining_rev_client_usd = float(btc_month_client * price_usd)
        commission_usd = float(mining_rev_client_usd * (scn.commission_hashrate_pct / 100.0))

        # EBITDA hosting (margine energia + commissione - opex fissi)
        ebitda = margin_energy + commission_usd - scn.fixed_opex_month_usd

        # Ricavi mensili totali "visivi" (senza markup one-off)
        rev_month_total = rev_energy + commission_usd

        rows.append({
            "month": month_start.strftime("%Y-%m"),
            "date": month_start.date(),
            "network_ths": nh_ths,
            "btc_price_usd": price_usd,
            "subsidy_btc": subsidy_m,
            "fleet_ths": fleet_ths,
            "it_power_kw": it_power_kw,

            # Energia / commissioni
            "kwh_equivalent": kwh_month,
            "energy_revenue_usd": rev_energy,
            "energy_cost_usd": cost_energy,
            "energy_margin_usd": margin_energy,
            "commission_usd": commission_usd,

            # Totali di periodo
            "rev_usd": rev_month_total,
            "fixed_opex_usd": scn.fixed_opex_month_usd,
            "ebitda_usd": ebitda,

            # Markup ASIC (mettiamo 0 di default, lo inseriamo nel mese 0 dopo)
            "asic_markup_usd": 0.0,
        })

        # Aggiorna traiettorie per mese successivo
        nh_ths *= (1.0 + nh_growth)
        price_usd = (price_usd * (1.0 + price_growth)) if (scn.btc_price_override is None) else scn.btc_price_override

    df = pd.DataFrame(rows)

    # Markup ASIC solo al mese 0
    if len(df) > 0:
        df.loc[df.index[0], "asic_markup_usd"] = asic_markup_usd

    # Cashflow: EBITDA - **SOLO infra CAPEX** (al mese 0) + markup_oneoff (mese 0)
    df["cashflow_usd"] = df["ebitda_usd"]
    if len(df) > 0:
        df.loc[df.index[0], "cashflow_usd"] = df.loc[df.index[0], "cashflow_usd"] - infra_capex + asic_markup_usd
    df["cum_cashflow_usd"] = df["cashflow_usd"].cumsum()

    # Payback & attributi
    payback_idx = df.index[df["cum_cashflow_usd"] >= 0]
    df.attrs["payback_months"] = int(payback_idx[0] + 1) if len(payback_idx) else None
    df.attrs["total_capex_usd"] = infra_capex          # mostriamo solo il CAPEX infrastrutturale
    df.attrs["fleet_ths"] = fleet_ths
    df.attrs["it_power_kw"] = it_power_kw
    df.attrs["asic_markup_usd"] = asic_markup_usd
    df.attrs["asic_buy_cost_usd"] = asic_buy_cost
    
    # -----------------------------
    # ROI Cliente (mesi)
    # -----------------------------
    # Costo iniziale per il cliente = somma dei prezzi DI VENDITA degli ASIC (sale_price_overrides)
    # Se non specificato un prezzo di vendita per un modello, usa il buy price (unit_price_usd).
    client_purchase_total = 0.0
    for m, units in scn.fleet.items():
        if m in catalog and units > 0:
            buy = float(catalog[m].unit_price_usd)
            sell = float(scn.sale_price_overrides.get(m, buy))
            client_purchase_total += sell * float(units)
    client_purchase_total = float(client_purchase_total)

    # Per-mese:
    # - Ricavo mining lordo del cliente = mining_rev_client_usd (già calcolato sopra)
    # - Commissione DESMO = commission_usd (già calcolata)
    # - Corrispettivo energia pagato dal cliente = revenue energia DESMO = energy_revenue_usd
    # Cashflow cliente mese = (mining_rev_client_usd - commission_usd) - energy_revenue_usd
    # Nota: l'acquisto ASIC è un esborso iniziale (al mese 0).
    df["client_cashflow_usd"] = (df["rev_usd"]  # rev_usd = energy_revenue + commission
                                 - df["energy_revenue_usd"]  # toglie l'energia
                                 - df["commission_usd"])     # toglie la commissione (=> resta 0)
    # ATTENZIONE: rev_usd = energy_revenue_usd + commission_usd, quindi la riga sopra darebbe 0.
    # Usiamo quindi direttamente il mining del cliente già calcolato in 'commission_usd':
    # commission_usd = mining_rev_client_usd * pct  =>  mining_rev_client_usd = commission_usd / pct
    # Per evitare divisioni per 0, ricalcoliamo il mining_rev_client_usd in modo robusto:
    pct = float(scn.commission_hashrate_pct) / 100.0
    if pct > 0:
        df["mining_rev_client_usd"] = df["commission_usd"] / pct
    else:
        # Se pct == 0, ricalcoliamo con le stesse formule usate nel loop:
        # Per semplicità: mining_rev_client_usd = rev_usd (non perfetto se commissione=0, ma ok)
        df["mining_rev_client_usd"] = df["rev_usd"]

    # Cashflow cliente corretto:
    df["client_cashflow_usd"] = df["mining_rev_client_usd"] - df["commission_usd"] - df["energy_revenue_usd"]

    # Aggiungi esborso iniziale d'acquisto al mese 0
    if len(df) > 0:
        df.loc[df.index[0], "client_cashflow_usd"] = df.loc[df.index[0], "client_cashflow_usd"] - client_purchase_total

    df["client_cum_cashflow_usd"] = df["client_cashflow_usd"].cumsum()

    # ROI Cliente (primo mese in cui il cumulato del cliente >= 0)
    idx_cli = df.index[df["client_cum_cashflow_usd"] >= 0]
    df.attrs["roi_cliente_months"] = int(idx_cli[0] + 1) if len(idx_cli) else None
    df.attrs["client_purchase_total_usd"] = client_purchase_total
      # informativo (non usato nel cashflow)
    return df


# -----------------------------
# Optimizer (Budget ASIC)
# -----------------------------

def per_unit_monthly_ebitda_series(
    model: AsicModel,
    horizon: int,
    network_ths_start: float,
    nh_growth_pct_m: float,
    btc_price_start: float,
    price_growth_pct_m: float,
    avg_fees_btc: float,
    uptime_pct: float,
    pue: float,
    flat_price_usd_per_kwh: float,
    price_curve_usd_per_kwh: Optional[np.ndarray],
    halving_date: date,
    subsidy_before: float,
    subsidy_after: float,
) -> np.ndarray:
    """Return EBITDA per month (USD) for 1 unit of this model across horizon."""
    months = make_month_index(horizon)
    nh_ths = float(network_ths_start)
    price = float(btc_price_start)
    nh_g = nh_growth_pct_m / 100.0
    pr_g = price_growth_pct_m / 100.0
    uptime = uptime_pct / 100.0

    series = np.zeros(horizon, dtype=float)
    for i, mstart in enumerate(months):
        dim = days_in_month(mstart)
        hours = 24 * dim
        subsidy_m = monthly_block_reward_btc(mstart, halving_date, subsidy_before, subsidy_after)
        btc_day = expected_btc_per_day(model.hashrate_ths, nh_ths, subsidy_m, avg_fees_btc or 0.0, uptime)
        btc_month = btc_day * dim
        rev_usd = btc_month * price

        price_curve = build_price_curve_for_month(hours, flat_price_usd_per_kwh, price_curve_usd_per_kwh)
        energy_cost = hourly_energy_cost_usd(model.power_kw, price_curve, uptime, pue)
        series[i] = rev_usd - energy_cost  # no fixed opex (handled globally)
        nh_ths *= (1.0 + nh_g)
        price *= (1.0 + pr_g)
    return series

def optimize_asic_budget(
    budget_usd: float,
    catalog: Dict[str, AsicModel],
    horizon_months: int,
    base_env: Scenario,
    network_ths_start: float,
    btc_price_start: float,
    avg_fees_btc: float,
    price_curve_usd_per_kwh: Optional[np.ndarray],
    halving_date: date,
    subsidy_before: float,
    subsidy_after: float,
) -> Tuple[Dict[str,int], float, pd.DataFrame]:
    """Maximize sum EBITDA over horizon under ASIC budget."""
    values: Dict[str, float] = {}
    costs: Dict[str, float] = {}
    for name, m in catalog.items():
        series = per_unit_monthly_ebitda_series(
            m, horizon_months,
            network_ths_start,
            base_env.monthly_network_growth_pct,
            btc_price_start if base_env.btc_price_override is None else base_env.btc_price_override,
            base_env.btc_price_monthly_growth_pct,
            (avg_fees_btc if base_env.avg_fees_per_block_btc_override is None else base_env.avg_fees_per_block_btc_override) or 0.0,
            base_env.uptime_pct,
            base_env.pue,
            base_env.variable_energy_usd_per_kwh,
            price_curve_usd_per_kwh,
            halving_date, subsidy_before, subsidy_after
        )
        values[name] = float(series.sum())
        costs[name] = float(m.unit_price_usd)

    model_names = list(catalog.keys())
    # Exact with PuLP if available
    if HAS_PULP:
        prob = pulp.LpProblem("ASIC_Budget_Max_EBITDA", pulp.LpMaximize)
        x = {n: pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger) for i, n in enumerate(model_names)}
        prob += pulp.lpSum(values[n] * x[n] for n in model_names)
        prob += pulp.lpSum(costs[n] * x[n] for n in model_names) <= budget_usd
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        sel = {n: int(pulp.value(x[n]) or 0) for n in model_names}
        best_val = float(pulp.value(prob.objective) or 0.0)
    else:
        # Fallback DP
        scale = 10  # 0.1 USD
        B = int(math.floor(budget_usd * scale + 1e-6))
        icosts = {n: int(math.ceil(costs[n] * scale - 1e-9)) for n in model_names}
        dp = np.zeros(B+1, dtype=float)
        choice = [-1]*(B+1)
        for b in range(B+1):
            for i, n in enumerate(model_names):
                c = icosts[n]
                if c <= b:
                    val = dp[b - c] + values[n]
                    if val > dp[b]:
                        dp[b] = val
                        choice[b] = i
        sel = {n: 0 for n in model_names}
        b = B
        while b > 0 and choice[b] != -1:
            i = choice[b]
            n = model_names[i]
            sel[n] += 1
            b -= icosts[n]
        best_val = float(dp[B])

    rows = []
    for n, cnt in sel.items():
        if cnt > 0:
            unit = catalog[n]
            rows.append({
                "model": n,
                "units": cnt,
                "unit_price_usd": unit.unit_price_usd,
                "capex_usd": unit.unit_price_usd * cnt,
                "ths_total": unit.hashrate_ths * cnt,
                "kw_total": unit.power_kw * cnt,
                "value_ebitda_h{0}_usd".format(horizon_months): per_unit_monthly_ebitda_series(
                    unit, horizon_months, network_ths_start, base_env.monthly_network_growth_pct,
                    btc_price_start if base_env.btc_price_override is None else base_env.btc_price_override,
                    base_env.btc_price_monthly_growth_pct,
                    (avg_fees_btc if base_env.avg_fees_per_block_btc_override is None else base_env.avg_fees_per_block_btc_override) or 0.0,
                    base_env.uptime_pct, base_env.pue, base_env.variable_energy_usd_per_kwh,
                    price_curve_usd_per_kwh, halving_date, subsidy_before, subsidy_after
                ).sum()
            })
    res_df = pd.DataFrame(rows).sort_values("capex_usd", ascending=False) if rows else pd.DataFrame()
    return sel, best_val, res_df

# -----------------------------
# UI — Global controls
# -----------------------------

st.title("⚡ DESMO Bitcoin Mining Optimizer")   
st.caption("Model mining economics, compare scenarios, schedule future steps and optimize ASIC budget.")

top_cols = st.columns([1,1,2])
mode = top_cols[0].radio(
    "Mode",
    ["Classica", "Prossimi Step", "Hosting", "Monitor"],  # <-- aggiunto "Monitor"
    index=0,
    horizontal=True
)


# Halving controls (data stimata in base all'altezza attuale → 10 min/blocco)
with top_cols[1]:
    st.write("Halving (3.125 → 1.5625 BTC)")

    HALVING_HEIGHT = 1_050_000
    curr_height_for_halving = fetch_block_height()  
    if curr_height_for_halving is not None:
        blocks_left = max(0, HALVING_HEIGHT - curr_height_for_halving)
        est_halving_dt_utc = datetime.utcnow() + timedelta(minutes=blocks_left * 10)
        default_halving_date = est_halving_dt_utc.date()

    halving_date_input = st.date_input("Data halving", value=default_halving_date)
    subsidy_before = 3.125
    subsidy_after  = 1.5625

# Live network data & energy pricing
with st.sidebar:
    st.subheader("Live Network Data")
    with st.spinner("Fetching network data..."):
        price_usd = fetch_btc_price_usd()
        diff = fetch_difficulty()
        height = fetch_block_height()
        live_subsidy_now = fetch_block_reward_btc()
        avg_fees = fetch_avg_fees_per_block_btc_7d_latest()
        avg_fees_1k = fetch_avg_fees_last_n_blocks_btc(1000)
        net_ths = difficulty_to_network_hashrate_ths(diff) if diff else float("nan")

with st.sidebar:
    st.subheader("Debug Antpool secrets")
    st.write("USER_ID set:", bool(ANTPOOL_USER_ID))
    st.write("API_KEY set:", bool(ANTPOOL_API_KEY))
    st.write("API_SECRET set:", bool(ANTPOOL_API_SECRET))

    # Facoltativo: mostra solo i primi caratteri, così controlli che siano quelli giusti
    st.write("USER_ID (mask):", ANTPOOL_USER_ID[:4] + "…" if ANTPOOL_USER_ID else "None")
    st.write("API_KEY (mask):", ANTPOOL_API_KEY[:4] + "…" if ANTPOOL_API_KEY else "None")


with st.sidebar:
    st.divider()
    st.subheader("Antpool (desmo-test)")

    overview = fetch_antpool_overview()
    if overview is None:
        st.caption("Impossibile leggere i dati da Antpool (controlla chiavi / permessi).")
    else:
        # I campi sono stringhe enormi di H/s. Convertiamo a TH/s:
        def to_ths(value: str) -> float:
            try:
                return float(value) / 1e12
            except Exception:
                return float("nan")

        hs_10m_ths = to_ths(overview.get("hsLast10m", "0"))
        hs_1h_ths  = to_ths(overview.get("hsLast1h", "0"))
        hs_1d_ths  = to_ths(overview.get("hsLast1d", "0"))

        c1, c2, c3 = st.columns(3)
        c1.metric("TH/s (10 min)", f"{hs_10m_ths:,.0f}")
        c2.metric("TH/s (1h)",     f"{hs_1h_ths:,.0f}")
        c3.metric("TH/s (1d)",     f"{hs_1d_ths:,.0f}")

        st.caption(f"Worker attivi: {overview.get('activeWorkerNum', 'N/A')} / "
                   f"{overview.get('totalWorkerNum', 'N/A')}")


    col1, col2 = st.columns(2)
    with col1:
        st.metric("BTC Spot (USD)", f"{price_usd:,.0f}" if price_usd==price_usd else "—")
        st.metric("Block Reward (BTC, now)", f"{live_subsidy_now:.3f}" if live_subsidy_now else "—")
        if avg_fees_1k is not None:
            avg_1k_btc, start_b, end_b = avg_fees_1k
            st.metric("Avg Fees / Block (BTC, last 7d)", f"{avg_1k_btc:.8f}")
        else:
            st.metric("Avg Fees / Block (BTC, last 7d)", "—")
            st.caption("N/D — impossibile calcolare la media ultimi 1000 blocchi")

    with col2:
        st.metric("Height", f"{height:,}" if height else "—")
        st.metric("Difficulty", f"{diff:,.0f}" if diff else "—")
        st.metric("Network Hashrate (TH/s)", f"{net_ths:,.0f}" if net_ths==net_ths else "—")
            
    st.divider()
    st.subheader("Energy Pricing")

    price_source = st.radio(
        "Fonte prezzo energia",
        ["Flat", "ERCOT RTM (Grid Status API)"],
        index=0, horizontal=True, key="price_source",
    )
    
    ercot_price = None
    if price_source == "ERCOT RTM (Grid Status API)":
        GRIDSTATUS_API_KEY = st.secrets.get("GRIDSTATUS_API_KEY", os.getenv("GRIDSTATUS_API_KEY"))

        if not GRIDSTATUS_API_KEY:
            st.warning(
                "⚠️ Nessuna API key trovata. Aggiungi `GRIDSTATUS_API_KEY` in `.streamlit/secrets.toml` "
                "oppure come variabile d'ambiente per usare l’ERCOT RTM."
            )
        else:
            location = st.selectbox(
                "Preset location",
                ["PIONR_DJ_RN","LZ_WEST","LZ_NORTH","LZ_SOUTH","LZ_HOUSTON",
                "NORTH_HUB","SOUTH_HUB","WEST_HUB","HOUSTON_HUB",
                "HB_HOUSTON","HB_NORTH","HB_SOUTH","HB_WEST"],
                index=0,
                key="ercot_location"
            )
            with st.spinner("Recupero prezzo ERCOT RTM…"):
                ercot_price = fetch_ercot_rtm_price_per_kwh_api(GRIDSTATUS_API_KEY, location)

            st.metric("ERCOT RTM (live) $/kWh", f"{ercot_price:.5f}" if ercot_price is not None else "—")
                    # --- Salvataggio automatico ogni ora ---
            now = datetime.utcnow()
            if ercot_price is not None:
                last_saved = st.session_state.get("last_ercot_save")
                if last_saved is None or (now - last_saved).total_seconds() >= 3600:
                    save_ercot_price(location, float(ercot_price))
                    st.session_state["last_ercot_save"] = now

            st.caption("Fonte: Grid Status API — ultimo SCED (5-min).")

    flat_default = float(ercot_price) if (ercot_price is not None) else 0.05
    ui_disabled = (price_source == "ERCOT RTM (Grid Status API)")

    # valore iniziale mostrato nell'input (se bloccato e c'è ERCOT → mostra ERCOT)
    init_value = float(st.session_state.get("flat_price_cached", flat_default))
    if ui_disabled and (ercot_price is not None):
        init_value = float(ercot_price)

    flat_price = st.number_input(
        "Flat $/kWh",
        min_value=0.0, step=0.001,
        value=init_value,
        format="%.3f",
        key="flat_price_input",
        disabled=ui_disabled,
    )

    # se è selezionato ERCOT, forziamo il valore usato nei calcoli al live
    if ui_disabled and (ercot_price is not None):
        flat_price = float(ercot_price)

    # aggiorna la cache per i prossimi rerun
    st.session_state["flat_price_cached"] = float(flat_price)


    uploaded_csv = st.file_uploader(
        "Curva oraria opzionale CSV (colonna 'price_usd_per_kwh')", type=["csv"]
    )
    price_curve = None
    if uploaded_csv is not None:
        try:
            curve_df = pd.read_csv(uploaded_csv)
            if "price_usd_per_kwh" in curve_df.columns:
                price_curve = curve_df["price_usd_per_kwh"].astype(float).values
                st.success(f"Loaded {len(price_curve)} hourly prices.")
            else:
                st.error("CSV must include a 'price_usd_per_kwh' column.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    
    
    # --- storico 24h via API live ---
    if price_source == "ERCOT RTM (Grid Status API)":
        if st.checkbox("📊 Mostra ultime 24h (API live)"):
            df_24h = fetch_ercot_last_24h(GRIDSTATUS_API_KEY, location)
            if not df_24h.empty:
                st.dataframe(df_24h)
                fig24 = go.Figure()
                fig24.add_trace(go.Scatter(
                    x=df_24h["timestamp"],
                    y=df_24h["price_usd_per_kwh"],
                    mode="lines+markers"
                ))
                fig24.update_layout(
                    title=f"Prezzi ERCOT ultime 24h ({location})",
                    xaxis_title="Ora",
                    yaxis_title="$/kWh"
                )
                st.plotly_chart(fig24, use_container_width=True)
                avg_24h = df_24h["price_usd_per_kwh"].mean()
                st.metric("Media ultime 24h ERCOT $/kWh", f"{avg_24h:.5f}")
    
    



# -----------------------------
# Catalog editor
# -----------------------------
st.subheader("1) Catalog & Fleet")
with st.expander("ASIC Catalog (editable)", expanded=False):
    cat_df = pd.DataFrame({k: asdict(v) for k, v in DEFAULT_CATALOG.items()}).T
    edited = st.data_editor(cat_df, num_rows="dynamic")
    catalog: Dict[str, AsicModel] = {}
    for name, row in edited.iterrows():
        try:
            catalog[name] = AsicModel(
                name=name,
                hashrate_ths=float(row["hashrate_ths"]),
                power_kw=float(row["power_kw"]),
                unit_price_usd=float(row["unit_price_usd"]),
                disponibility=str(row.get("disponibility", "now"))  # <- QUI
            )
        except Exception:
            pass

# -----------------------------
# Scenari — Modalità Classica
# -----------------------------
if mode == "Classica":
    st.subheader("2) Scenarios")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios: List[Scenario] = [] # type: ignore
    
    with st.expander("2) Fleet (units per model)", expanded=False):
        fleet_counts_classic: Dict[str, int] = {}
        fleet_cols = st.columns(3)
        for i, model_name in enumerate(catalog.keys()):
            key = f"fleet_classic_{model_name}"
            val = fleet_cols[i % 3].number_input(
                model_name, min_value=0, step=1, value=st.session_state.get(key, 0), key=key
            )
            fleet_counts_classic[model_name] = int(val)


        # Real-time ASIC budget & live metrics
        def _fleet_sum(catalog, counts):
            ths = sum(catalog[m].hashrate_ths * n for m, n in counts.items() if m in catalog)
            kw = sum(catalog[m].power_kw * n for m, n in counts.items() if m in catalog)
            cap = sum(catalog[m].unit_price_usd * n for m, n in counts.items() if m in catalog)
            return float(ths), float(kw), float(cap)

        live_ths, live_kw, live_asic_capex = _fleet_sum(catalog, fleet_counts_classic)

        calc_cols = st.columns(4)
        calc_cols[0].metric("ASIC CAPEX stimato", f"${live_asic_capex:,.0f}")
        calc_cols[1].metric("Fleet TH/s (selezionata)", f"{live_ths:,.0f}")
        calc_cols[2].metric("IT kW (stimati)", f"{live_kw:,.0f}")
        calc_cols[3].metric("$ per TH", f"${(live_asic_capex/live_ths):,.2f}" if live_ths > 0 else "—")

        budget_classic = st.number_input("Budget ASICs (opzionale)", min_value=0.0, step=1000.0, value=st.session_state.get("budget_classic", 0.0), key="budget_classic")
        if budget_classic and budget_classic > 0:
            delta = budget_classic - live_asic_capex
            if delta >= 0:
                st.success(f"✅ Entro budget: residuo ${delta:,.0f}")
            else:
                st.error(f"⚠️ Fuori budget: mancano ${-delta:,.0f}")

  # --- Form for the rest + submit ---
    st.subheader("3) Parametri")
    with st.expander("3) Parametri scenario", expanded=False):
        with st.form("new_scenario"):
            cols = st.columns(3)
            name = cols[0].text_input("Name", value=f"Scenario {len(st.session_state.scenarios)+1}")
            pue = cols[1].number_input("PUE", min_value=1.0, max_value=2.0, value=1.08, step=0.01)
            uptime_pct = cols[2].number_input("Uptime %", min_value=0.0, max_value=100.0, value=97.0, step=0.5)

            st.markdown("**Costs (USD)**")
            c1, c2, c3, c4 = st.columns(4)
            fixed_opex = c1.number_input("Fixed OPEX / month", min_value=0.0, step=100.0, value=13950.0)
            var_price = c2.number_input("Variable $/kWh (scenario override)", min_value=0.0, step=0.001, value=float(flat_price), format="%.3f")
            capex_asics = c3.number_input("CAPEX ASICs (0 = compute from catalog)", min_value=0.0, step=1000.0, value=0.0)
            capex_container = c4.number_input("CAPEX Containers", min_value=0.0, step=1000.0, value=60000.0)

            c5, c6, c7, c8 = st.columns(4)
            capex_transformer = c5.number_input("CAPEX Transformer", min_value=0.0, step=1000.0, value=50000.0)
            other_capex = c6.number_input("Other CAPEX", min_value=0.0, step=1000.0, value=140_000.0)
            btc_price_override = c7.number_input("BTC price override (0 = live path)", min_value=0.0, step=1000.0, value=0.0)
            default_avg_fees_btc = float(avg_fees_1k[0]) if avg_fees_1k is not None else float(avg_fees or 0.0)

            avg_fees_override = c8.number_input(
                "Avg fees per block BTC (valore live)",
                min_value=0.0,
                step=0.00000001,
                value=default_avg_fees_btc,
                format="%.8f",
                key="avg_fees_override_classic",
            )

            c9, c10, c11 = st.columns(3)
            monthly_net_growth = c9.number_input("Network hashrate growth % / month", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
            btc_price_mom = c10.number_input("BTC price growth % / month", min_value=-50.0, max_value=100.0, value=0.0, step=0.1)
            months_horizon = int(c11.number_input("Months horizon", min_value=6, max_value=120, value=60, step=6))
            public_box = st.checkbox("Salva scenario come pubblico (visibile a tutti)", value=False, key="save_public_classic")
            author_name = st.text_input("Autore (facoltativo)", value="", key="author_classic")

            submitted = st.form_submit_button("➕ Add scenario")


    if submitted:
        scn = Scenario(
            name=name,
            fleet={k:int(v) for k,v in fleet_counts_classic.items() if int(v)>0},
            pue=float(pue),
            uptime_pct=float(uptime_pct),
            fixed_opex_month_usd=float(fixed_opex),
            variable_energy_usd_per_kwh=float(var_price),
            capex_asics_usd=float(capex_asics),
            capex_container_usd=float(capex_container),
            capex_transformer_usd=float(capex_transformer),
            other_capex_usd=float(other_capex),
            btc_price_override=float(btc_price_override) if btc_price_override>0 else None,
            avg_fees_per_block_btc_override=float(avg_fees_override) if avg_fees_override>0 else None,
            monthly_network_growth_pct=float(monthly_net_growth),
            btc_price_monthly_growth_pct=float(btc_price_mom),
            months_horizon=int(months_horizon),
        )
        st.session_state.scenarios.append(scn)
        st.success(f"Added {scn.name} ✅")

        if public_box:
            author = author_name.strip() or "anonymous"
            payload = scenario_to_public_dict(scn, catalog, author=author)
            save_public_scenario("classica", payload)
            st.success("Scenario pubblicato ✅")

    if not st.session_state.scenarios:
        st.info("Add one or more scenarios to begin.")
    else:
        live_btc_price = price_usd
        live_avg_fees = avg_fees if avg_fees is not None else 0.0
        live_network_ths = net_ths

        tabs = st.tabs([s.name for s in st.session_state.scenarios] + ["📊 Compare"])

        dfs = []
        for idx, scn in enumerate(st.session_state.scenarios):
            with tabs[idx]:
                btc_price = scn.btc_price_override or live_btc_price
                fees_block = scn.avg_fees_per_block_btc_override if scn.avg_fees_per_block_btc_override is not None else live_avg_fees
                use_live_ercot = (price_source == "ERCOT RTM (Grid Status API)") and (ercot_price is not None)
                energy_flat_for_sim = ercot_price if use_live_ercot else scn.variable_energy_usd_per_kwh


                df = simulate_scenario(
                    scn,
                    catalog=catalog,
                    network_ths_start=live_network_ths,
                    avg_fees_btc=fees_block,
                    btc_price_usd_start=btc_price,
                    flat_price_usd_per_kwh=energy_flat_for_sim, 
                    price_curve_usd_per_kwh=price_curve,
                    halving_date=halving_date_input,
                    subsidy_before=subsidy_before,
                    subsidy_after=subsidy_after,
                )
                dfs.append((scn, df))

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Fleet TH/s (t0)", f"{df.attrs['fleet_ths']:,.0f}")
                k2.metric("IT Power kW (t0)", f"{df.attrs['it_power_kw']:,.0f}")
                k3.metric("Total CAPEX", f"${df.attrs['total_capex_usd']:,.0f}")
                k4.metric("Payback (months)", df.attrs["payback_months"] if df.attrs["payback_months"] else "—")

                st.dataframe(
                    df[["month","btc_price_usd","subsidy_btc","btc_month","rev_usd","energy_cost_usd","fixed_opex_usd","ebitda_usd","cashflow_usd","cum_cashflow_usd"]],
                    key=f"df_classic_{idx}"
                )

                # Charts
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df["month"], y=df["cashflow_usd"], name="Monthly Cashflow"))
                fig.add_trace(go.Scatter(x=df["month"], y=df["cum_cashflow_usd"], name="Cumulative", mode="lines+markers"))
                fig.update_layout(title=f"Cashflow — {scn.name}", xaxis_title="Month", yaxis_title="USD", barmode="group")
                st.plotly_chart(fig, use_container_width=True, key=f"cf_classic_{idx}")

                # Support charts
                c1, c2, c3 = st.columns(3)
                with c1:
                    f1 = go.Figure()
                    f1.add_trace(go.Bar(x=df["month"], y=df["btc_month"], name="BTC / month"))
                    f1.update_layout(title="BTC produced", xaxis_title="Month", yaxis_title="BTC")
                    st.plotly_chart(f1, use_container_width=True, key=f"btc_classic_{idx}")
                with c2:
                    f2 = go.Figure()
                    f2.add_trace(go.Scatter(x=df["month"], y=df["fleet_ths"], mode="lines+markers", name="Fleet TH/s"))
                    f2.update_layout(title="Fleet TH/s over time", xaxis_title="Month", yaxis_title="TH/s")
                    st.plotly_chart(f2, use_container_width=True, key=f"ths_classic_{idx}")
                with c3:
                    f3 = go.Figure()
                    f3.add_trace(go.Scatter(x=df["month"], y=df["it_power_kw"], mode="lines+markers", name="IT kW"))
                    f3.update_layout(title="IT Power (kW) over time", xaxis_title="Month", yaxis_title="kW")
                    st.plotly_chart(f3, use_container_width=True, key=f"kw_classic_{idx}")

        # Comparison tab
        with tabs[-1]:
            comp_rows = []
            for scn, df in dfs:
                comp_rows.append({
                    "Scenario": scn.name,
                    "Fleet TH/s (t0)": df.attrs['fleet_ths'],
                    "IT kW (t0)": df.attrs['it_power_kw'],
                    "Total CAPEX $": df.attrs['total_capex_usd'],
                    "Payback months": df.attrs['payback_months'] or np.nan,
                    "Year-1 EBITDA $": float(df.loc[:11, "ebitda_usd"].sum()),
                    "Year-2 EBITDA $": float(df.loc[12:23, "ebitda_usd"].sum()) if len(df) >= 24 else np.nan,
                    "Year-3 EBITDA $": float(df.loc[24:35, "ebitda_usd"].sum()) if len(df) >= 36 else np.nan,
                    "Cum CF @ 36m $": float(df.loc[min(len(df)-1,35), "cum_cashflow_usd"]) if len(df) >= 12 else float(df.iloc[-1]["cum_cashflow_usd"]) ,
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, key="df_compare_classic")

            best = comp_df.sort_values(by=["Cum CF @ 36m $"], ascending=False).head(1)
            if not best.empty:
                nameb = best.iloc[0]["Scenario"]
                st.success(f"🏆 Best 36m cumulative cashflow: **{nameb}**")

        with st.expander("📚 Scenari pubblici (Classica)"):
            pub = list_public_scenarios("classica")
            if not pub:
                st.caption("Nessuno scenario pubblico salvato.")
            else:
                df_pub = pd.DataFrame(pub)
                st.dataframe(df_pub)

                # bottone per cancellare tutto
                if st.button("❌ Elimina tutti gli scenari pubblici (Classica)"):
                    clear_public_scenarios("classica")
                    st.success("Scenari pubblici (Classica) eliminati.")
                    st.rerun()

                # dropdown per cancellare singolo scenario
                idx_to_delete = st.selectbox(
                    "Seleziona scenario da eliminare",
                    options=list(range(len(pub))),
                    format_func=lambda i: f"{i} — {pub[i].get('name','?')}",
                    key="del_classica"
                )
                if st.button("Elimina scenario selezionato (Classica)"):
                    if delete_public_scenario("classica", idx_to_delete):
                        st.success("Scenario eliminato.")
                        st.rerun()

# -----------------------------
# Modalità Prossimi Step
# -----------------------------
elif mode == "Prossimi Step":
    st.subheader("2) Base Scenarios (t0)")

    if "scenarios_ns" not in st.session_state:
        st.session_state.scenarios_ns: List[Scenario] = [] # type: ignore
    if "future_steps" not in st.session_state:
        st.session_state.future_steps: List[FutureStep] = [] # type: ignore

    # --- Fleet grid OUTSIDE the form for real-time budget calc (t0) ---
    with st.expander("2) Fleet (units per model) — t0", expanded=False):
        fleet_counts_ns: Dict[str, int] = {}
        fcols_ns = st.columns(3)
        for i, model_name in enumerate(catalog.keys()):
            key = f"fleet_ns_{model_name}"
            val = fcols_ns[i % 3].number_input(
                model_name, min_value=0, step=1, value=st.session_state.get(key, 0), key=key
            )
            fleet_counts_ns[model_name] = int(val)

        live_ths_ns, live_kw_ns, live_asic_capex_ns = ( 
            sum(catalog[m].hashrate_ths * n for m, n in fleet_counts_ns.items() if m in catalog),
            sum(catalog[m].power_kw * n for m, n in fleet_counts_ns.items() if m in catalog),
            sum(catalog[m].unit_price_usd * n for m, n in fleet_counts_ns.items() if m in catalog),
        )

        calc_ns = st.columns(4)
        calc_ns[0].metric("ASIC CAPEX stimato (t0)", f"${live_asic_capex_ns:,.0f}")
        calc_ns[1].metric("Fleet TH/s (t0)", f"{live_ths_ns:,.0f}")
        calc_ns[2].metric("IT kW (t0)", f"{live_kw_ns:,.0f}")
        calc_ns[3].metric("$ per TH (t0)", f"${(live_asic_capex_ns/live_ths_ns):,.2f}" if live_ths_ns>0 else "—")

        budget_ns = st.number_input("Budget ASICs t0 (opzionale)", min_value=0.0, step=1000.0, value=st.session_state.get("budget_ns", 0.0), key="budget_ns")
        if budget_ns and budget_ns > 0:
            delta = budget_ns - live_asic_capex_ns
            if delta >= 0:
                st.success(f"✅ Entro budget t0: residuo ${delta:,.0f}")
            else:
                st.error(f"⚠️ Fuori budget t0: mancano ${-delta:,.0f}")

    # --- Base scenario form ---
    st.subheader("3) Parametri")
    with st.expander("3) Parametri scenario base (t0)", expanded=False):
        with st.form("new_scenario_ns"):
            st.markdown("**Add base scenario (t0)**")
            cols = st.columns(3)
            name = cols[0].text_input("Name", value=f"Plan {len(st.session_state.scenarios_ns)+1}")
            pue = cols[1].number_input("PUE", min_value=1.0, max_value=2.0, value=1.08, step=0.01, key="pue_ns")
            uptime_pct = cols[2].number_input("Uptime %", min_value=0.0, max_value=100.0, value=97.0, step=0.5, key="uptime_ns")

            st.markdown("**Costs (USD)**")
            c1, c2, c3, c4 = st.columns(4)
            fixed_opex = c1.number_input("Fixed OPEX / month", min_value=0.0, step=100.0, value=13950.0, key="fop_ns")
            var_price = c2.number_input("Variable $/kWh (scenario override)", min_value=0.0, step=0.001, value=float(flat_price), format="%.3f", key="varp_ns")
            capex_asics = c3.number_input("CAPEX ASICs (0 = compute from catalog)", min_value=0.0, step=1000.0, value=0.0, key="ascc_ns")
            capex_container = c4.number_input("CAPEX Containers", min_value=0.0, step=1000.0, value=60000.0, key="cont_ns")

            c5, c6, c7, c8 = st.columns(4)
            capex_transformer = c5.number_input("CAPEX Transformer", min_value=0.0, step=1000.0, value=50000.0, key="trf_ns")
            other_capex = c6.number_input("Other CAPEX", min_value=0.0, step=1000.0, value=140_000.0, key="oth_ns")
            btc_price_override = c7.number_input("BTC price override (0 = live path)", min_value=0.0, step=1000.0, value=0.0, key="btc_ns")
            default_avg_fees_btc = float(avg_fees_1k[0]) if avg_fees_1k is not None else float(avg_fees or 0.0)

            avg_fees_override = c8.number_input(
                "Avg fees per block BTC (valore live)",
                min_value=0.0,
                step=0.00000001,
                value=default_avg_fees_btc,
                format="%.8f",
                key="avg_fees_override_classic",
            )

            c9, c10, c11 = st.columns(3)
            monthly_net_growth = c9.number_input("Network hashrate growth % / month", min_value=-50.0, max_value=50.0, value=0.0, step=0.1, key="netg_ns")
            btc_price_mom = c10.number_input("BTC price growth % / month", min_value=-50.0, max_value=100.0, value=0.0, step=0.1, key="prg_ns")
            months_horizon = int(c11.number_input("Months horizon", min_value=6, max_value=120, value=60, step=6, key="hor_ns"))
            public_box_ns = st.checkbox("Salva base scenario t0 come pubblico", value=False, key="save_public_ns")
            author_name_ns = st.text_input("Autore (facoltativo)", value="", key="author_ns")

            submitted = st.form_submit_button("➕ Add base scenario (t0)")

    if submitted:
        scn = Scenario(
            name=name,
            fleet={k:int(v) for k,v in fleet_counts_ns.items() if int(v)>0},
            pue=float(pue),
            uptime_pct=float(uptime_pct),
            fixed_opex_month_usd=float(fixed_opex),
            variable_energy_usd_per_kwh=float(var_price),
            capex_asics_usd=float(capex_asics),
            capex_container_usd=float(capex_container),
            capex_transformer_usd=float(capex_transformer),
            other_capex_usd=float(other_capex),
            btc_price_override=float(btc_price_override) if btc_price_override>0 else None,
            avg_fees_per_block_btc_override=float(avg_fees_override) if avg_fees_override>0 else None,
            monthly_network_growth_pct=float(monthly_net_growth),
            btc_price_monthly_growth_pct=float(btc_price_mom),
            months_horizon=int(months_horizon),
        )
        st.session_state.scenarios_ns.append(scn)
        st.success(f"Added {scn.name} ✅")

        if public_box_ns:
            author = author_name_ns.strip() or "anonymous"
            payload = scenario_to_public_dict(scn, catalog, author=author)
            save_public_scenario("prossimi_step", payload)
            st.success("Scenario t0 pubblicato ✅")

    if not st.session_state.scenarios_ns:
        st.info("Aggiungi almeno uno scenario base t0 per definire i *Prossimi Step* qui sotto.")
    else:
        st.subheader("3) Future Steps (applicati da t+N)")
        st.caption("Gli step **aggiungono** fleet e CAPEX nel mese di applicazione; eventuali override si applicano da quel mese in poi.")

        # --- Fleet grid OUTSIDE the form for real-time budget calc (step) ---
        st.markdown("**Fleet da aggiungere nello Step**")
        fleet_counts_step_live: Dict[str, int] = {}
        fcols_step = st.columns(3)
        for i, model_name in enumerate(catalog.keys()):
            key = f"step_{model_name}"
            val = fcols_step[i % 3].number_input(
                model_name, min_value=0, step=1, value=st.session_state.get(key, 0), key=key
            )
            fleet_counts_step_live[model_name] = int(val)

        live_ths_step, live_kw_step, live_asic_capex_step = (
            sum(catalog[m].hashrate_ths * n for m, n in fleet_counts_step_live.items() if m in catalog),
            sum(catalog[m].power_kw * n for m, n in fleet_counts_step_live.items() if m in catalog),
            sum(catalog[m].unit_price_usd * n for m, n in fleet_counts_step_live.items() if m in catalog),
        )
        calc_step = st.columns(4)
        calc_step[0].metric("ASIC CAPEX stimato (step)", f"${live_asic_capex_step:,.0f}")
        calc_step[1].metric("TH/s aggiunti (step)", f"{live_ths_step:,.0f}")
        calc_step[2].metric("IT kW aggiunti (step)", f"{live_kw_step:,.0f}")
        calc_step[3].metric("$ per TH (step)", f"${(live_asic_capex_step/live_ths_step):,.2f}" if live_ths_step>0 else "—")

        budget_step = st.number_input("Budget ASICs per step (opzionale)", min_value=0.0, step=1000.0, value=st.session_state.get("budget_step", 0.0), key="budget_step")
        if budget_step and budget_step > 0:
            delta = budget_step - live_asic_capex_step
            if delta >= 0:
                st.success(f"✅ Entro budget step: residuo ${delta:,.0f}")
            else:
                st.error(f"⚠️ Fuori budget step: mancano ${-delta:,.0f}")

        # --- Step form ---
        with st.expander("4) Future Steps (aggiunte + override)", expanded=False):
            with st.form("new_future_step"):
                scn_names = [s.name for s in st.session_state.scenarios_ns]
                target_scn = st.selectbox("Scenario target", scn_names)
                month_offset = st.number_input("Applica da mese (t+N)", min_value=1, max_value=240, value=4, step=1)

                st.markdown("**CAPEX dello step (USD)**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                s_capex_asics = cc1.number_input("CAPEX ASICs (0 = compute da catalogo)", min_value=0.0, step=1000.0, value=0.0, key="sc_asics")
                s_capex_container = cc2.number_input("CAPEX Containers", min_value=0.0, step=1000.0, value=0.0, key="sc_cont")
                s_capex_transformer = cc3.number_input("CAPEX Transformer", min_value=0.0, step=1000.0, value=0.0, key="sc_trf")
                s_other_capex = cc4.number_input("Other CAPEX", min_value=0.0, step=1000.0, value=0.0, key="sc_other")

                st.markdown("**Override opzionali (lascia vuoto = eredita)**")
                oo1, oo2, oo3 = st.columns(3)
                s_pue = oo1.text_input("PUE (es. 1.08)", value="")
                s_uptime = oo2.text_input("Uptime % (es. 97)", value="")
                s_var_price = oo3.text_input("Variable $/kWh (es. 0.05)", value="")
                oo4, oo5, oo6 = st.columns(3)
                s_fixed_opex = oo4.text_input("Fixed OPEX / month", value="")
                s_btc_override = oo5.text_input("BTC price override", value="")
                s_fees_override = oo6.text_input("Avg fees / block BTC", value="")
                oo7, oo8 = st.columns(2)
                s_nh_growth = oo7.text_input("Network growth %/m", value="")
                s_price_growth = oo8.text_input("BTC price growth %/m", value="")

                submitted_step = st.form_submit_button("➕ Add future step")

        if submitted_step:
            def opt_float(x: str) -> Optional[float]:
                x = x.strip()
                return float(x) if x not in ("", None) else None
            fleet_counts_step = {k:int(v) for k,v in fleet_counts_step_live.items() if int(v)>0}
            step = FutureStep(
                name=f"{target_scn} @ t+{int(month_offset)}",
                scenario_name=target_scn,
                month_offset=int(month_offset),
                fleet=fleet_counts_step,
                pue=opt_float(s_pue),
                uptime_pct=opt_float(s_uptime),
                fixed_opex_month_usd=opt_float(s_fixed_opex),
                variable_energy_usd_per_kwh=opt_float(s_var_price),
                capex_asics_usd=float(s_capex_asics),
                capex_container_usd=float(s_capex_container),
                capex_transformer_usd=float(s_capex_transformer),
                other_capex_usd=float(s_other_capex),
                btc_price_override=opt_float(s_btc_override),
                avg_fees_per_block_btc_override=opt_float(s_fees_override),
                monthly_network_growth_pct=opt_float(s_nh_growth),
                btc_price_monthly_growth_pct=opt_float(s_price_growth),
            )
            st.session_state.future_steps.append(step)
            st.success(f"Added step: {step.name} ✅")

            if st.checkbox("Pubblica anche questo Step", value=False, key=f"pub_step_{target_scn}_{month_offset}"):
                save_public_scenario("prossimi_step", step_to_public_dict(step))
                st.success("Step pubblicato ✅")

        # Show per-scenario tabs with steps
        tabs = st.tabs([s.name for s in st.session_state.scenarios_ns] + ["📊 Compare"])
        dfs = []
        live_btc_price = price_usd
        live_avg_fees = avg_fees if avg_fees is not None else 0.0
        live_network_ths = net_ths

        for idx, scn in enumerate(st.session_state.scenarios_ns):
            with tabs[idx]:
                fees_block = scn.avg_fees_per_block_btc_override if scn.avg_fees_per_block_btc_override is not None else live_avg_fees
                btc_price0 = scn.btc_price_override or live_btc_price

                df = simulate_scenario_with_steps(
                    scn,
                    steps=st.session_state.future_steps,
                    catalog=catalog,
                    network_ths_start=live_network_ths,
                    avg_fees_btc=fees_block,
                    btc_price_usd_start=btc_price0,
                    flat_price_usd_per_kwh_global=flat_price,
                    price_curve_usd_per_kwh=price_curve,
                    halving_date=halving_date_input,
                    subsidy_before=subsidy_before,
                    subsidy_after=subsidy_after,
                )
                dfs.append((scn, df))

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Fleet TH/s (final)", f"{df.attrs['fleet_ths']:,.0f}")
                k2.metric("IT Power kW (final)", f"{df.attrs['it_power_kw']:,.0f}")
                k3.metric("Total CAPEX (t0 + steps)", f"${df.attrs['total_capex_usd']:,.0f}")
                k4.metric("Payback (months)", df.attrs["payback_months"] if df.attrs["payback_months"] else "—")

                st.dataframe(
                    df[["month","btc_price_usd","subsidy_btc","btc_month","rev_usd","energy_cost_usd","fixed_opex_usd","ebitda_usd","cashflow_usd","cum_cashflow_usd"]],
                    key=f"df_steps_{idx}"
                )
                
                # Charts
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df["month"], y=df["cashflow_usd"], name="Monthly Cashflow"))
                fig.add_trace(go.Scatter(x=df["month"], y=df["cum_cashflow_usd"], name="Cumulative", mode="lines+markers"))
                fig.update_layout(title=f"Cashflow — {scn.name} (with Steps)", xaxis_title="Month", yaxis_title="USD", barmode="group")
                st.plotly_chart(fig, use_container_width=True, key=f"cf_steps_{idx}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    f1 = go.Figure()
                    f1.add_trace(go.Bar(x=df["month"], y=df["btc_month"], name="BTC / month"))
                    f1.update_layout(title="BTC produced", xaxis_title="Month", yaxis_title="BTC")
                    st.plotly_chart(f1, use_container_width=True, key=f"btc_steps_{idx}")
                with c2:
                    f2 = go.Figure()
                    f2.add_trace(go.Scatter(x=df["month"], y=df["fleet_ths"], mode="lines+markers", name="Fleet TH/s"))
                    f2.update_layout(title="Fleet TH/s over time", xaxis_title="Month", yaxis_title="TH/s")
                    st.plotly_chart(f2, use_container_width=True, key=f"ths_steps_{idx}")
                with c3:
                    f3 = go.Figure()
                    f3.add_trace(go.Scatter(x=df["month"], y=df["it_power_kw"], mode="lines+markers", name="IT kW"))
                    f3.update_layout(title="IT Power (kW) over time", xaxis_title="Month", yaxis_title="kW")
                    st.plotly_chart(f3, use_container_width=True, key=f"kw_steps_{idx}")

        with tabs[-1]:
            comp_rows = []
            for scn, df in dfs:
                comp_rows.append({
                    "Scenario": scn.name,
                    "Total CAPEX $ (t0+steps)": df.attrs['total_capex_usd'],
                    "Payback months": df.attrs['payback_months'] or np.nan,
                    "Year-1 EBITDA $": float(df.loc[:11, "ebitda_usd"].sum()),
                    "Year-2 EBITDA $": float(df.loc[12:23, "ebitda_usd"].sum()) if len(df) >= 24 else np.nan,
                    "Year-3 EBITDA $": float(df.loc[24:35, "ebitda_usd"].sum()) if len(df) >= 36 else np.nan,
                    "Cum CF @ 36m $": float(df.loc[min(len(df)-1,35), "cum_cashflow_usd"]) if len(df) >= 12 else float(df.iloc[-1]["cum_cashflow_usd"]),
                    "Fleet TH/s (final)": float(df.iloc[-1]["fleet_ths"]) if len(df) else 0.0,
                    "IT kW (final)": float(df.iloc[-1]["it_power_kw"]) if len(df) else 0.0,
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, key="df_compare_classic")

            best = comp_df.sort_values(by=["Cum CF @ 36m $"], ascending=False).head(1)
            if not best.empty:
                nameb = best.iloc[0]["Scenario"]
                st.success(f"🏆 Best 36m cumulative cashflow: **{nameb}**")

        with st.expander("📚 Scenari pubblici (Prossimi Step)"):
            pub = list_public_scenarios("prossimi_step")  # <-- minuscolo + underscore
            if not pub:
                st.caption("Nessuno scenario pubblico salvato.")
            else:
                df_pub = pd.DataFrame(pub)
                st.dataframe(df_pub)

                # bottone per cancellare tutto
                if st.button("❌ Elimina tutti gli scenari pubblici (Prossimi Step)"):
                    clear_public_scenarios("prossimi_step")  # <-- qui
                    st.success("Scenari pubblici (Prossimi Step) eliminati.")
                    st.rerun()

                # dropdown per cancellare singolo scenario
                idx_to_delete = st.selectbox(
                    "Seleziona scenario da eliminare",
                    options=list(range(len(pub))),
                    format_func=lambda i: f"{i} — {pub[i].get('name','?')}",
                    key="del_prossimi_step"  # (consiglio: niente spazi nei key)
                )
                if st.button("Elimina scenario selezionato (Prossimi Step)"):
                    if delete_public_scenario("prossimi_step", idx_to_delete):  # <-- qui
                        st.success("Scenario eliminato.")
                        st.rerun()


# -----------------------------
# Modalità Hosting
# -----------------------------
elif mode == "Hosting":
    st.subheader("2) Parametri Hosting")

    if "hosting_scenarios" not in st.session_state:
        st.session_state.hosting_scenarios: List[HostingScenario] = []  # type: ignore

    # --- Fleet hosted (units per model) ---
    with st.expander("2) Fleet ospitata (units per model)", expanded=False):
        st.markdown("**Fleet ospitata (units per model)**")
        fleet_counts_host: Dict[str, int] = {}
        hcols = st.columns(3)
        for i, model_name in enumerate(catalog.keys()):
            key = f"fleet_host_{model_name}"
            val = hcols[i % 3].number_input(
                model_name, min_value=0, step=1, value=st.session_state.get(key, 0), key=key
            )
            fleet_counts_host[model_name] = int(val)

        # Metriche live
        live_ths_h = sum(catalog[m].hashrate_ths * n for m, n in fleet_counts_host.items() if m in catalog)
        live_kw_h = sum(catalog[m].power_kw * n for m, n in fleet_counts_host.items() if m in catalog)
        live_asic_cost_h = sum(catalog[m].unit_price_usd * n for m, n in fleet_counts_host.items() if m in catalog)

        mcols = st.columns(4)
        mcols[0].metric("ASIC CAPEX (acquisto)", f"${live_asic_cost_h:,.0f}")
        mcols[1].metric("Fleet TH/s (ospitata)", f"{live_ths_h:,.0f}")
        mcols[2].metric("IT kW (ospitata)", f"{live_kw_h:,.0f}")
        mcols[3].metric("$ per TH (acquisto)", f"${(live_asic_cost_h/live_ths_h):,.2f}" if live_ths_h>0 else "—")

    st.subheader("3) Set vendita")
    with st.expander("**Prezzi di vendita ASIC (per modello)** — opzionale (default = costo acquisto)", expanded=False):
        st.caption("Compila uno dei due campi qui sotto per impostare TUTTI i prezzi di vendita...")

        c1, c2 = st.columns(2)
        perc_markup = c1.number_input("Markup % su prezzo di acquisto (es. 10 = +10%)",
                                    value=0.0, step=0.5, format="%.2f", key="sale_markup_pct")
        abs_markup = c2.number_input("Markup assoluto $/unit (es. 100 = +$100)",
                                    value=0.0, step=10.0, format="%.2f", key="sale_markup_abs")

        if not catalog:
            st.warning("Catalogo vuoto: compila almeno un modello nell’editor sopra.")
            sale_df_edit = pd.DataFrame(columns=["buy_unit_price_usd", "sale_unit_price_usd"])
        else:
            sale_table = []
            for name, mdl in catalog.items():
                buy = float(mdl.unit_price_usd)
                if abs_markup != 0.0:
                    sale = buy + float(abs_markup)
                elif perc_markup != 0.0:
                    sale = buy * (1.0 + float(perc_markup) / 100.0)
                else:
                    sale = buy

                sale_table.append({
                    "model": name,
                    "buy_unit_price_usd": buy,
                    "sale_unit_price_usd": float(sale)
                })

            sale_df = pd.DataFrame(sale_table)
            if "model" in sale_df.columns:
                sale_df = sale_df.set_index("model")

            sale_df_edit = st.data_editor(sale_df, num_rows="dynamic", key="sale_df_host")



    # --- Form scenario hosting ---
    st.subheader("4) Parametri")
    with st.expander("4) Parametri scenario Hosting", expanded=False):
        with st.form("new_hosting_scenario"):
            cols = st.columns(3)
            name = cols[0].text_input("Name", value=f"Hosting {len(st.session_state.hosting_scenarios)+1}")
            pue = cols[1].number_input("PUE", min_value=1.0, max_value=2.0, value=1.08, step=0.01, key="pue_host")
            uptime_pct = cols[2].number_input("Uptime %", min_value=0.0, max_value=100.0, value=97.0, step=0.5, key="up_host")

            st.markdown("**Energia (USD/kWh)**")
            e1, e2 = st.columns(2)
            our_energy = e1.number_input("Nostro costo $/kWh", min_value=0.0, step=0.001, value=float(st.session_state.get("flat_price_cached", 0.05)), format="%.3f")
            hosting_sell = e2.number_input("Prezzo Hosting $/kWh (vendita)", min_value=0.0, step=0.001, value=max(our_energy, 0.08), format="%.3f")

            st.markdown("**OPEX e CAPEX (USD)**")
            c1, c2, c3, c4 = st.columns(4)
            fixed_opex = c1.number_input("Fixed OPEX / month", min_value=0.0, step=100.0, value=13950.0, key="fop_host")
            capex_asics = c2.number_input("CAPEX ASICs (0 = calcola da catalogo)", min_value=0.0, step=1000.0, value=0.0, key="capexa_host")
            capex_container = c3.number_input("CAPEX Containers", min_value=0.0, step=1000.0, value=60000.0, key="capexc_host")
            capex_transformer = c4.number_input("CAPEX Transformer", min_value=0.0, step=1000.0, value=50000.0, key="capext_host")

            c5, c6 = st.columns(2)
            other_capex = c5.number_input("Other CAPEX", min_value=0.0, step=1000.0, value=140_000.0, key="capexo_host")
            commission_pct = c6.number_input("Commissione mining % su revenue cliente", min_value=0.0, max_value=100.0, step=0.5, value=1.0, key="comm_host")

            st.markdown("**Assunzioni finanziarie**")
            f1, f2, f3 = st.columns(3)
            btc_price_override = f1.number_input("BTC price override (0 = live path)", min_value=0.0, step=1000.0, value=0.0, key="btc_host")
            default_avg_fees_btc = float((avg_fees if avg_fees is not None else 0.0))
            avg_fees_override = f2.number_input("Avg fees per block BTC", min_value=0.0, step=0.00000001, value=default_avg_fees_btc, format="%.8f", key="fees_host")
            months_horizon = int(f3.number_input("Months horizon", min_value=6, max_value=120, value=60, step=6, key="hor_host"))
            public_box_host = st.checkbox("Salva scenario Hosting come pubblico", value=False, key="save_public_host")
            author_name_host = st.text_input("Autore (facoltativo)", value="", key="author_host")

            g1, g2 = st.columns(2)
            monthly_net_growth = g1.number_input("Network hashrate growth % / month", min_value=-50.0, max_value=50.0, value=0.0, step=0.1, key="netg_host")
            btc_price_mom = g2.number_input("BTC price growth % / month", min_value=-50.0, max_value=100.0, value=0.0, step=0.1, key="prg_host")

            submitted_host = st.form_submit_button("➕ Add hosting scenario")

    if submitted_host:
        # Costruisci dizionario prezzi vendita (override)
        sale_overrides: Dict[str, float] = {}
        try:
          for model_name, row in sale_df_edit.iterrows():
            default_price = catalog.get(model_name).unit_price_usd if model_name in catalog else 0.0
            sale_overrides[model_name] = float(row.get("sale_unit_price_usd", default_price))
        except Exception:
            pass

        scn_h = HostingScenario(
            name=name,
            fleet={k:int(v) for k,v in fleet_counts_host.items() if int(v)>0},
            pue=float(pue),
            uptime_pct=float(uptime_pct),
            fixed_opex_month_usd=float(fixed_opex),
            our_energy_usd_per_kwh=float(our_energy),
            hosting_sell_usd_per_kwh=float(hosting_sell),
            sale_price_overrides=sale_overrides,
            capex_asics_usd=float(capex_asics),
            capex_container_usd=float(capex_container),
            capex_transformer_usd=float(capex_transformer),
            other_capex_usd=float(other_capex),
            btc_price_override=float(btc_price_override) if btc_price_override>0 else None,
            avg_fees_per_block_btc_override=float(avg_fees_override) if avg_fees_override>0 else None,
            monthly_network_growth_pct=float(monthly_net_growth),
            btc_price_monthly_growth_pct=float(btc_price_mom),
            months_horizon=int(months_horizon),
            commission_hashrate_pct=float(commission_pct),
        )
        st.session_state.hosting_scenarios.append(scn_h)
        st.success(f"Added {scn_h.name} ✅")

        if public_box_host:
            author = author_name_host.strip() or "anonymous"
            payload = hosting_to_public_dict(scn_h, author=author)
            save_public_scenario("hosting", payload)
            st.success("Scenario Hosting pubblicato ✅")

    if not st.session_state.hosting_scenarios:
        st.info("Aggiungi almeno uno scenario Hosting.")
    else:
        tabs = st.tabs([s.name for s in st.session_state.hosting_scenarios] + ["📊 Compare"])
        dfs_h = []

        # Live data comuni
        live_btc_price = price_usd
        live_avg_fees = avg_fees if avg_fees is not None else 0.0
        live_network_ths = net_ths

        for idx, scn_h in enumerate(st.session_state.hosting_scenarios):
            with tabs[idx]:
                btc_p0 = scn_h.btc_price_override or live_btc_price
                fees0 = scn_h.avg_fees_per_block_btc_override if scn_h.avg_fees_per_block_btc_override is not None else live_avg_fees

                dfh = simulate_hosting_scenario(
                    scn=scn_h,
                    catalog=catalog,
                    network_ths_start=live_network_ths,
                    avg_fees_btc=fees0,
                    btc_price_usd_start=btc_p0,
                    price_curve_usd_per_kwh=price_curve,
                    halving_date=halving_date_input,
                    subsidy_before=subsidy_before,
                    subsidy_after=subsidy_after,
                )
                dfs_h.append((scn_h, dfh))

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Fleet TH/s (ospitata)", f"{dfh.attrs['fleet_ths']:,.0f}")
                k2.metric("IT Power kW", f"{dfh.attrs['it_power_kw']:,.0f}")
                k3.metric("Total CAPEX", f"${dfh.attrs['total_capex_usd']:,.0f}")
                k4.metric("Payback (months)", dfh.attrs["payback_months"] if dfh.attrs["payback_months"] else "—")

                # KPI aggiuntivi lato cliente
                ccli1, ccli2 = st.columns(2)
                ccli1.metric("Costo iniziale cliente $", f"${dfh.attrs.get('client_purchase_total_usd', 0.0):,.0f}")
                ccli2.metric("ROI Cliente (mesi)", dfh.attrs.get("roi_cliente_months") if dfh.attrs.get("roi_cliente_months") else "—")

                # Evidenzia il one-off markup
                st.metric("ASIC Markup (one-off @ t0)", f"${dfh.attrs['asic_markup_usd']:,.0f}")

                st.dataframe(
                    dfh[[
                        "month","btc_price_usd","subsidy_btc",
                        "kwh_equivalent",
                        "energy_revenue_usd","energy_cost_usd","energy_margin_usd",
                        "commission_usd",
                        "rev_usd","fixed_opex_usd","ebitda_usd",
                        "asic_markup_usd","cashflow_usd","cum_cashflow_usd"
                    ]],
                    key=f"df_host_{idx}"
                )

                # Charts
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dfh["month"], y=dfh["cashflow_usd"], name="Monthly Cashflow"))
                fig.add_trace(go.Scatter(x=dfh["month"], y=dfh["cum_cashflow_usd"], name="Cumulative", mode="lines+markers"))
                fig.update_layout(title=f"Cashflow — {scn_h.name} (Hosting)", xaxis_title="Month", yaxis_title="USD", barmode="group")
                st.plotly_chart(fig, use_container_width=True, key=f"cf_host_{idx}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    f1 = go.Figure()
                    f1.add_trace(go.Bar(x=dfh["month"], y=dfh["energy_margin_usd"], name="Energy margin"))
                    f1.update_layout(title="Margine energia (mensile)", xaxis_title="Month", yaxis_title="USD")
                    st.plotly_chart(f1, use_container_width=True, key=f"em_host_{idx}")
                with c2:
                    f2 = go.Figure()
                    f2.add_trace(go.Bar(x=dfh["month"], y=dfh["commission_usd"], name="Commission"))
                    f2.update_layout(title="Commissione mining (mensile)", xaxis_title="Month", yaxis_title="USD")
                    st.plotly_chart(f2, use_container_width=True, key=f"cm_host_{idx}")
                with c3:
                    f3 = go.Figure()
                    f3.add_trace(go.Scatter(x=dfh["month"], y=dfh["btc_price_usd"], mode="lines+markers", name="BTC Price"))
                    f3.update_layout(title="BTC Price path", xaxis_title="Month", yaxis_title="USD")
                    st.plotly_chart(f3, use_container_width=True, key=f"bp_host_{idx}")

        # Tab di confronto
        with tabs[-1]:
            comp_rows = []
            for scn_h, dfh in dfs_h:
                comp_rows.append({
                    "Scenario": scn_h.name,
                    "Fleet TH/s (ospitata)": dfh.attrs['fleet_ths'],
                    "IT kW": dfh.attrs['it_power_kw'],
                    "Total CAPEX $": dfh.attrs['total_capex_usd'],
                    "ASIC Markup one-off $": dfh.attrs['asic_markup_usd'],
                    "Payback months": dfh.attrs['payback_months'] or np.nan,
                    "Year-1 EBITDA $": float(dfh.loc[:11, "ebitda_usd"].sum()),
                    "Year-2 EBITDA $": float(dfh.loc[12:23, "ebitda_usd"].sum()) if len(dfh) >= 24 else np.nan,
                    "Year-3 EBITDA $": float(dfh.loc[24:35, "ebitda_usd"].sum()) if len(dfh) >= 36 else np.nan,
                    "Cum CF @ 36m $": float(dfh.loc[min(len(dfh)-1,35), "cum_cashflow_usd"]) if len(dfh) >= 12 else float(dfh.iloc[-1]["cum_cashflow_usd"]),
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, key="df_compare_hosting")

            best = comp_df.sort_values(by=["Cum CF @ 36m $"], ascending=False).head(1)
            if not best.empty:
                nameb = best.iloc[0]["Scenario"]
                st.success(f"🏆 Best 36m cumulative cashflow (hosting): **{nameb}**")
        with st.expander("📚 Scenari pubblici (hosting)"):
            pub = list_public_scenarios("hosting")
            if not pub:
                st.caption("Nessuno scenario pubblico salvato.")
            else:
                df_pub = pd.DataFrame(pub)
                st.dataframe(df_pub)

                # bottone per cancellare tutto
                if st.button("❌ Elimina tutti gli scenari pubblici (hosting)"):
                    clear_public_scenarios("hosting")
                    st.success("Scenari pubblici (hosting) eliminati.")
                    st.rerun()

                # dropdown per cancellare singolo scenario
                idx_to_delete = st.selectbox(
                    "Seleziona scenario da eliminare",
                    options=list(range(len(pub))),
                    format_func=lambda i: f"{i} — {pub[i].get('name','?')}",
                    key="del_hosting"
                )
                if st.button("Elimina scenario selezionato (hosting)"):
                    if delete_public_scenario("hosting", idx_to_delete):
                        st.success("Scenario eliminato.")
                        st.rerun()


                        # -----------------------------
# Modalità Monitor: Hashrate vs ERCOT
# -----------------------------
elif mode == "Monitor":
    st.subheader("📈 Hashrate pool vs prezzo ERCOT LZ_WEST")

    st.markdown(
        """
        Carica un CSV con l'hashrate aggregato della pool (Antpool, sommato su tutti gli account).
        
        **Formato consigliato CSV:**
        - `timestamp`: ISO 8601 (es. `2025-11-24T03:00:00Z` oppure `2025-11-24 03:00:00`)
        - una o più colonne numeriche con l'hashrate in **TH/s** (es. `account1_ths, account2_ths, ...`)
        
        Il tool somma tutte le colonne numeriche e mostra la serie in **PH/s**.
        """
    )

    # Upload hashrate pool (Antpool)
    hash_file = st.file_uploader(
        "Carica CSV hashrate pool (Antpool)",
        type=["csv"],
        key="hashrate_csv_monitor"
    )

    # Carichiamo storico ERCOT salvato in ercot_prices.json
    ercot_df = list_ercot_prices()
    if ercot_df.empty:
        st.warning("⚠️ Nessun dato ERCOT salvato in `ercot_prices.json`. Apri la modalità con ERCOT RTM attivo per iniziare a loggare i prezzi.")
        ercot_df = None
    else:
        # parse timestamp & filtra LZ_WEST
        ercot_df["timestamp"] = pd.to_datetime(ercot_df["timestamp"], utc=True, errors="coerce")
        ercot_df = ercot_df.dropna(subset=["timestamp"])
        # per sicurezza, filtra solo LZ_WEST (o quello che ti interessa)
        ercot_location = st.selectbox(
            "Location ERCOT da mostrare",
            sorted(ercot_df["location"].unique()),
            index=list(sorted(ercot_df["location"].unique())).index("LZ_WEST") if "LZ_WEST" in ercot_df["location"].unique() else 0,
            key="ercot_loc_monitor"
        )
        ercot_df = ercot_df[ercot_df["location"] == ercot_location].copy()
        ercot_df = ercot_df.set_index("timestamp").sort_index()

    if hash_file is None:
        st.info("⬆️ Carica il CSV con l'hashrate per vedere il grafico.")
        if ercot_df is not None:
            with st.expander("👉 Vedi solo storico ERCOT salvato"):
                st.dataframe(ercot_df.tail(200))
        st.stop()

    # --- Parsing CSV hashrate ---
    try:
        hdf = pd.read_csv(hash_file)
    except Exception as e:
        st.error(f"Errore nel leggere il CSV hashrate: {e}")
        st.stop()

    if "timestamp" not in hdf.columns:
        st.error("Il CSV deve avere una colonna 'timestamp'.")
        st.stop()

    # parse timestamp
    hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], utc=True, errors="coerce")
    hdf = hdf.dropna(subset=["timestamp"])
    hdf = hdf.set_index("timestamp").sort_index()

    # individua colonne numeriche da sommare (es. vari account Antpool)
    num_cols = hdf.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.error("Nel CSV non ci sono colonne numeriche di hashrate (TH/s). Aggiungi almeno una colonna numerica.")
        st.stop()

    st.write(f"Colonne di hashrate utilizzate (somma): `{', '.join(num_cols)}`")

    # somma su tutte le colonne numeriche → TH/s totali
    hdf["hashrate_ths_total"] = hdf[num_cols].sum(axis=1)

    # converte in PH/s per avere numeri più leggibili
    hdf["hashrate_phs"] = hdf["hashrate_ths_total"] / 1_000.0

    # opzionale: resampling frequenza
    freq_label = st.selectbox(
        "Risoluzione temporale per il grafico",
        options=["5T", "15T", "30T", "H", "4H"],
        format_func=lambda x: {
            "5T": "5 minuti",
            "15T": "15 minuti",
            "30T": "30 minuti",
            "H": "1 ora",
            "4H": "4 ore",
        }.get(x, x),
        index=1,
        key="freq_monitor"
    )

    # resample hashrate
    h_res = hdf[["hashrate_phs"]].resample(freq_label).mean()

    if ercot_df is None or ercot_df.empty:
        st.warning("Mostro solo l'hashrate (nessun dato ERCOT disponibile).")
        merged = h_res.copy()
        merged["price_usd_per_kwh"] = np.nan
    else:
        # resample ERCOT sulla stessa frequenza
        e_res = ercot_df[["price_usd_per_kwh"]].resample(freq_label).mean()

        # allinea periodo comune
        start = max(h_res.index.min(), e_res.index.min())
        end = min(h_res.index.max(), e_res.index.max())
        h_res = h_res.loc[(h_res.index >= start) & (h_res.index <= end)]
        e_res = e_res.loc[(e_res.index >= start) & (e_res.index <= end)]

        merged = h_res.join(e_res, how="inner")

    if merged.empty:
        st.warning("Non ci sono dati sovrapposti tra hashrate e prezzi ERCOT nel periodo selezionato.")
        st.stop()

    # --- Grafico a linee con doppio asse Y ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["hashrate_phs"],
            name="Hashrate pool (PH/s)",
            mode="lines"
        ),
        secondary_y=False,
    )

    if merged["price_usd_per_kwh"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=merged.index,
                y=merged["price_usd_per_kwh"],
                name=f"ERCOT {ercot_location} ($/kWh)",
                mode="lines"
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Hashrate pool vs prezzo ERCOT",
        xaxis_title="Tempo (UTC)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Hashrate pool (PH/s)", secondary_y=False)
    fig.update_yaxes(title_text="Prezzo energia ($/kWh)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Dati uniti (hashrate + ERCOT)"):
        st.dataframe(merged.reset_index().rename(columns={"index": "timestamp"}))



st.divider()