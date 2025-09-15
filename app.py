# DESMO Bitcoin Mining Optimizer 

from __future__ import annotations

import math, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta

import requests, pandas as pd, numpy as np # type: ignore
from dateutil.relativedelta import relativedelta 
import streamlit as st # type: ignore
import plotly.graph_objects as go # type: ignore
import os


try:
    import pulp  # type: ignore
    HAS_PULP = True
except Exception:
    HAS_PULP = False

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
@st.cache_data(ttl=900)
def fetch_avg_fees_per_block_btc_7d(max_blocks: int = 1200) -> Optional[float]:
    try:
        url = "https://mempool.space/api/v1/blocks-extras"
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        page = r.json()
        if not isinstance(page, list) or not page:
            return None

        now = int(time.time())
        threshold = now - 7 * 24 * 3600

        fees_btc: List[float] = []

        def collect_from(page_list):
            for b in page_list:
                ts = int(b.get("timestamp") or b.get("time") or 0)
                if ts <= 0:
                    continue
                extras = b.get("extras") if isinstance(b.get("extras"), dict) else {}
                sats = extras.get("totalFees") or b.get("totalFees")
                if sats is not None:
                    fees_btc.append(float(sats) / 1e8)  # sats -> BTC

        collect_from(page)
        next_height = int(page[-1].get("height", page[0].get("height"))) - 1

        while len(fees_btc) < max_blocks and next_height > 0:
            r = requests.get(f"https://mempool.space/api/v1/blocks-extras/{next_height}", timeout=15, headers=HEADERS)
            r.raise_for_status()
            page = r.json()
            if not isinstance(page, list) or not page:
                break
            oldest_ts = int(page[-1].get("timestamp") or page[-1].get("time") or 0)
            collect_from(page)
            if oldest_ts and oldest_ts < threshold:
                break
            next_height = int(page[-1].get("height", next_height)) - 1

        if not fees_btc:
            return None
        return float(np.mean(fees_btc))
    except Exception:
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

# Catalogo (puoi modificarlo in UI)
DEFAULT_CATALOG: Dict[str, AsicModel] = {
    "Antminer S19 PRO 110T": AsicModel("Antminer S19 PRO 110T", hashrate_ths=110.0, power_kw=3.25, unit_price_usd=319.0),
    "Antminer S19 XP 141T": AsicModel("Antminer S19 XP 141T", hashrate_ths=141.0, power_kw=3.01, unit_price_usd=1100.0),
    "Bitmain S21 200T": AsicModel("Bitmain S21 200T", hashrate_ths=200.0, power_kw=3.50, unit_price_usd=2840.0),
    "Bitmain S21 Pro 220T": AsicModel("Bitmain S21 Pro 220T", hashrate_ths=220.0, power_kw=3.300, unit_price_usd=3432.0),
    "Bitmain S21+ 225T": AsicModel("Bitmain S21+ 225T", hashrate_ths=225.0, power_kw=3.7125, unit_price_usd=3217.5),
    "Bitmain S21 Pro 234T": AsicModel("Bitmain S21 Pro 234T", hashrate_ths=234.0, power_kw=3.510, unit_price_usd=3814.2),
    "Bitmain S21+ 235T": AsicModel("Bitmain S21+ 235T", hashrate_ths=235.0, power_kw=3.8775, unit_price_usd=3384.0),
    "Bitmain S21 Pro 245T": AsicModel("Bitmain S21 Pro 245T", hashrate_ths=245.0, power_kw=3.675, unit_price_usd=4091.5),
    "Bitmain S21XP 270T": AsicModel("Bitmain S21XP 270T", hashrate_ths=270.0, power_kw=3.645, unit_price_usd=5805.0),
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
st.caption("Model mining economics, compare scenarios, schedule future steps, and (optimizer UI disabilitato) ottimizza la fleet sotto budget.")

top_cols = st.columns([1,1,2])
mode = top_cols[0].radio("Mode", ["Classica", "Prossimi Step"], index=0, horizontal=True)

# Halving controls (solo data visibile; subsidy fisse dietro le quinte)
with top_cols[1]:
    st.write("Prossimo halving (subsidy 3.125 → 1.5625 BTC)")
    halving_date_input = st.date_input("Data halving", value=date(2028,4,11))
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
        avg_fees = fetch_avg_fees_per_block_btc_7d()  # 7d average, can be None
        net_ths = difficulty_to_network_hashrate_ths(diff) if diff else float("nan")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("BTC Spot (USD)", f"{price_usd:,.0f}" if price_usd==price_usd else "—")
        st.metric("Block Reward (BTC, now)", f"{live_subsidy_now:.3f}" if live_subsidy_now else "—")
        st.metric("Avg Fees / Block (BTC, 7d)", f"{avg_fees:.4f}" if (avg_fees is not None) else "(set below)")
        if avg_fees is not None:
            st.caption(f"≈ {avg_fees*1e8:,.0f} sats per block (7d)")
    with col2:
        st.metric("Height", f"{height:,}" if height else "—")
        st.metric("Difficulty", f"{diff:,.0f}" if diff else "—")
        st.metric("Network Hashrate (TH/s)", f"{net_ths:,.0f}" if net_ths==net_ths else "—")

    st.divider()
    st.subheader("Energy Pricing")
    flat_price = st.number_input("Flat $/kWh", min_value=0.0, step=0.001, value=0.05, format="%.3f")
    uploaded_csv = st.file_uploader("Curva oraria opzionale CSV (colonna 'price_usd_per_kwh')", type=["csv"])
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
            catalog[name] = AsicModel(name=name,
                                      hashrate_ths=float(row["hashrate_ths"]),
                                      power_kw=float(row["power_kw"]),
                                      unit_price_usd=float(row["unit_price_usd"]))
        except Exception:
            pass

# -----------------------------
# Scenari — Modalità Classica
# -----------------------------
if mode == "Classica":
    st.subheader("2) Scenarios")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios: List[Scenario] = [] # type: ignore

    # --- Fleet grid OUTSIDE the form for real-time budget calc ---
    st.markdown("**Fleet (units per model)**")
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
    with st.form("new_scenario"):
        st.markdown("**Parametri scenario** (premi *Add scenario* per salvare)")
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
        avg_fees_override = c8.number_input("Avg fees per block BTC (0 = live/none)", min_value=0.0, step=0.01, value=0.2)

        c9, c10, c11 = st.columns(3)
        monthly_net_growth = c9.number_input("Network hashrate growth % / month", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
        btc_price_mom = c10.number_input("BTC price growth % / month", min_value=-50.0, max_value=100.0, value=0.0, step=0.1)
        months_horizon = int(c11.number_input("Months horizon", min_value=6, max_value=120, value=60, step=6))

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

                df = simulate_scenario(
                    scn,
                    catalog=catalog,
                    network_ths_start=live_network_ths,
                    avg_fees_btc=fees_block,
                    btc_price_usd_start=btc_price,
                    flat_price_usd_per_kwh=scn.variable_energy_usd_per_kwh,
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

# -----------------------------
# Modalità Prossimi Step
# -----------------------------
else:
    st.subheader("2) Base Scenarios (t0)")

    if "scenarios_ns" not in st.session_state:
        st.session_state.scenarios_ns: List[Scenario] = [] # type: ignore
    if "future_steps" not in st.session_state:
        st.session_state.future_steps: List[FutureStep] = [] # type: ignore

    # --- Fleet grid OUTSIDE the form for real-time budget calc (t0) ---
    st.markdown("**Fleet (units per model) — t0**")
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
        avg_fees_override = c8.number_input("Avg fees per block BTC (0 = live/none)", min_value=0.0, step=0.01, value=0.2, key="fee_ns")

        c9, c10, c11 = st.columns(3)
        monthly_net_growth = c9.number_input("Network hashrate growth % / month", min_value=-50.0, max_value=50.0, value=0.0, step=0.1, key="netg_ns")
        btc_price_mom = c10.number_input("BTC price growth % / month", min_value=-50.0, max_value=100.0, value=0.0, step=0.1, key="prg_ns")
        months_horizon = int(c11.number_input("Months horizon", min_value=6, max_value=120, value=60, step=6, key="hor_ns"))

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

st.divider()

