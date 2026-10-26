"""Procedural synthetic building data + optional UCI household data import."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def generate_building_data(days: int = 60, seed: int = 42, interval_minutes: int = 60) -> pd.DataFrame:
    """Generate a physically-inspired, reproducible dataset; values are synthetic, not measurements."""
    if days < 3: raise ValueError("days must be >= 3")
    if interval_minutes not in (15, 30, 60): raise ValueError("interval_minutes must be 15, 30, or 60")
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // interval_minutes
    idx = pd.date_range('2025-01-01', periods=n, freq=f'{interval_minutes}min')
    hour = idx.hour.to_numpy() + idx.minute.to_numpy()/60
    dow = idx.dayofweek.to_numpy()
    day_idx = np.arange(n) * interval_minutes / 1440
    weekend = (dow >= 5).astype(float)
    # Smooth daily and seasonal components + stochastic perturbations.
    seasonal = 2.4 * np.sin(2*np.pi*(day_idx/30 - .15))
    outdoor = 28 + 4.8*np.sin(2*np.pi*(hour-8)/24) + seasonal + rng.normal(0, .8, n)
    humidity = np.clip(62 - .55*(outdoor-28) + rng.normal(0, 5, n), 25, 95)
    daylight = np.maximum(0, np.sin(np.pi*(hour-6)/12))
    cloud = np.clip(rng.beta(2.2, 2.5, n) + .12*np.sin(day_idx/4), 0, 1)
    solar = np.maximum(0, (14*daylight*(1-cloud) + rng.normal(0, .35, n)))
    # Occupancy schedule is stochastic around a building timetable.
    work_profile = np.exp(-0.5*((hour-13)/4.0)**2)
    open_hours = ((hour >= 7) & (hour < 21)).astype(float)
    occupancy = np.clip((0.10 + 0.78*work_profile)*open_hours*(1-.68*weekend) + rng.normal(0,.045,n), 0, 1)
    price = np.select([((hour>=17)&(hour<21)),((hour>=8)&(hour<17)),((hour>=0)&(hour<6))],[12.8,8.2,5.4],default=6.5).astype(float)
    price += .35*np.sin(day_idx/5) + rng.normal(0,.12,n)
    base = 7.2 + 2.2*open_hours + 7.0*occupancy + 0.10*np.maximum(outdoor-24,0) + rng.normal(0,.65,n)
    hvac = np.maximum(0, 1.3 + .55*np.maximum(outdoor-22,0) + 2.2*occupancy + rng.normal(0,.4,n))
    lighting = 0.8 + 3.2*occupancy*(1-.45*daylight) + rng.normal(0,.2,n)
    ev = np.where((hour>=18)&(hour<22)&(occupancy>.15), 2.5, 0) * (rng.random(n)>.35)
    total = np.maximum(1, base + hvac + lighting + ev)
    # Sensor noise / rare anomalies and gaps are injected for preprocessing practice.
    total += rng.normal(0, .25, n)
    if n > 80:
        anomaly_ix = rng.choice(np.arange(24,n-24), size=max(1,n//700), replace=False)
        total[anomaly_ix] *= rng.uniform(2.2,3.0,len(anomaly_ix))
        missing_ix = rng.choice(np.arange(n), size=max(1,n//250), replace=False)
        total[missing_ix] = np.nan
    return pd.DataFrame({
        'timestamp':idx, 'load_kw':total, 'outdoor_temp_c':outdoor, 'humidity_pct':humidity,
        'occupancy':occupancy, 'solar_kw':solar, 'price_per_kwh':price,
        'hvac_kw':hvac, 'lighting_kw':np.maximum(lighting,0), 'ev_kw':ev,
        'is_synthetic':True
    })


def load_uci_household(cache_dir='data', max_rows=None) -> pd.DataFrame:
    """Download UCI Individual Household Electric Power Consumption (real household data).

    Auxiliary weather/occupancy/solar/tariff columns are intentionally not fabricated here;
    callers should add explicitly labelled simulated covariates if needed.
    """
    import requests
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / 'household_power_consumption.zip'
    url = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    if not zip_path.exists():
        response = requests.get(url, timeout=60); response.raise_for_status(); zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        member = next((x for x in zf.namelist() if x.lower().endswith('.txt')), None)
        if member is None: raise ValueError('UCI archive did not contain the expected text file')
        with zf.open(member) as f:
            df = pd.read_csv(f, sep=';', na_values=['?'], low_memory=False, nrows=max_rows)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df['load_kw'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    # Original global active power is in kW at minute granularity.
    hourly = df.dropna(subset=['timestamp']).set_index('timestamp')['load_kw'].resample('1h').mean().to_frame().reset_index()
    hourly['is_synthetic'] = False
    return hourly.rename(columns={'timestamp':'timestamp'})
