import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_BASE = ['load_kw','outdoor_temp_c','humidity_pct','occupancy','solar_kw','price_per_kwh']

def clean_and_engineer(df: pd.DataFrame, interval_minutes=60) -> pd.DataFrame:
    data = df.copy()
    if 'timestamp' not in data: raise ValueError("Required column: timestamp")
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    freq = f'{interval_minutes}min'
    data = data.resample(freq).mean(numeric_only=True)
    for col in NUMERIC_BASE:
        if col not in data: data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].interpolate(limit=4, limit_direction='both').ffill().bfill()
        if data[col].isna().all(): data[col] = 0.0
        # Winsorize only gross outliers using training-independent robust IQR bounds.
        q1, q3 = data[col].quantile([.25,.75]); iqr = q3-q1
        if np.isfinite(iqr) and iqr > 0:
            data[col] = data[col].clip(q1-3*iqr, q3+3*iqr)
    data['hour'] = data.index.hour + data.index.minute/60
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    data['hour_sin'] = np.sin(2*np.pi*data['hour']/24)
    data['hour_cos'] = np.cos(2*np.pi*data['hour']/24)
    data['dow_sin'] = np.sin(2*np.pi*data['day_of_week']/7)
    data['dow_cos'] = np.cos(2*np.pi*data['day_of_week']/7)
    for lag in [1,4,24]: data[f'load_lag_{lag}'] = data['load_kw'].shift(lag)
    data['load_rolling_mean_6'] = data['load_kw'].shift(1).rolling(6,min_periods=1).mean()
    data['load_rolling_std_6'] = data['load_kw'].shift(1).rolling(6,min_periods=2).std().fillna(0)
    return data.dropna().reset_index()

def chronological_split(df, train_frac=.70, val_frac=.15):
    if train_frac <= 0 or val_frac <= 0 or train_frac+val_frac >= 1: raise ValueError('Invalid split fractions')
    n=len(df); a=int(n*train_frac); b=int(n*(train_frac+val_frac))
    if a<2 or b<=a or b>=n: raise ValueError('Not enough rows for chronological split')
    return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()

def fit_feature_scaler(train, features):
    scaler=StandardScaler(); scaler.fit(train[features].astype(float)); return scaler
