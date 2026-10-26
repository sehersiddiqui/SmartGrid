import numpy as np

def solar_profile_forecast(timestamp, peak_kw=14.0, cloud_factor=0.75):
    """Simple daylight-shaped reference forecast; not a trained solar ML model."""
    ts=np.asarray(timestamp)
    hour=np.array([x.hour+x.minute/60 for x in ts])
    return np.maximum(0, peak_kw*np.maximum(0,np.sin(np.pi*(hour-6)/12))*cloud_factor)

def empirical_interval(predictions, residuals, coverage=.90):
    """Symmetric empirical residual interval; assumes residuals are representative."""
    q=np.quantile(np.abs(np.asarray(residuals)), coverage)
    p=np.asarray(predictions)
    return p-q,p+q
