"""Compact PyTorch LSTM multi-step forecasting model."""
import numpy as np

FEATURES=['load_kw','outdoor_temp_c','humidity_pct','occupancy','solar_kw','price_per_kwh','hour_sin','hour_cos','dow_sin','dow_cos']
try:
 import torch
 from torch import nn
 class LoadLSTM(nn.Module):
  def __init__(self,n_features=len(FEATURES),hidden1=64,hidden2=32,horizon=24,dropout=.15):
   super().__init__(); self.lstm1=nn.LSTM(n_features,hidden1,batch_first=True); self.drop=nn.Dropout(dropout); self.lstm2=nn.LSTM(hidden1,hidden2,batch_first=True); self.head=nn.Sequential(nn.Linear(hidden2,64),nn.ReLU(),nn.Linear(64,horizon)); self.horizon=horizon
  def forward(self,x):
   x,_=self.lstm1(x); x=self.drop(x); x,_=self.lstm2(x); return self.head(x[:,-1,:])
except ImportError:
 torch=None; LoadLSTM=None

def make_windows(df, features=FEATURES, lookback=24, horizon=24):
    x=df[features].to_numpy(dtype=np.float32); y=df['load_kw'].to_numpy(dtype=np.float32)
    X,Y=[],[]
    for i in range(lookback, len(df)-horizon+1): X.append(x[i-lookback:i]); Y.append(y[i:i+horizon])
    if not X: raise ValueError('Insufficient rows for requested lookback/horizon')
    return np.asarray(X),np.asarray(Y)

def train_lstm(train_df, val_df=None, epochs=8, lookback=24, horizon=24, batch_size=64, seed=42, device=None):
    if torch is None: raise ImportError('PyTorch is required. Install torch from https://pytorch.org/')
    torch.manual_seed(seed); np.random.seed(seed)
    # Standardization parameters are learned only on the training period.
    means=train_df[FEATURES].mean(); stds=train_df[FEATURES].std().replace(0,1).fillna(1)
    def transform(frame):
        f=frame.copy(); f[FEATURES]=(f[FEATURES]-means)/stds; f['load_kw']=(f['load_kw']-means['load_kw'])/stds['load_kw']; return f
    tr=transform(train_df); X,Y=make_windows(tr,lookback=lookback,horizon=horizon)
    # Targets returned by make_windows are already standardized because load_kw is scaled in tr.
    dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=LoadLSTM(n_features=len(FEATURES),horizon=horizon).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.MSELoss()
    ds=torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(Y)); loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True)
    history=[]
    for ep in range(epochs):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
        history.append({'epoch':ep+1,'train_mse_scaled':float(np.mean(losses))})
    model.eval()
    return {'model':model,'means':means,'stds':stds,'history':history,'device':dev,'lookback':lookback,'horizon':horizon}

def predict_windows(bundle, frame):
    if torch is None: raise ImportError('PyTorch is required')
    means,stds=bundle['means'],bundle['stds']; f=frame.copy()
    f[FEATURES]=(f[FEATURES]-means)/stds; f['load_kw']=(f['load_kw']-means['load_kw'])/stds['load_kw']
    X,_=make_windows(f,lookback=bundle['lookback'],horizon=bundle['horizon'])
    # Return targets in original kW units, aligned with each forecast window.
    raw_y=frame['load_kw'].to_numpy(dtype=np.float32); lookback=bundle['lookback']; horizon=bundle['horizon']
    actual=np.asarray([raw_y[i:i+horizon] for i in range(lookback, len(frame)-horizon+1)])
    with torch.no_grad(): pred=bundle['model'](torch.tensor(X,device=bundle['device'])).cpu().numpy()
    pred=pred*stds['load_kw']+means['load_kw']
    return actual,pred

def forecast_metrics(actual,pred):
    a=np.asarray(actual).reshape(-1); p=np.asarray(pred).reshape(-1); err=a-p
    denom=np.maximum(np.abs(a),1e-3)
    return {'MAE':float(np.mean(np.abs(err))),'RMSE':float(np.sqrt(np.mean(err**2))),'MAPE_pct':float(np.mean(np.abs(err)/denom)*100)}
