import numpy as np
try:
 import gymnasium as gym
 from gymnasium import spaces
except ImportError:
 gym=None; spaces=None
from config import EnergyConfig

class BuildingEnergyEnv(gym.Env if gym else object):
    """Educational building MDP. Action = [battery kW, HVAC setpoint, EV kW, lighting fraction]."""
    metadata={'render_modes':[]}
    def __init__(self, profile=None, config=None, episode_steps=24, seed=42):
        if gym is None: raise ImportError('Install gymnasium to use BuildingEnergyEnv')
        self.cfg=config or EnergyConfig(seed=seed); self.profile=profile; self.episode_steps=episode_steps; self.rng=np.random.default_rng(seed)
        self.action_space=spaces.Box(low=np.array([-self.cfg.battery_max_power_kw,18.,0.,0.],dtype=np.float32),high=np.array([self.cfg.battery_max_power_kw,28.,7.,1.],dtype=np.float32),dtype=np.float32)
        self.observation_space=spaces.Box(low=-10,high=100,shape=(12,),dtype=np.float32)
        self.reset(seed=seed)
    def _exo(self,t):
        if self.profile is not None and len(self.profile):
            row=self.profile.iloc[t%len(self.profile)]
            return float(row.get('load_kw',25)),float(row.get('outdoor_temp_c',29)),float(row.get('occupancy',.5)),float(row.get('solar_kw',0)),float(row.get('price_per_kwh',7))
        h=t%24; occ=float(np.clip(.15+.72*np.exp(-.5*((h-13)/4)**2),0,1)); temp=29+4*np.sin(2*np.pi*(h-8)/24); solar=max(0,14*np.sin(np.pi*(h-6)/12)*.7); price=12 if 17<=h<21 else (8 if 8<=h<17 else 5.5); load=12+8*occ+.12*max(temp-24,0)+2.5*occ
        return load,temp,occ,solar,price
    def _obs(self):
        load,temp,occ,solar,price=self._exo(self.t); h=self.t%24
        return np.array([load/60,temp/45,occ,solar/20,price/15,self.soc,self.indoor_temp/35,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),self.last_battery/self.cfg.battery_max_power_kw,self.last_hvac/10,self.t/self.episode_steps],dtype=np.float32)
    def reset(self,seed=None,options=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.t=0; self.soc=float(self.cfg.initial_soc); self.indoor_temp=26.; self.last_battery=0.; self.last_hvac=4.; self.total_cost=0.
        return self._obs(),{'soc':self.soc}
    def step(self,action):
        battery,hvac_setpoint,ev_power,lighting=map(float,np.asarray(action).reshape(-1)[:4]); battery=float(np.clip(battery,-self.cfg.battery_max_power_kw,self.cfg.battery_max_power_kw)); hvac_setpoint=float(np.clip(hvac_setpoint,18,28)); ev_power=float(np.clip(ev_power,0,7)); lighting=float(np.clip(lighting,0,1))
        base,temp,occ,solar,price=self._exo(self.t)
        # Thermal dynamics: setpoint tracking, outdoor heat transfer, and occupancy gains.
        hvac_kw=float(np.clip(1.0+.75*abs(temp-hvac_setpoint)+2.0*occ,0,12)); self.indoor_temp += .12*(temp-self.indoor_temp)-.30*(24-hvac_setpoint)*.15 + .10*occ - .16*(24-hvac_setpoint)
        # Battery power positive=charge; enforce SOC and efficiency constraints.
        dt=self.cfg.interval_minutes/60; cap=self.cfg.battery_capacity_kwh
        if battery>=0: battery=min(battery,max(0,(self.cfg.max_soc-self.soc)*cap/(self.cfg.charge_efficiency*dt))); self.soc += battery*dt*self.cfg.charge_efficiency/cap
        else: battery=-min(abs(battery),max(0,(self.soc-self.cfg.min_soc)*cap*self.cfg.discharge_efficiency/dt)); self.soc += battery*dt/(self.cfg.discharge_efficiency*cap)
        lighting_kw=1.0+3.5*occ*lighting; demand=max(0,base+hvac_kw+lighting_kw+ev_power); net=demand+ battery-solar; grid=max(0,net); export=max(0,-net)
        energy_cost=(grid-export*self.cfg.export_price_fraction)*price*dt
        peak_penalty=max(0,grid-self.cfg.peak_threshold_kw)*self.cfg.peak_penalty_per_kw
        comfort=max(0,self.cfg.comfort_low_c-self.indoor_temp)+max(0,self.indoor_temp-self.cfg.comfort_high_c)
        comfort_penalty=comfort*self.cfg.comfort_penalty_per_degree
        battery_penalty=abs(battery)*dt*self.cfg.battery_degradation_per_kwh
        # Reward is negative cost; service loads are not switched off to chase energy savings.
        cost=energy_cost+peak_penalty+comfort_penalty+battery_penalty; reward=-float(cost); self.total_cost+=cost
        self.last_battery=battery; self.last_hvac=hvac_kw; self.t+=1; terminated=self.t>=self.episode_steps; truncated=False
        info={'grid_kw':grid,'demand_kw':demand,'solar_kw':solar,'price':price,'soc':self.soc,'indoor_temp_c':self.indoor_temp,'energy_cost':energy_cost,'peak_penalty':peak_penalty,'comfort_penalty':comfort_penalty,'battery_penalty':battery_penalty,'cost':cost,'battery_kw':battery,'hvac_setpoint_c':hvac_setpoint,'ev_kw':ev_power,'lighting_fraction':lighting}
        return self._obs(),reward,terminated,truncated,info
