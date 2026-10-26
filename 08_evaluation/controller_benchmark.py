"""Matched-profile controller comparison for the five methods in the project specification."""
from importlib import import_module
import numpy as np
from .metrics import summarize_episode

def evaluate_controllers(profile, seed=42, ppo_model=None, ga_schedule=None):
    env_cls=import_module('05_energy_environment.building_env').BuildingEnergyEnv
    ga=import_module('06_genetic_optimizer.ga_optimizer')
    profile=profile.reset_index(drop=True).copy()
    if ga_schedule is None:
        ga_result=ga.optimize_battery_schedule(profile.load_kw.to_numpy(),profile.solar_kw.to_numpy(),profile.price_per_kwh.to_numpy(),seed=seed)
        ga_schedule=ga_result['schedule']
    results={}
    for method in ['No optimization','Rule-based','GA-only','PPO-only','GA+PPO']:
        env=env_cls(profile=profile,episode_steps=len(profile),seed=seed)
        obs,_=env.reset(seed=seed); records=[]; done=False; t=0
        while not done:
            row=profile.iloc[t%len(profile)]
            if method=='No optimization':
                action=np.array([0.,24.,min(3.,float(row.get('ev_kw',0))),.75],dtype=np.float32)
            elif method=='Rule-based':
                # Charge when price is low or solar is abundant; discharge at expensive peak hours.
                price=float(row.price_per_kwh); solar=float(row.solar_kw)
                batt=7.0 if (price<=6.5 or solar>=8) else (-8.0 if price>=11 else 0.0)
                action=np.array([batt,23.5 if float(row.outdoor_temp_c)>30 else 24.5,0.0,.7],dtype=np.float32)
            elif method=='GA-only':
                action=np.array([float(ga_schedule[t]),24.,0.,.7],dtype=np.float32)
            elif method=='PPO-only':
                if ppo_model is None: continue
                action,_=ppo_model.predict(obs,deterministic=True)
            else: # GA+PPO: PPO adjusts the day-ahead battery plan conservatively.
                if ppo_model is None: continue
                learned,_=ppo_model.predict(obs,deterministic=True)
                action=np.asarray(learned,dtype=np.float32).copy()
                action[0]=float(np.clip(.65*float(ga_schedule[t])+.35*float(action[0]),-20,20))
            obs,reward,terminated,truncated,info=env.step(action)
            records.append(info); t+=1; done=terminated or truncated
        results[method]=summarize_episode(records)
    return results
