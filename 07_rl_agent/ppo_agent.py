from pathlib import Path
from importlib import import_module

def _env_class(): return import_module('05_energy_environment.building_env').BuildingEnergyEnv

def train_ppo(profile=None, total_timesteps=20000, seed=42, save_path='artifacts/ppo_energy'):
    try: from stable_baselines3 import PPO
    except ImportError as e: raise ImportError('PPO requires stable-baselines3. Install it using requirements.txt') from e
    env=_env_class()(profile=profile,seed=seed)
    model=PPO('MlpPolicy',env,verbose=0,seed=seed,learning_rate=3e-4,n_steps=512,batch_size=64,gamma=.99,gae_lambda=.95,clip_range=.2,policy_kwargs={'net_arch':{'pi':[256,128],'vf':[256,128]}})
    model.learn(total_timesteps=total_timesteps)
    Path(save_path).parent.mkdir(parents=True,exist_ok=True); model.save(save_path); return model

def load_ppo(path='artifacts/ppo_energy'):
    from stable_baselines3 import PPO
    return PPO.load(path)
