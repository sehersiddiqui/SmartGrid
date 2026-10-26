"""DEAP day-ahead optimizer. If DEAP is missing, a deterministic heuristic remains available."""
import numpy as np
from config import EnergyConfig

def evaluate_schedule(schedule, load, solar, prices, config=None, demand_temp=None, occupancy=None):
    cfg=config or EnergyConfig(); sched=np.asarray(schedule,dtype=float); load=np.asarray(load,float); solar=np.asarray(solar,float); prices=np.asarray(prices,float); n=min(map(len,[sched,load,solar,prices])); sched,load,solar,prices=sched[:n],load[:n],solar[:n],prices[:n]
    dt=cfg.interval_minutes/60; soc=cfg.initial_soc; costs=[]; grid_list=[]; socs=[]; throughput=0.
    for t in range(n):
        b=float(np.clip(sched[t],-cfg.battery_max_power_kw,cfg.battery_max_power_kw)); cap=cfg.battery_capacity_kwh
        if b>=0: b=min(b,max(0,(cfg.max_soc-soc)*cap/(cfg.charge_efficiency*dt))); soc+=b*dt*cfg.charge_efficiency/cap
        else: b=-min(abs(b),max(0,(soc-cfg.min_soc)*cap*cfg.discharge_efficiency/dt)); soc+=b*dt/(cfg.discharge_efficiency*cap)
        grid=max(0,load[t]+b-solar[t]); cost=grid*prices[t]*dt + max(0,grid-cfg.peak_threshold_kw)*cfg.peak_penalty_per_kw + abs(b)*dt*cfg.battery_degradation_per_kwh
        costs.append(cost); grid_list.append(grid); socs.append(soc); throughput+=abs(b)*dt
    # discourage terminal SOC depletion so schedules do not exploit the finite horizon
    terminal_penalty=abs(soc-cfg.initial_soc)*cfg.battery_capacity_kwh*.15
    return float(np.sum(costs)+terminal_penalty), {'cost':float(np.sum(costs)+terminal_penalty),'grid_kw':grid_list,'soc':socs,'throughput_kwh':throughput}

def optimize_battery_schedule(load, solar, prices, config=None, population_size=50, generations=35, seed=42):
    cfg=config or EnergyConfig(); n=min(len(load),len(solar),len(prices)); load=np.asarray(load[:n]); solar=np.asarray(solar[:n]); prices=np.asarray(prices[:n]); rng=np.random.default_rng(seed)
    try:
        from deap import base, creator, tools, algorithms
    except ImportError:
        # Greedy fallback: charge in cheap/solar-rich hours and discharge in expensive hours.
        score=(prices-prices.min())/(np.ptp(prices)+1e-8) - .65*(solar/(solar.max()+1e-8)); sched=np.where(score>.65,-cfg.battery_max_power_kw*.65,np.where(score<.20,cfg.battery_max_power_kw*.45,0.))
        cost,detail=evaluate_schedule(sched,load,solar,prices,cfg); return {'schedule':sched,'cost':cost,'history':[cost],'detail':detail,'algorithm':'heuristic fallback (DEAP unavailable)'}
    # Unique creator names permit re-running notebooks without DEAP duplicate-class errors.
    fit_name='FitnessMinEnergy'; ind_name='EnergyIndividual'
    if not hasattr(creator,fit_name): creator.create(fit_name,base.Fitness,weights=(-1.0,))
    if not hasattr(creator,ind_name): creator.create(ind_name,list,fitness=getattr(creator,fit_name))
    toolbox=base.Toolbox(); toolbox.register('gene',rng.uniform,-cfg.battery_max_power_kw,cfg.battery_max_power_kw); toolbox.register('individual',tools.initRepeat,getattr(creator,ind_name),toolbox.gene,n); toolbox.register('population',tools.initRepeat,list,toolbox.individual)
    def fit(ind): return (evaluate_schedule(ind,load,solar,prices,cfg)[0],)
    toolbox.register('evaluate',fit); toolbox.register('mate',tools.cxTwoPoint); toolbox.register('mutate',tools.mutGaussian,mu=0,sigma=cfg.battery_max_power_kw*.22,indpb=.12); toolbox.register('select',tools.selTournament,tournsize=3)
    pop=toolbox.population(n=population_size); hof=tools.HallOfFame(1); history=[]
    algorithms.eaSimple(pop,toolbox,cxpb=.7,mutpb=.25,ngen=generations,halloffame=hof,verbose=False)
    best=np.clip(np.asarray(hof[0],float),-cfg.battery_max_power_kw,cfg.battery_max_power_kw); cost,detail=evaluate_schedule(best,load,solar,prices,cfg)
    # Convergence trace is a post-run best-so-far diagnostic when no per-generation logbook is retained.
    history=[float(min(evaluate_schedule(ind,load,solar,prices,cfg)[0] for ind in pop)),cost]
    return {'schedule':best,'cost':cost,'history':history,'detail':detail,'algorithm':'DEAP Genetic Algorithm'}
