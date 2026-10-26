import numpy as np

def summarize_episode(records):
    if not records: return {}
    d={k:np.asarray([r.get(k,0) for r in records],dtype=float) for k in records[0] if isinstance(records[0].get(k),(int,float,np.number))}
    cost=float(d.get('cost',np.zeros(1)).sum()); grid=d.get('grid_kw',np.zeros(1)); comfort=d.get('comfort_penalty',np.zeros(1))
    return {'total_cost':cost,'peak_grid_kw':float(grid.max(initial=0)),'grid_energy_kwh':float(grid.sum()),'comfort_penalty_total':float(comfort.sum()),'comfort_violation_steps':int(np.sum(comfort>0)),'battery_throughput_kwh':float(np.abs(d.get('battery_kw',np.zeros(1))).sum()),'steps':len(records)}

def compare_costs(results):
    baseline=results.get('No optimization',{}).get('total_cost',None)
    rows=[]
    for method,metrics in results.items():
        row={'method':method,**metrics}
        if baseline and baseline>0: row['cost_saving_pct']=100*(baseline-metrics.get('total_cost',baseline))/baseline
        else: row['cost_saving_pct']=0.
        rows.append(row)
    return rows
