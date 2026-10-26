import argparse, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from importlib import import_module
source=import_module('01_data_acquisition.data_source'); prep=import_module('02_data_preprocessing.preprocess'); ga=import_module('06_genetic_optimizer.ga_optimizer'); metrics=import_module('08_evaluation.metrics'); storage=import_module('10_deployment.storage')

def main():
 p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=30); p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=5); p.add_argument('--skip-lstm',action='store_true'); args=p.parse_args()
 raw=source.generate_building_data(args.days,args.seed); clean=prep.clean_and_engineer(raw); tr,va,te=prep.chronological_split(clean)
 print(f'Data rows: raw={len(raw)} clean={len(clean)} train={len(tr)} val={len(va)} test={len(te)}')
 forecast_metrics={}
 if not args.skip_lstm:
  try:
   lstm=import_module('03_load_forecasting.lstm_forecaster'); bundle=lstm.train_lstm(tr,va,epochs=args.epochs,lookback=24,horizon=24,seed=args.seed); actual,pred=lstm.predict_windows(bundle,te); forecast_metrics=lstm.forecast_metrics(actual,pred); print('LSTM metrics:',forecast_metrics)
  except Exception as e: print('LSTM skipped:',repr(e))
 n=min(24,len(te)); day=te.iloc[:n]; load=day.load_kw.to_numpy(); solar=day.solar_kw.to_numpy(); price=day.price_per_kwh.to_numpy()
 base_cost=float(np.sum(np.maximum(load-solar,0)*price)); result=ga.optimize_battery_schedule(load,solar,price,population_size=35,generations=20,seed=args.seed)
 report={'forecast':forecast_metrics,'optimization':{'baseline_energy_cost':base_cost,'optimized_cost':result['cost'],'algorithm':result['algorithm'],'schedule_kw':result['schedule'].tolist()}}
 print('GA result:',report['optimization']); storage.log_experiment(report); print('Saved run to artifacts/experiments.sqlite')
if __name__=='__main__': main()
