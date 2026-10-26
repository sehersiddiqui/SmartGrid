import sys
from pathlib import Path
from importlib import import_module
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
def test_synthetic_data_reproducible():
 m=import_module('01_data_acquisition.data_source'); a=m.generate_building_data(4,7); b=m.generate_building_data(4,7); assert a.equals(b); assert len(a)==96

def test_preprocessing_chronological():
 src=import_module('01_data_acquisition.data_source'); prep=import_module('02_data_preprocessing.preprocess'); df=prep.clean_and_engineer(src.generate_building_data(8,9)); tr,va,te=prep.chronological_split(df); assert tr.timestamp.max()<va.timestamp.min()<te.timestamp.min(); assert not tr.isna().any().any()

def test_ga_respects_bounds():
 ga=import_module('06_genetic_optimizer.ga_optimizer'); import numpy as np
 out=ga.optimize_battery_schedule(np.ones(24)*25,np.ones(24)*4,np.ones(24)*7,population_size=10,generations=2,seed=2); assert len(out['schedule'])==24; assert np.max(np.abs(out['schedule']))<=20.00001

def test_fuzzy_range():
 f=import_module('08_evaluation.fuzzy_logic'); assert 0<=f.fuzzy_cooling_priority(31,.8,6)<=1
