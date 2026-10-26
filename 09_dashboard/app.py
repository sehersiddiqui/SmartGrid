from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from importlib import import_module
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
source=import_module('01_data_acquisition.data_source'); prep=import_module('02_data_preprocessing.preprocess'); ga=import_module('06_genetic_optimizer.ga_optimizer'); fuzzy=import_module('08_evaluation.fuzzy_logic'); storage=import_module('10_deployment.storage')

st.set_page_config(page_title='GridMind | Smart Energy AI',page_icon='⚡',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Manrope',sans-serif} .stApp{background:linear-gradient(135deg,#07131d 0%,#0b1f2b 50%,#0c1724 100%);color:#edf7f5}
[data-testid="stSidebar"]{background:#0a1822;border-right:1px solid #1d3946} .block-container{padding-top:1.6rem;max-width:1500px}
.hero{padding:28px 32px;border:1px solid #234651;border-radius:22px;background:radial-gradient(circle at 80% 20%,#184a48 0%,transparent 35%),linear-gradient(120deg,#102a35,#0c1c28);margin-bottom:22px}
.eyebrow{color:#77e0c0;font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase}.hero h1{font-size:38px;line-height:1.12;margin:10px 0;color:#f3fffb}.hero p{color:#9eb9c1;font-size:15px;max-width:760px}
.kpi{border:1px solid #24434d;background:linear-gradient(140deg,#122c38,#10232f);padding:17px 19px;border-radius:16px;min-height:115px}.k-label{font-size:12px;color:#9ab6be;font-weight:700;letter-spacing:.7px;text-transform:uppercase}.k-value{font-size:28px;font-weight:800;color:#f1fffa;margin-top:8px}.k-note{font-size:12px;color:#7ce0bd;margin-top:3px}
.section{font-size:20px;font-weight:800;color:#e8f8f5;margin:20px 0 8px}.muted{color:#9bb5bd}.stButton>button{border-radius:11px;border:1px solid #3f827a;background:#176b5e;color:white;font-weight:700;min-height:42px}.stButton>button:hover{background:#218b78;border-color:#67d8ba;color:white}
div[data-testid="stMetric"]{background:#102832;padding:14px;border:1px solid #24434d;border-radius:14px}.stAlert{border-radius:12px}
</style>''',unsafe_allow_html=True)
st.sidebar.markdown('## ⚡ GRIDMIND')
st.sidebar.caption('INTELLIGENT ENERGY LAB · IS2')
st.sidebar.divider()
days=st.sidebar.slider('Simulation history (days)',7,120,45,step=1)
seed=st.sidebar.number_input('Reproducibility seed',min_value=0,max_value=999999,value=42,step=1)
mode=st.sidebar.selectbox('Data source',['Generated synthetic building data','UCI real household data (download)'])
use_ppo=st.sidebar.checkbox('Use trained PPO policy if available',value=False)
st.sidebar.caption('Synthetic covariates are explicitly labeled. No live building controls are connected.')

@st.cache_data(show_spinner=False)
def get_data(days,seed,mode):
 if mode.startswith('UCI'):
  raw=source.load_uci_household(max_rows=450000)
  # UCI provides measured household load only; covariates below are simulated and flagged as such.
  rng=np.random.default_rng(seed); n=len(raw); ts=pd.to_datetime(raw.timestamp); h=ts.dt.hour.to_numpy()
  raw['outdoor_temp_c']=28+4*np.sin(2*np.pi*(h-8)/24)+rng.normal(0,1,n); raw['humidity_pct']=np.clip(60+rng.normal(0,6,n),25,95); raw['occupancy']=np.clip(.1+.75*np.exp(-.5*((h-13)/4)**2)+rng.normal(0,.05,n),0,1); raw['solar_kw']=np.maximum(0,14*np.sin(np.pi*(h-6)/12)*.7); raw['price_per_kwh']=np.where((h>=17)&(h<21),12.8,np.where((h>=8)&(h<17),8.2,5.8)); raw['is_synthetic_covariates']=True
  return raw.tail(days*24).copy()
 return source.generate_building_data(days,seed)

try:
 raw=get_data(days,int(seed),mode); clean=prep.clean_and_engineer(raw); st.session_state['raw']=raw; st.session_state['clean']=clean
except Exception as e:
 st.error(f'Data loading failed: {e}'); st.stop()

st.markdown('''<div class="hero"><div class="eyebrow">SMART BUILDING · DECISION INTELLIGENCE</div><h1>Energy, orchestrated.<br><span style="color:#79e2bf">One system. Three kinds of intelligence.</span></h1><p>GridMind combines LSTM demand forecasting, evolutionary day-ahead scheduling, and adaptive control in a transparent building simulator. Observe → Predict → Optimize → Act → Learn.</p></div>''',unsafe_allow_html=True)
last=clean.iloc[-1]; df=clean.copy(); df['timestamp']=pd.to_datetime(df.timestamp)
# Use the last complete 24-step window for an interactive schedule demonstration.
day=df.tail(24).copy(); load=day.load_kw.to_numpy(float); solar=day.solar_kw.to_numpy(float); prices=day.price_per_kwh.to_numpy(float)
with st.spinner('Optimizing the next operating window with the Genetic Algorithm…'):
 result=ga.optimize_battery_schedule(load,solar,prices,population_size=35,generations=20,seed=int(seed))
base_cost=float(np.sum(np.maximum(load-solar,0)*prices)); optimized_cost=float(result['cost']); saving=(base_cost-optimized_cost)/base_cost*100 if base_cost else 0
k1,k2,k3,k4,k5=st.columns(5)
for col,label,val,note in [(k1,'CURRENT LOAD',f'{last.load_kw:.1f} kW','Latest simulated / observed point'),(k2,'SOLAR OUTPUT',f'{last.solar_kw:.1f} kW','Available renewable supply'),(k3,'TARIFF',f'₹{last.price_per_kwh:.2f}/kWh','Time-of-use profile'),(k4,'GA COST ESTIMATE',f'₹{optimized_cost:,.0f}','Planning-window objective'),(k5,'COST DELTA',f'{saving:+.1f}%','vs no-battery energy-cost proxy')]:
 col.markdown(f'<div class="kpi"><div class="k-label">{label}</div><div class="k-value">{val}</div><div class="k-note">{note}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section">01 / Building pulse</div>',unsafe_allow_html=True)
c1,c2=st.columns([1.65,1])
with c1:
 fig=go.Figure(); fig.add_trace(go.Scatter(x=df.timestamp,y=df.load_kw,mode='lines',name='Building load',line=dict(color='#7ce0bd',width=2),fill='tozeroy',fillcolor='rgba(124,224,189,.08)')); fig.add_trace(go.Scatter(x=df.timestamp,y=df.solar_kw,mode='lines',name='Solar',line=dict(color='#f5c36a',width=1.5))); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=350,margin=dict(l=5,r=5,t=25,b=5),legend=dict(orientation='h',y=1.12),xaxis_title=None,yaxis_title='Power (kW)'); st.plotly_chart(fig,use_container_width=True)
with c2:
 fig2=px.area(df.tail(min(168,len(df))),x='timestamp',y='occupancy',title='Occupancy profile',color_discrete_sequence=['#b99aff']); fig2.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=350,margin=dict(l=5,r=5,t=45,b=5),yaxis_title='Occupancy fraction'); st.plotly_chart(fig2,use_container_width=True)

st.markdown('<div class="section">02 / Day-ahead optimizer</div>',unsafe_allow_html=True)
a,b=st.columns([1.4,1])
with a:
 plan=pd.DataFrame({'timestamp':day.timestamp,'Forecast load (kW)':load,'Solar (kW)':solar,'GA battery (kW)':result['schedule'],'Tariff (₹/kWh)':prices})
 fig=go.Figure(); fig.add_trace(go.Bar(x=plan.timestamp,y=plan['GA battery (kW)'],name='Battery action',marker_color='#b99aff')); fig.add_trace(go.Scatter(x=plan.timestamp,y=plan['Forecast load (kW)'],name='Load',line=dict(color='#7ce0bd',width=2))); fig.add_trace(go.Scatter(x=plan.timestamp,y=plan['Solar (kW)'],name='Solar',line=dict(color='#f5c36a',width=2))); fig.update_layout(template='plotly_dark',barmode='relative',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=350,margin=dict(l=5,r=5,t=25,b=5),yaxis_title='kW',legend=dict(orientation='h',y=1.12)); st.plotly_chart(fig,use_container_width=True)
with b:
 st.markdown('#### Schedule diagnostics')
 st.write(f"**Optimizer:** {result['algorithm']}")
 st.write(f"**Planning horizon:** {len(day)} steps")
 st.write(f"**Estimated objective:** ₹{optimized_cost:,.2f}")
 st.write(f"**Energy-cost proxy without battery:** ₹{base_cost:,.2f}")
 st.write(f"**Battery throughput:** {result['detail']['throughput_kwh']:.1f} kWh")
 st.caption('Objective includes a terminal SOC penalty and battery degradation. The simple baseline proxy does not include all penalties; treat the percentage as a directional demonstration, not a fully matched controller comparison.')
 st.download_button('Export schedule CSV',plan.to_csv(index=False).encode(),file_name='gridmind_day_ahead_schedule.csv',mime='text/csv')

st.markdown('<div class="section">03 / Interpretable IS2 reasoning</div>',unsafe_allow_html=True)
f1,f2,f3=st.columns(3)
priority=fuzzy.fuzzy_cooling_priority(float(last.outdoor_temp_c),float(last.occupancy),float(last.price_per_kwh))
with f1: st.metric('Fuzzy cooling priority',f'{priority:.0%}',help='Triangular memberships and a simple interpretable rule base.')
with f2: st.metric('Peak load in window',f'{max(load):.1f} kW')
with f3: st.metric('Mean occupancy',f'{day.occupancy.mean():.0%}')
with st.expander('Inspect fuzzy rules and modelling assumptions'):
 st.markdown('- IF temperature is high AND occupancy is high AND price is low, THEN cooling priority is high.\n- Moderate priority responds to high temperature and occupancy.\n- The score is a small Mamdani-style illustrative controller, not a trained fuzzy system.\n- GA chromosome values are battery kW; positive means charge, negative means discharge.\n- The simulator enforces SOC and power bounds; this dashboard GA schedule is a day-ahead planning demonstration.')

st.markdown('<div class="section">04 / Model lab</div>',unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(['Forecasting','Controller training','Experiment log'])
with tab1:
 st.write('The research notebook trains the PyTorch LSTM with chronological splits and training-only normalization. This dashboard view shows observed history; it does not mislabel a heuristic as a trained forecast.')
 if st.button('Train LSTM now (small demo)',key='train_lstm'):
  with st.spinner('Training a compact LSTM…'):
   try:
    lstm=import_module('03_load_forecasting.lstm_forecaster'); tr,va,te=prep.chronological_split(clean)
    bundle=lstm.train_lstm(tr,va,epochs=3,lookback=24,horizon=24,seed=int(seed)); actual,pred=lstm.predict_windows(bundle,te); m=lstm.forecast_metrics(actual,pred)
    st.success('LSTM training complete.'); st.json(m); st.line_chart(pd.DataFrame({'actual':actual[0],'predicted':pred[0]}))
   except Exception as e: st.error(f'LSTM training unavailable: {e}')
with tab2:
 st.write('PPO uses a continuous action vector: battery kW, HVAC setpoint, EV charging power, and lighting fraction. Training can be compute-intensive.')
 steps=st.select_slider('PPO training timesteps',options=[2000,5000,10000,20000],value=5000)
 if st.button('Train and save PPO policy',key='train_ppo'):
  with st.spinner(f'Training PPO for {steps:,} timesteps…'):
   try:
    agent=import_module('07_rl_agent.ppo_agent'); model=agent.train_ppo(profile=day,total_timesteps=int(steps),seed=int(seed)); st.success('PPO model saved to artifacts/ppo_energy.zip')
   except Exception as e: st.error(f'PPO training failed or dependency missing: {e}')
 if use_ppo:
  if (ROOT/'artifacts/ppo_energy.zip').exists():
   if st.button('Run one PPO inference step'):
    try:
     env=import_module('05_energy_environment.building_env').BuildingEnergyEnv(profile=day,seed=int(seed)); obs,_=env.reset(seed=int(seed)); model=import_module('07_rl_agent.ppo_agent').load_ppo(str(ROOT/'artifacts/ppo_energy')); act,_=model.predict(obs,deterministic=True); _,reward,_,_,info=env.step(act); st.json({'action':np.asarray(act).tolist(),'reward':float(reward),'transition':info})
    except Exception as e: st.error(str(e))
  else: st.info('No trained PPO checkpoint found yet. Train a policy above.')
with tab3:
 if st.button('Save current GA experiment',key='save_exp'):
  report={'seed':int(seed),'data_mode':mode,'days':int(days),'ga_cost':optimized_cost,'baseline_proxy':base_cost,'ga_algorithm':result['algorithm'],'schedule':result['schedule'].tolist()}; storage.log_experiment(report); st.success('Experiment stored in SQLite.')
 rows=storage.read_experiments()
 if rows:
  st.dataframe(pd.DataFrame([{'id':r[0],'created_at':r[1],**__import__('json').loads(r[2])} for r in rows]),use_container_width=True)
 else: st.caption('No experiment records yet. Save a run to create the SQLite database.')
st.divider(); st.caption('GRIDMIND / IS2 research prototype · Simulator-only · Not connected to physical equipment · Synthetic variables clearly identified')
