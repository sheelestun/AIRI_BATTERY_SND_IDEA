from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import urllib.request
BASE='https://raw.githubusercontent.com/petermattia/revisit-severson-et-al/main/data'
DATA=Path(__file__).parent/'data'
def fetch(rel):
 p=DATA/rel; p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): p.write_bytes(urllib.request.urlopen(f'{BASE}/{rel}',timeout=60).read())
 return p
def load(split):
 labels=np.loadtxt(fetch(f'cycle_lives/{split}_cycle_lives.csv')); out=[]
 for i,y in enumerate(labels,1):
  x=np.loadtxt(fetch(f'{split}/cell{i}.csv'),delimiter=',')
  out.append((i,x[:250].mean(0),float(y)))
 return out
def windows(split):
 labels=np.loadtxt(fetch(f'cycle_lives/{split}_cycle_lives.csv')); out=[]
 for i,y in enumerate(labels,1):
  x=np.loadtxt(fetch(f'{split}/cell{i}.csv'),delimiter=',')
  for j in range(4): out.append((i,j,x[j*250:(j+1)*250].mean(0),float(y)))
 return out
ROOT=Path(__file__).parent
train=windows('train'); test=load('test1')
# Build one profile per training cell; cluster only training profiles (no test leakage).
profiles=np.array([np.mean([r[2] for r in train if r[0]==i],axis=0) for i in range(1,42)])
scaler=StandardScaler().fit(profiles); z=scaler.transform(profiles)
km=KMeans(n_clusters=2,random_state=17,n_init=20).fit(z); groups=km.labels_
# Fixed N=12 and S=12. Diverse set is balanced across profile clusters; homogeneous set is drawn from the largest cluster.
counts=np.bincount(groups); majority=int(np.argmax(counts)); rng=np.random.default_rng(17)
diverse=[]
for g in range(2): diverse.extend(rng.choice(np.where(groups==g)[0],6,replace=False).tolist())
homogeneous=rng.choice(np.where(groups==majority)[0],12,replace=False).tolist()
testx=np.array([r[1] for r in test]); testy=np.array([r[2] for r in test]); test_profile=scaler.transform(testx); test_group=km.predict(test_profile)
def evaluate(ids,seed):
 # One segment per selected cell, same pre-defined first temporal window.
 x=np.array([next(r[2] for r in train if r[0]==i+1 and r[1]==0) for i in ids]); y=np.array([next(r[3] for r in train if r[0]==i+1 and r[1]==0) for i in ids])
 model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=.7,random_state=seed,n_jobs=-1).fit(x,y)
 p=model.predict(testx)
 group_mae={str(g):float(mean_absolute_error(testy[test_group==g],p[test_group==g])) for g in sorted(set(test_group)) if np.any(test_group==g)}
 return {'overall_MAE_cycles':float(mean_absolute_error(testy,p)),'cluster_MAE_cycles':group_mae,'predictions':p.tolist()}
d=evaluate(diverse,17); h=evaluate(homogeneous,23)
report={'dataset':'Severson processed feature profiles; 41 train cells, 43 fixed test1 cells','operating_condition_proxy':'KMeans clusters of standardized cell-mean 99D profiles; no explicit temperature/chemistry metadata used','design':'same N=12 cells and S=12 temporal feature segments (one first window per cell); diverse=6 per each of 2 clusters; homogeneous=12 from largest cluster','cluster_sizes':counts.tolist(),'largest_cluster':majority,'diverse':{k:v for k,v in d.items() if k!='predictions'},'homogeneous':{k:v for k,v in h.items() if k!='predictions'},'delta_MAE_diverse_minus_homogeneous':d['overall_MAE_cycles']-h['overall_MAE_cycles'],'limitations':'Feature-space clusters are only a surrogate for condition diversity, not known temperature/current/load metadata. All data are one LFP laboratory family; this cannot establish generalization to unseen operating conditions. One fixed draw per design makes this descriptive pilot, not a significance test.'}
(ROOT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
