import sys, json, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMRegressor
d=pd.read_csv(sys.argv[1],parse_dates=['time']).sort_values(['pond','time'])
out={}
for h in (1,3,6):
    fr=[]
    for p,g in d.groupby('pond'):
        g=g.set_index('time')[['DO','pH','Temp','Turbidity']].asfreq('1h')
        for L in (1,2,3): g[f'l{L}']=g.DO.shift(L)
        g['y']=g.DO.shift(-h); g['pond']=p; g['hn']=g.index.hour; g['ht']=(g.index.hour+h)%24
        fr.append(g.reset_index())
    f=pd.concat(fr).dropna(subset=['y','DO','l1','l2','l3','pH','Temp','Turbidity']).reset_index(drop=True)
    P={k:np.zeros(len(f)) for k in ['damped','hybrid_LGBM','anomaly_linear']}
    for p in f.pond.unique():
        tr,te=(f.pond!=p).values,(f.pond==p).values
        dt=d[d.pond!=p]; prof=dt.groupby(dt.time.dt.hour).DO.mean()
        pn=prof.reindex(f.hn).values; pt=prof.reindex(f.ht).values
        an=f.DO.values-pn; tgt=f.y.values-pt
        r=np.dot(an[tr],tgt[tr])/np.dot(an[tr],an[tr]); base=pt+r*an
        P['damped'][te]=base[te]
        F=np.c_[an, f.l1.values-prof.reindex((f.hn-1)%24).values, f.pH, f.Temp, f.Turbidity, pt]
        m=LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=4,min_child_samples=40,reg_lambda=10,verbose=-1).fit(F[tr],(f.y.values-base)[tr])
        P['hybrid_LGBM'][te]=base[te]+m.predict(F[te])
        lm=Ridge(1.0).fit(F[tr],tgt[tr]); P['anomaly_linear'][te]=pt[te]+lm.predict(F[te])
    y=f.y.values
    out[h]={k:{'R2':round(r2_score(y,v),3),'RMSE':round(float(np.sqrt(mean_squared_error(y,v))),3)} for k,v in P.items()}
print(json.dumps(out,indent=1))
json.dump(out,open(sys.argv[2],'w'),indent=1)
