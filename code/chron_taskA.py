import sys, json, numpy as np, pandas as pd
sys.argv=[sys.argv[0], sys.argv[1], sys.argv[2], "10"]
src=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'analyse2.py')).read().split("# ============================================================== TASK A")[0]
exec(src)
cut=d.groupby("pond").time.transform(lambda t: t.quantile(0.75)); trm=(d.time<=cut).values; tem=~trm
X=d[["pH","Temp","Turbidity","hour_sin","hour_cos"]].values; y=d.DO.values; g=d.pond.values; hr=d.time.dt.hour.values
out={}
for m in ML: out[m]=mets(y[tem], fit(m,X[trm],y[trm],g[trm])[0].predict(X[tem]))
out["Diel profile"]=mets(y[tem], pd.Series(y[trm]).groupby(hr[trm]).mean().reindex(hr[tem]).values)
out["Mean"]=mets(y[tem], np.full(tem.sum(), y[trm].mean()))
out["n_train_test"]=[int(trm.sum()), int(tem.sum())]
json.dump(out, open(sys.argv[2]+"/chron_taskA.json","w"), indent=1); print(json.dumps(out, indent=1))
