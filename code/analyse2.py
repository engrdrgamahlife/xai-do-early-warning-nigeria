"""Full analysis for the revised manuscript.
Task A: soft-sensor estimation of DO(t) from pH, temperature, turbidity (+ time of day).
Task B: early-warning forecasting of DO(t+h), h = 1, 3, 6 h, from DO history + sensors + time.
Primary validation: leave-one-pond-out (LOPO) with nested GroupKFold tuning.
python -I analyse2.py <clean_1h.csv> <out_dir> [n_iter]
"""
import sys, os, json, time, pickle, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 7; rng = np.random.default_rng(SEED)
PATH, OUT = sys.argv[1], sys.argv[2]; NIT = int(sys.argv[3]) if len(sys.argv) > 3 else 15
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
d = pd.read_csv(PATH, parse_dates=["time"]).sort_values(["pond", "time"]).reset_index(drop=True)
R = {"n_rows": int(len(d)), "ponds": sorted(map(int, d.pond.unique())), "n_iter": NIT}

SPACE = {
    "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [5, 10, 20, 40], "max_features": [0.5, 0.8, 1.0]}),
    "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
        {"n_estimators": [100, 300, 600], "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.03, 0.1],
         "subsample": [0.7, 1.0], "min_child_weight": [5, 10, 20], "reg_lambda": [1, 10, 50]}),
    "LightGBM": (LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
        {"n_estimators": [100, 300, 600], "num_leaves": [4, 8, 15], "learning_rate": [0.01, 0.03, 0.1],
         "min_child_samples": [10, 20, 40, 80], "subsample": [0.7, 1.0], "subsample_freq": [1], "reg_lambda": [1, 10, 50]}),
    "MLP": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=4000, early_stopping=True, n_iter_no_change=40, random_state=SEED)),
        {"mlpregressor__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
         "mlpregressor__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0], "mlpregressor__learning_rate_init": [1e-3, 3e-3, 1e-2]}),
}
ML = list(SPACE)

def fit(name, X, y, groups):
    est, grid = SPACE[name]
    if not grid: return clone(est).fit(X, y), {}
    s = RandomizedSearchCV(clone(est), grid, n_iter=NIT, cv=GroupKFold(n_splits=4), scoring="neg_root_mean_squared_error",
                           random_state=SEED, n_jobs=-1).fit(X, y, groups=groups)
    return s.best_estimator_, {k: str(v) for k, v in s.best_params_.items()}

def mets(y, p):
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p))}

def summarise(y, ponds, preds):
    up = np.unique(ponds); res = {}
    for m, pr in preds.items():
        per = [mets(y[ponds == p], pr[ponds == p]) for p in up]
        res[m] = {"pooled": mets(y, pr), "per_pond_RMSE": [x["RMSE"] for x in per],
                  "mean_pond_RMSE": float(np.mean([x["RMSE"] for x in per]))}
    return res

def cluster_ci(y, ponds, pred, B=2000):
    up = np.unique(ponds); out = {"R2": [], "RMSE": [], "MAE": []}
    idxs = {p: np.where(ponds == p)[0] for p in up}
    for _ in range(B):
        idx = np.concatenate([idxs[q] for q in rng.choice(up, len(up), replace=True)])
        m = mets(y[idx], pred[idx]); [out[k].append(m[k]) for k in out]
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}

def tests_vs(res, ref, names):
    M = {m: np.array(res[m]["per_pond_RMSE"]) for m in names}
    rows = []
    for m in names:
        if m == ref: continue
        diff = M[m] - M[ref]
        p = float(stats.wilcoxon(M[m], M[ref]).pvalue) if np.any(diff != 0) else 1.0
        rows.append({"model": m, "median_RMSE_diff": float(np.median(diff)), "ponds_ref_better": int((diff > 0).sum()), "p": p})
    ps = np.array([r["p"] for r in rows]); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for k, i in enumerate(order): run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1, run)
    for r, a in zip(rows, adj): r["p_holm"] = float(a)
    fr = stats.friedmanchisquare(*[M[m] for m in names])
    return {"reference": ref, "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "pairwise": rows}

# ============================================================== TASK A: soft sensor
SENS = ["pH", "Temp", "Turbidity"]; TIME = ["hour_sin", "hour_cos"]
yA = d.DO.values; gA = d.pond.values; hrA = d.time.dt.hour.values; up = np.unique(gA)
A = {}
for fs, cols in {"Sensors": SENS, "Sensors+Time": SENS + TIME}.items():
    X = d[cols].values
    preds = {m: np.zeros(len(d)) for m in ML + ["Mean", "Diel profile"]}
    for p in up:
        tr, te = gA != p, gA == p
        for m in ML: preds[m][te] = fit(m, X[tr], yA[tr], gA[tr])[0].predict(X[te])
        preds["Mean"][te] = yA[tr].mean()
        preds["Diel profile"][te] = pd.Series(yA[tr]).groupby(hrA[tr]).mean().reindex(hrA[te]).values
    res = summarise(yA, gA, preds)
    best = min(ML, key=lambda m: res[m]["pooled"]["RMSE"])
    A[fs] = {"results": res, "best": best, "best_CI": cluster_ci(yA, gA, preds[best]),
             "diel_CI": cluster_ci(yA, gA, preds["Diel profile"]),
             "tests_vs_diel": tests_vs(res, "Diel profile", ML + ["Mean", "Diel profile"]),
             "alert_AUC_DO<5": {m: float(roc_auc_score(yA < 5, -preds[m])) for m in ML + ["Diel profile"]}}
    print("Task A", fs, "done", flush=True)
# explain Task A with a tuned XGBoost on all data (descriptive explanation of what the sensors carry)
XA = d[SENS + TIME].values
xa, _ = fit("XGBoost", XA, yA, gA)
svA = shap.TreeExplainer(xa).shap_values(XA)
A["SHAP_XGB_all"] = {"mean_abs": dict(zip(SENS + TIME, np.abs(svA).mean(0).round(4).tolist())),
                     "time_combined": float(np.abs(svA[:, 3] + svA[:, 4]).mean()),
                     "direction": {c: float(stats.spearmanr(XA[:, i], svA[:, i]).correlation) for i, c in enumerate(SENS + TIME)}}
# within-pond partial association (removes pond offsets and diel cycle)
dd = d.copy()
for c in ["DO", "pH", "Temp", "Turbidity"]:
    dd[c + "_res"] = dd[c] - dd.groupby(["pond", dd.time.dt.hour])[c].transform("mean")
A["within_pond_anomaly_spearman_with_DO"] = {c: float(stats.spearmanr(dd[c + "_res"], dd["DO_res"]).correlation) for c in SENS}
R["TaskA"] = A

# ============================================================== TASK B: forecasting
def make_lags(h):
    fr = []
    for p, g in d.groupby("pond"):
        g = g.set_index("time")[["DO", "pH", "Temp", "Turbidity"]].asfreq("1h")
        for L in (1, 2, 3): g[f"DO_lag{L}"] = g.DO.shift(L)
        g["y"] = g.DO.shift(-h); g["pond"] = p
        th = (g.index.hour + h) % 24
        g["tgt_hour_sin"] = np.sin(2 * np.pi * th / 24); g["tgt_hour_cos"] = np.cos(2 * np.pi * th / 24)
        g["hour_now"] = g.index.hour; g["hour_tgt"] = th
        fr.append(g.reset_index())
    f = pd.concat(fr)
    return f.dropna(subset=["y", "DO", "DO_lag1", "DO_lag2", "DO_lag3", "pH", "Temp", "Turbidity"]).reset_index(drop=True)

BF = ["DO", "DO_lag1", "DO_lag2", "DO_lag3", "pH", "Temp", "Turbidity", "tgt_hour_sin", "tgt_hour_cos"]
B = {}; keep = {}
for h in (1, 3, 6):
    f = make_lags(h); y = f.y.values; g = f.pond.values; X = f[BF].values
    preds = {m: np.zeros(len(f)) for m in ML + ["Persistence", "Diel profile", "Damped anomaly"]}
    for p in np.unique(g):
        tr, te = g != p, g == p
        for m in ML: preds[m][te] = fit(m, X[tr], y[tr], g[tr])[0].predict(X[te])
        prof = pd.Series(d.DO.values[d.pond.values != p]).groupby(d.time.dt.hour.values[d.pond.values != p]).mean()
        preds["Persistence"][te] = f.DO.values[te]
        preds["Diel profile"][te] = prof.reindex(f.hour_tgt.values[te]).values
        an_tr = f.DO.values[tr] - prof.reindex(f.hour_now.values[tr]).values
        tgt_tr = y[tr] - prof.reindex(f.hour_tgt.values[tr]).values
        r = float(np.dot(an_tr, tgt_tr) / np.dot(an_tr, an_tr))
        preds["Damped anomaly"][te] = prof.reindex(f.hour_tgt.values[te]).values + r * (f.DO.values[te] - prof.reindex(f.hour_now.values[te]).values)
    res = summarise(y, g, preds)
    best = min(ML, key=lambda m: res[m]["pooled"]["RMSE"])
    bestbase = min(["Persistence", "Diel profile", "Damped anomaly"], key=lambda m: res[m]["pooled"]["RMSE"])
    act = y < 5
    alerts = {}
    for m in [best, "Persistence", "Damped anomaly"]:
        a = {"AUC": float(roc_auc_score(act, -preds[m]))}
        for thr in (5.0, 6.0):
            pr = preds[m] < thr; tp, fp, fn = int((pr & act).sum()), int((pr & ~act).sum()), int((~pr & act).sum())
            a[f"rule_pred<{thr}"] = {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "alerts": int(pr.sum())}
        alerts[m] = a
    B[f"h{h}"] = {"n": int(len(f)), "n_low_DO<5": int(act.sum()), "results": res, "best": best, "best_baseline": bestbase,
                  "best_CI": cluster_ci(y, g, preds[best]), "baseline_CI": cluster_ci(y, g, preds[bestbase]),
                  "tests_vs_best_ML": tests_vs(res, best, ML + ["Persistence", "Diel profile", "Damped anomaly"]),
                  "alerts_DO<5": alerts}
    keep[h] = (f, preds, best)
    print("Task B h", h, "done", flush=True)

# ---- Task B explanation, ablation, robustness at h = 3
f3, p3, best3 = keep[3]; y3 = f3.y.values; g3 = f3.pond.values; X3 = f3[BF].values
fam = best3 if best3 in ("Random Forest", "XGBoost", "LightGBM") else "XGBoost"
m3, bp3 = fit(fam, X3, y3, g3)
sv3 = shap.TreeExplainer(m3).shap_values(X3)
labels = ["DO(t)", "DO(t-1)", "DO(t-2)", "DO(t-3)", "pH", "Temperature", "Turbidity", "Target hour (sin)", "Target hour (cos)"]
B["SHAP_h3"] = {"model": fam, "params": bp3, "mean_abs": dict(zip(BF, np.abs(sv3).mean(0).round(4).tolist())),
                "share_DO_history": float(np.abs(sv3[:, :4]).mean(0).sum() / np.abs(sv3).mean(0).sum()),
                "share_sensors": float(np.abs(sv3[:, 4:7]).mean(0).sum() / np.abs(sv3).mean(0).sum()),
                "share_time": float(np.abs(sv3[:, 7:]).mean(0).sum() / np.abs(sv3).mean(0).sum()),
                "direction": {c: float(stats.spearmanr(X3[:, i], sv3[:, i]).correlation) for i, c in enumerate(BF)}}
plt.figure(); shap.summary_plot(sv3, X3, feature_names=labels, show=False, plot_size=(5.2, 3.6))
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_shap_forecast_h3.png", dpi=300); plt.close()
fig = plt.figure(); shap.summary_plot(svA, XA, feature_names=["pH", "Temperature", "Turbidity", "Hour (sin)", "Hour (cos)"], show=False, plot_size=(5.2, 3.0))
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_shap_softsensor.png", dpi=300); plt.close()

def lopo_fixed(cols):
    Xc = f3[cols].values; pr = np.zeros(len(f3))
    for p in np.unique(g3):
        tr, te = g3 != p, g3 == p
        pr[te] = clone(m3).fit(Xc[tr], y3[tr]).predict(Xc[te])
    return mets(y3, pr)
DOH = ["DO", "DO_lag1", "DO_lag2", "DO_lag3"]; TT = ["tgt_hour_sin", "tgt_hour_cos"]
B["ablation_h3"] = {"model": fam, "all": lopo_fixed(BF), "DO_history+time": lopo_fixed(DOH + TT),
                    "DO_history_only": lopo_fixed(DOH), "DO(t)+time": lopo_fixed(["DO"] + TT),
                    "sensors+time (no DO history)": lopo_fixed(SENS + TT),
                    **{f"drop_{c}": lopo_fixed([x for x in BF if x != c]) for c in SENS}}
# robustness: noise injected into LOPO test inputs (model trained on clean data of other ponds)
spec = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 5.0, 0, 0])
models_lopo = {p: clone(m3).fit(X3[g3 != p], y3[g3 != p]) for p in np.unique(g3)}
def noisy_rmse(scale_vec, reps=30):
    out = []
    for _ in range(reps):
        pr = np.zeros(len(f3))
        for p, mm in models_lopo.items():
            te = g3 == p; pr[te] = mm.predict(X3[te] + rng.normal(0, 1, X3[te].shape) * scale_vec)
        out.append(np.sqrt(mean_squared_error(y3, pr)))
    return float(np.mean(out)), float(np.std(out))
levels = [0, 0.5, 1, 2, 3]
B["noise_h3"] = {"spec_1x": dict(zip(BF, spec.tolist())),
                 "all_inputs": {str(k): noisy_rmse(spec * k) for k in levels},
                 "sensors_only": {str(k): noisy_rmse(spec * np.array([0, 0, 0, 0, 1, 1, 1, 0, 0]) * k) for k in levels},
                 "DO_only": {str(k): noisy_rmse(spec * np.array([1, 1, 1, 1, 0, 0, 0, 0, 0]) * k) for k in levels}}
blob = pickle.dumps(m3); t0 = time.perf_counter()
for _ in range(2000): m3.predict(X3[:1])
B["edge_h3"] = {"model": fam, "bytes": len(blob), "ms_per_prediction": (time.perf_counter() - t0) / 2000 * 1000,
                "n_trees": int(getattr(m3, "n_estimators", 0))}
open(f"{OUT}/forecast_h3_model.pkl", "wb").write(blob)
R["TaskB"] = B

# ---- figures
v = d.assign(h=d.time.dt.hour); gq = v.groupby("h").DO
fig, ax = plt.subplots(figsize=(5, 2.9))
ax.fill_between(range(24), gq.quantile(.25), gq.quantile(.75), color="#a9cce3", label="Interquartile range")
ax.plot(range(24), gq.median(), color="#1b4f72", lw=2, label="Median"); ax.axhline(5, ls="--", c="#c0392b", lw=1)
ax.text(0.3, 4.4, "5 mg/L", color="#c0392b", fontsize=8); ax.set_xlim(0, 23)
ax.set_xlabel("Hour of day"); ax.set_ylabel("DO (mg/L)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_diel_DO.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(5.2, 3))
hs = [1, 3, 6]
for name, col, mk in [("best", "#1b4f72", "o"), ("Persistence", "#c0392b", "s"), ("Damped anomaly", "#7f8c8d", "^")]:
    vals = [B[f"h{h}"]["results"][B[f"h{h}"]["best"] if name == "best" else name]["pooled"]["RMSE"] for h in hs]
    ax.plot(hs, vals, marker=mk, color=col, label="Best ML model" if name == "best" else name)
ax.set_xticks(hs); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("LOPO RMSE (mg/L)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_forecast_rmse.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for key, lab, col in [("all_inputs", "All inputs", "#1b4f72"), ("sensors_only", "pH, temperature, turbidity", "#2e86c1"), ("DO_only", "DO inputs", "#c0392b")]:
    ax.plot(levels, [B["noise_h3"][key][str(k)][0] for k in levels], marker="o", label=lab, color=col)
ax.set_xlabel("Injected noise (× sensor accuracy)"); ax.set_ylabel("RMSE, h = 3 (mg/L)"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_noise_h3.png", dpi=300); plt.close()

json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=float)
print("ALL DONE", flush=True)
