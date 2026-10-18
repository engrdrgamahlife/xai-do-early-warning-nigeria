"""Quality control and aggregation of the UNN aquaponics IoT pond dataset (Udanor et al., 2022).
python -I prepare.py <raw_dir> <out_dir> [freq]
Outputs: clean_<freq>.csv and qc_report.json
"""
import sys, glob, re, json, numpy as np, pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
FREQ = sys.argv[3] if len(sys.argv) > 3 else "1h"
MIN_READINGS = 10           # minimum valid raw readings per window
LIMITS = {"Temp": (15, 40), "pH": (4, 11), "Turbidity": (0, 100), "DO": (0, 20)}  # DO: (0, 20] exclusive of 0

def std_cols(df):
    ren = {}
    for c in df.columns:
        l = c.lower().strip()
        if l.startswith("temp"): ren[c] = "Temp"
        elif "turb" in l: ren[c] = "Turbidity"
        elif l in ("do", "dissolved o2", "dssolved o2", "dissolved_o2"): ren[c] = "DO"
        elif l == "ph": ren[c] = "pH"
        elif "ammon" in l: ren[c] = "Ammonia"
        elif "trate" in l: ren[c] = "Nitrate"
        elif l == "created_at": ren[c] = "time"
    return df.rename(columns=ren)

qc = {"files": {}, "rules": {
    "Temp": "keep 15-40 degC; -127 is the DS18B20 disconnection code",
    "pH": "keep 4-11",
    "Turbidity": "keep 0-100 (sensor output range); windows where turbidity is pinned at 100 are flagged",
    "DO": "drop exact 0 (probe dropout) and values > 20 mg/L (above ~250% saturation at 25 degC, physically implausible)",
    "Ammonia/Nitrate": "excluded: non-finite values and magnitudes up to 1e4 mg/L indicate uncalibrated output",
    "aggregation": f"per pond, {FREQ} windows, median of valid readings, window kept if >= {MIN_READINGS} valid readings of every variable"}}
out = []
for f in sorted(glob.glob(f"{RAW}/*.csv")):
    name = f.split("/")[-1]; pond = int(re.search(r"(\d+)", name).group(1))
    df = std_cols(pd.read_csv(f, low_memory=False))
    info = {"pond": pond, "raw_rows": int(len(df))}
    need = {"time", "Temp", "pH", "Turbidity", "DO"}
    if not need.issubset(df.columns):
        info["excluded"] = f"missing columns {sorted(need - set(df.columns))} (no timestamps or unlabeled fields)"
        qc["files"][name] = info; continue
    df["time"] = pd.to_datetime(df["time"].astype(str).str.replace(" CET", ""), errors="coerce")
    for c in ["Temp", "pH", "Turbidity", "DO"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    info["span"] = [str(df.time.min()), str(df.time.max())]
    flags = {}
    for c, (lo, hi) in LIMITS.items():
        bad = (df[c] <= lo) if c == "DO" else (df[c] < lo)
        bad = bad | (df[c] > hi) | df[c].isna()
        flags[c] = float(bad.mean()); df.loc[bad, c] = np.nan
    info["fraction_invalid"] = {k: round(v, 3) for k, v in flags.items()}
    g = df.set_index("time")[["Temp", "pH", "Turbidity", "DO"]].resample(FREQ)
    med, cnt = g.median(), g.count()
    keep = (cnt >= MIN_READINGS).all(axis=1)
    w = med[keep].copy(); w["n_raw"] = cnt[keep].min(axis=1)
    w["DO_iqr"] = df.set_index("time")["DO"].resample(FREQ).quantile(.75)[keep] - \
                  df.set_index("time")["DO"].resample(FREQ).quantile(.25)[keep]
    w["pond"] = pond
    info["windows_kept"] = int(keep.sum())
    if info["windows_kept"] < 24:
        info["excluded"] = f"fewer than 24 valid {FREQ} windows"
        qc["files"][name] = info; continue
    info["turbidity_pinned_100_share"] = round(float((w.Turbidity >= 99.5).mean()), 3)
    qc["files"][name] = info
    out.append(w.reset_index())

d = pd.concat(out, ignore_index=True)
d["hour"] = d.time.dt.hour + d.time.dt.minute / 60
d["hour_sin"] = np.sin(2 * np.pi * d.hour / 24); d["hour_cos"] = np.cos(2 * np.pi * d.hour / 24)
d.to_csv(f"{OUT}/clean_{FREQ}.csv", index=False)
qc["final"] = {"rows": int(len(d)), "ponds": sorted(d.pond.unique().tolist()),
               "rows_per_pond": d.pond.value_counts().sort_index().to_dict(),
               "describe": d[["Temp", "pH", "Turbidity", "DO"]].describe().round(3).to_dict(),
               "median_within_window_DO_IQR": float(d.DO_iqr.median())}
json.dump(qc, open(f"{OUT}/qc_report_{FREQ}.json", "w"), indent=2, default=str)
print(json.dumps(qc["final"], indent=1, default=str))
print({k: v.get("excluded", v.get("windows_kept")) for k, v in qc["files"].items()})
