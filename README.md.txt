Reproduction package: Explainable ML for DO early warning (UNN Nsukka aquaponics IoT data)

Raw Data: Udanor et al. (2022), Data in Brief 43, 108400. 108400, https://doi.org/10.1016/j.dib.2022.108400 Put the 13 IoT_Pond*.csv files in ./raw

Steps (Python 3; pip install scikit-learn xgboost lightgbm shap pandas scipy matplotlib):
  python code/prepare.py raw results 1h            # QC + hourly aggregation -> results/clean_1h.csv, qc_report_1h.json
  python code/analyse2.py results/clean_1h.csv results 10   # Tasks A and B, LOPO, stats, SHAP, ablation, noise, alerts, figures
  python code/chron_taskA.py results/clean_1h.csv results   # chronological check for Task A (run from the package root)
  python code/hybrid.py results/clean_1h.csv results/hybrid_check.json   # sensor residual-correction check

Random seed 7. Full run takes about 45 minutes on 2 CPU cores.
