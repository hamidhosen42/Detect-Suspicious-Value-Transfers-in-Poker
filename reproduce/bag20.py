"""Step A of the final assembly: 20-seed full-data retrains of the two pair rankers.

  (a) Step 8 pair features  (work/dev_pair_features_v8.parquet, work/eval_pair_features_v18.parquet, OOF work/dev_oof_v18.parquet)
  (b) Step 9 pair features  (work/dev_pair_features_v9.parquet, work/eval_pair_features_v9.parquet,  OOF work/dev_oof_v9.parquet)

Each retrain uses the same 3-stage PU label/weight recipe as the pipelines (pseudo-positives = unlabelled pairs whose OOF score is above the
25th percentile of the labelled positives' OOF, weight 1.0; "ambiguous" band above the 5th percentile weight 0.2; other unlabelled 0.1;
confirmed positives 5.0; confirmed negatives 1.0) and the pipelines' LightGBM parameters, trained on ALL development rows (both windows).
Usage:  POKER_WORK_DIR=work_step3 python3 reproduce/bag20.py
"""
import os, pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
W = os.environ.get("POKER_WORK_DIR", "work_step3").rstrip("/") + "/"
SEEDS = [42, 53, 65, 79, 88, 101, 113, 127, 139, 151, 211, 223, 229, 233, 239, 241, 251, 257, 263, 269]
META = {"pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk", "pred_family", "window", "pair_id_orig"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=os.cpu_count())


def bag(dev_file, ev_file, oof_file, out):
    dev = pd.read_parquet(W + dev_file); ev = pd.read_parquet(W + ev_file)
    oof = pd.read_parquet(W + oof_file).set_index("pair_id").reindex(dev.pair_id)
    F = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c != "top5s_same_loser"]
    assert all(c in ev.columns for c in F), "evaluation pair features do not match the development ones"
    y = dev.label.fillna(0).astype(int).values; labm = dev.is_labeled.values; o = oof.oof.values
    pos_oof = o[labm & (y == 1)]; t_amb, t_ps = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.25)
    unl = ~labm; pseudo = unl & (o >= t_ps); amb = unl & (o >= t_amb) & ~pseudo
    y2 = np.where(pseudo, 1, y); w = np.where(labm, np.where(y == 1, 5.0, 1.0), np.where(pseudo, 1.0, np.where(amb, 0.2, 0.1)))
    X = dev[F].astype("float32"); Xe = ev[F].astype("float32"); risk = np.zeros(len(ev))
    for i, sd in enumerate(SEEDS):
        risk += lgb.train({**P, "seed": sd}, lgb.Dataset(X, y2, weight=w), num_boost_round=750).predict(Xe) / len(SEEDS)
        print(f"{out}: seed {i + 1}/{len(SEEDS)}", flush=True)
    pd.DataFrame({"pair_id": ev.pair_id, "risk": risk}).to_parquet(W + out, index=False)


bag("dev_pair_features_v8.parquet", "eval_pair_features_v18.parquet", "dev_oof_v18.parquet", "final_bag20_step8.parquet")
bag("dev_pair_features_v9.parquet", "eval_pair_features_v9.parquet", "dev_oof_v9.parquet", "final_bag20_step9.parquet")
print("done")
