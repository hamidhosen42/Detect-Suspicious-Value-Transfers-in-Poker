#!/usr/bin/env bash
# Full reproduction of the selected submission (final_J) from the raw competition files.
# Requirements: Python 3.12, polars, pandas, numpy, lightgbm, numba, scikit-learn, scipy. ~24 GB RAM, ~6-8 h on a 10-core CPU.
set -euo pipefail
export POKER_DATA_DIR=${POKER_DATA_DIR:-data}      # folder with hands/seats/actions.parquet, development_*.csv, evaluation_pairs.csv, sample_submission.csv
export POKER_WORK_DIR=${POKER_WORK_DIR:-work_step3}
mkdir -p "$POKER_WORK_DIR/hyp/eval"
python3 step9_pipeline.py                                   # shared caches + Step 9 pair features/OOF (dev_pair_features_v9, eval_pair_features_v9, dev_oof_v9)
python3 hyp/category_combo.py --phase dev        # evidence hypothesis features, development (writes work_step3/hyp/category_combo.parquet)
python3 hyp/street_timing.py                     #   (writes work_step3/hyp/street_timing.parquet)
PHASE=development python3 hyp/player_policy.py   #   (writes work_step3/hyp/player_policy.parquet)
python3 hyp/category_combo.py --phase eval --pairs "$POKER_WORK_DIR/eval_pairs.parquet" --out "$POKER_WORK_DIR/hyp/eval/category_combo_eval.parquet"
python3 hyp/street_timing.py --phase evaluation --pairs "$POKER_WORK_DIR/eval_pair_hands.parquet" --out "$POKER_WORK_DIR/hyp/eval/street_timing_eval.parquet"
PHASE=evaluation python3 hyp/player_policy.py && mv "$POKER_WORK_DIR/hyp/player_policy_eval.parquet" "$POKER_WORK_DIR/hyp/eval/"
python3 step18_pipeline.py                                  # Step 8 pair features + Step 8 pair model + Step 18 evidence -> work/submission.csv, eval_risk_v18, eval_pair_features_v18, dev_oof_v18
python3 reproduce/bag20.py                                  # 20-seed bags of the Step 8 and Step 9 pair rankers
python3 reproduce/make_final.py                             # -> final_submission.csv  (== deliverables_final/final_J_bags20_ev18.csv)
