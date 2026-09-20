# Reproducing the selected submission

Selected submission: `deliverables_final/final_J_bags20_ev18.csv` (public 0.90025). The second selection, `final_F_allbag_ev18.csv`, uses the same evidence and a risk blend that additionally includes the fold-CV models written by `step18_pipeline.py` (see the end of this file).

## Environment
* Python 3.12; `polars >= 1.0`, `pandas`, `numpy`, `lightgbm >= 4`, `numba`, `scikit-learn`, `scipy`, `jupytext` (only to render notebooks).
* ~24 GB RAM, 10 CPU cores; total wall time ≈ 6–8 h. All stages cache to `POKER_WORK_DIR` and are skipped when their outputs exist.
* Competition files (`hands.parquet`, `seats.parquet`, `actions.parquet`, `players.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, `sample_submission.csv`) in `POKER_DATA_DIR` (default `data/`). `players.parquet` is read but no feature is derived from it.

## One command
```bash
POKER_DATA_DIR=data POKER_WORK_DIR=work_step3 bash reproduce/run_all.sh
```
writes `final_submission.csv`, identical in structure to `deliverables_final/final_J_bags20_ev18.csv` (LightGBM seeds are fixed; floating-point differences across machines can move a few risk values in the 6th decimal).

## What the steps do
| step | script | outputs (in `POKER_WORK_DIR`) |
|---|---|---|
| 1 | `step9_pipeline.py` | shared caches: `action_context`, `player_hands_*`, action-policy model → `action_surprise`, `player_hand_surprise`, hand strength, pair-hand maps, pair-hand features `dev_hand_features_v9_*` / `eval_hand_features_v9`, pair features `dev_pair_features_v9` / `eval_pair_features_v9`, pair-model OOF `dev_oof_v9` |
| 2 | `hyp/category_combo.py`, `hyp/street_timing.py`, `hyp/player_policy.py` (development, then evaluation) | evidence-ranker feature families in `hyp/` and `hyp/eval/` |
| 3 | `step18_pipeline.py` | Step 8 pair-hand/pair features (`*_v8`), the Step 8 pair model and family heads (`dev_oof_v18`, `eval_pair_features_v18`, `eval_risk_v18`), the two-stage evidence ranker, and `submission.csv` (its evidence columns are the final ones) |
| 4 | `reproduce/bag20.py` | 20-seed full-data bags of the two pair rankers: `final_bag20_step8.parquet`, `final_bag20_step9.parquet` |
| 5 | `reproduce/make_final.py` | rank-average of the two bags → `risk_score`; family heads from `eval_risk_v18` gated at the top 5 %; evidence from step 3 → `final_submission.csv` |

The notebooks in `deliverables_step*/` are `jupytext` renderings of the same scripts (`jupytext --to ipynb step18_pipeline.py`).

## Rules compliance
* Features use only gameplay: cards, board, positions, stacks, the ordered action log and hand chronology (`started_at`). Hand position within a phase (`hand_idx`) is derived by sorting `started_at`, never from IDs or file order.
* No use of ID formats, row/file ordering, `players.parquet` attributes, or anything about the generator.
* Unlabelled development pairs are treated as unknown (weak negatives / pseudo-positives), as the host clarified.

## `final_F` (second selection)
`final_F_allbag_ev18.csv` = rank-average of four risk vectors — the two 10-seed halves of the bags above and the fold-CV pair models from `step18_pipeline.py` (`eval_risk_v18.risk`) and `step9_pipeline.py` (`eval_risk_v9.risk`) — with the same evidence and family rule. Its construction is the `build("final_A_allbag_equal", ...)` call in the session script `work_step3/scripts/` (recorded in git); it differs from `final_J` only in the risk blend weights.
