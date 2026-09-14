"""Fixed, table-held-out P/N baseline; deterministic, unlearned evidence retrieval."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import inspect
import platform
import time

import numpy as np
import pandas as pd
import pyarrow
import sklearn
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from official_metric import score, _average_precision, EVIDENCE_COLUMNS

FAMILIES = ['directed_transfer', 'soft_play', 'coordinated_isolation']
SEED = 2026


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evidence_predictions(candidates):
    """Return top-five without labels, model predictions, or risk gating."""
    ranked = candidates.sort_values(['key', 'heuristic', 'hand_id'], ascending=[True, False, True])
    ranked = ranked.drop_duplicates(['key', 'hand_id']).groupby('key', sort=False).head(5).copy()
    ranked['slot'] = ranked.groupby('key').cumcount() + 1
    wide = ranked.pivot(index='key', columns='slot', values='hand_id').reindex(columns=range(1, 6))
    wide.columns = list(EVIDENCE_COLUMNS)
    return wide.fillna('NO_EVIDENCE').reset_index()


def build_features(root, out):
    print('Loading selected hand/seat columns', flush=True)
    hands = pd.read_parquet(root/'hands.parquet')
    seats = pd.read_parquet(root/'seats.parquet', columns=[
        'hand_id', 'player_id', 'net_chips', 'total_contribution', 'folded', 'went_to_showdown'])
    players = pd.Index(sorted(seats.player_id.unique()))
    nplayers = len(players)
    hid = pd.Index(hands.hand_id)
    assert hid.is_unique
    hi = hid.get_indexer(seats.hand_id)
    pi = players.get_indexer(seats.player_id)
    assert (hi >= 0).all() and (pi >= 0).all()
    order = np.lexsort((pi, hi))
    assert len(seats) == 6 * len(hands)
    assert np.array_equal(np.bincount(hi), np.full(len(hands), 6))
    p = pi[order].reshape(-1, 6)
    assert (np.diff(p, axis=1) > 0).all()
    arrays = {c: seats[c].to_numpy()[order].reshape(-1, 6) for c in
              ['net_chips', 'total_contribution', 'folded', 'went_to_showdown']}
    assert (arrays['net_chips'].sum(axis=1) == 0).all()
    membership = pd.DataFrame({'player': p.ravel(), 'table': np.repeat(hands.table_id.to_numpy(), 6)})
    assert membership.groupby('player').table.nunique().eq(1).all()
    del seats, hi, pi, order, membership
    labels = pd.read_csv(root/'development_labels.csv')
    evaluation = pd.read_csv(root/'evaluation_pairs.csv')
    for frame in [labels, evaluation]:
        a, b = players.get_indexer(frame.player_1), players.get_indexer(frame.player_2)
        assert (a >= 0).all() and (b >= 0).all()
        frame['key'] = np.minimum(a, b).astype(np.int64)*nplayers + np.maximum(a, b)
        assert frame.key.is_unique
    labels.to_parquet(out/'labels.parquet', index=False)
    evaluation.to_parquet(out/'evaluation.parquet', index=False)
    eval_keys = set(evaluation.key)
    dev_keys = set(labels.key)
    i, j = np.array(list(itertools.combinations(range(6), 2))).T
    feats, evidence = [], []
    for ti, (table, indices) in enumerate(hands.groupby('table_id', sort=True).indices.items()):
        for phase in ['development', 'evaluation']:
            ix = indices[hands.phase.to_numpy()[indices] == phase]
            bb = hands.big_blind.to_numpy()[ix, None]
            net = arrays['net_chips'][ix]/bb
            contrib = arrays['total_contribution'][ix]/bb
            pl = p[ix]
            a, b = net[:, i], net[:, j]
            ca, cb = contrib[:, i], contrib[:, j]
            f = arrays['folded'][ix]
            sd = arrays['went_to_showdown'][ix]
            t12 = np.minimum(np.maximum(-a, 0), np.maximum(b, 0))
            t21 = np.minimum(np.maximum(-b, 0), np.maximum(a, 0))
            values = {
                'flow12': t12, 'flow21': t21,
                'transfer': t12+t21, 'net_gap': np.abs(a-b),
                'joint_net': a+b, 'contrib_gap': np.abs(ca-cb),
                'min_contrib': np.minimum(ca, cb), 'pair_contrib': ca+cb,
                'opposite_net': a*b < 0, 'both_showdown': sd[:, i] & sd[:, j],
                'both_fold': f[:, i] & f[:, j], 'one_fold': f[:, i] ^ f[:, j],
                'both_invested': (ca > 1) & (cb > 1),
            }
            frame = pd.DataFrame({k: v.ravel().astype(np.float32) for k,v in values.items()})
            frame['key'] = (pl[:, i].astype(np.int64)*nplayers + pl[:, j]).ravel()
            group = frame.groupby('key', sort=True)
            agg = group[list(values)].agg(['mean','std','max'])
            agg.columns = [f'{a}_{b}' for a,b in agg.columns]
            agg['shared_hands'] = group.size()
            agg['direction_imbalance'] = (agg.flow12_mean-agg.flow21_mean).abs()/(agg.transfer_mean+1)
            # Raw direction slots depend on arbitrary ID ordering; keep only symmetric summaries.
            agg = agg.drop(columns=[c for c in agg if c.startswith(('flow12_', 'flow21_'))])
            agg['table_id'] = table
            agg['phase'] = phase
            agg = agg.fillna(0).reset_index()
            if phase == 'development':
                agg = agg[(agg.shared_hands >= 57) | agg.key.isin(dev_keys)]
            else:
                agg = agg[agg.key.isin(eval_keys)]
            feats.append(agg)
            frame['hand_id'] = np.repeat(hands.hand_id.to_numpy()[ix], 15)
            # Fixed initial heuristic, deliberately not tuned on evidence labels.
            frame['heuristic'] = (np.log1p(frame.transfer) + .25*np.log1p(frame.pair_contrib)
                                  + .5*frame.both_showdown + .25*frame.both_invested)
            ev = evidence_predictions(frame[frame.key.isin(agg.key)][['key','hand_id','heuristic']])
            ev['phase'] = phase
            evidence.append(ev)
        if (ti+1) % 40 == 0:
            print(f'Features/evidence: {ti+1}/400 tables', flush=True)
    pd.concat(feats, ignore_index=True).to_parquet(out/'features.parquet', index=False)
    pd.concat(evidence, ignore_index=True).to_parquet(out/'evidence.parquet', index=False)


def predictions(rows, risk, behavior, ev):
    result = rows[['pair_id','key']].copy()
    result['risk_score'] = risk
    result['predicted_behavior'] = behavior
    return result.merge(ev.drop(columns='phase'), on='key', validate='one_to_one').drop(columns='key')


def truth_for(rows, evidence):
    truth = rows[['pair_id','label','behavior_family']].rename(
        columns={'label':'risk_score','behavior_family':'predicted_behavior'}).copy()
    wide = evidence.pivot(index='pair_id',columns='evidence_rank',values='hand_id').reindex(columns=range(1,6))
    wide.columns = list(EVIDENCE_COLUMNS)
    return truth.merge(wide, on='pair_id', how='left').fillna('NO_EVIDENCE')


def components(truth, pred):
    total = score(truth, pred, 'pair_id')
    t = truth.set_index('pair_id').sort_index()
    s = pred.set_index('pair_id').loc[t.index]
    pair = _average_precision(t.risk_score.to_numpy(), s.risk_score.to_numpy())
    behavior = np.mean([_average_precision((t.predicted_behavior == f).to_numpy(),
                        np.where(s.predicted_behavior == f, s.risk_score, 0)) for f in FAMILIES])
    return {'pair_ap':float(pair),'evidence_map5':float((total-.7*pair-.1*behavior)/.2),
            'behavior_map':float(behavior),'composite':float(total)}


def model(multiclass=False):
    return lgb.LGBMClassifier(objective='multiclass' if multiclass else 'binary',
        n_estimators=250, learning_rate=.03, num_leaves=15, max_depth=5,
        min_child_samples=25, reg_lambda=5., colsample_bytree=.8,
        random_state=SEED, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)


def train(root, out):
    all_f = pd.read_parquet(out/'features.parquet')
    labels = pd.read_parquet(out/'labels.parquet')
    evaluation = pd.read_parquet(out/'evaluation.parquet')
    ev = pd.read_parquet(out/'evidence.parquet')
    dev = all_f[all_f.phase.eq('development')].merge(labels[['key','pair_id','label','behavior_family']],on='key',how='left')
    known = dev.label.notna()
    # Synthetic diagnostic IDs are keys only, never model inputs.
    dev['pair_id'] = dev.pair_id.fillna('UNLABELED_'+dev.key.astype(str))
    dev['label'] = dev.label.fillna(0).astype(int)
    dev['behavior_family'] = dev.behavior_family.fillna('none')
    features = [c for c in all_f if c not in ['key','table_id','phase']]
    # Define outer fold assignment from trusted labels, then map entire tables.
    trusted = dev[known]
    splitter = StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    mapping = {}
    for fold,(_, va) in enumerate(splitter.split(trusted, trusted.behavior_family, trusted.table_id)):
        mapping.update({table:fold for table in trusted.iloc[va].table_id.unique()})
    for table in sorted(set(dev.table_id)-set(mapping)):
        mapping[table] = len(mapping) % 5
    dev['fold'] = dev.table_id.map(mapping)
    risk = np.zeros(len(dev))
    behavior = np.full(len(dev),'none',dtype=object)
    folds=[]
    for fold in range(5):
        tr = known & dev.fold.ne(fold)
        va = dev.fold.eq(fold)
        assert set(dev.loc[tr,'table_id']).isdisjoint(dev.loc[va,'table_id'])
        assert set(dev.loc[tr & dev.label.eq(1),'behavior_family']) == set(FAMILIES)
        m=model();m.fit(dev.loc[tr,features],dev.loc[tr,'label'])
        risk[va]=m.predict_proba(dev.loc[va,features])[:,1]
        pos=tr & dev.label.eq(1)
        b=model(True);b.fit(dev.loc[pos,features],dev.loc[pos,'behavior_family'])
        behavior[va]=b.predict(dev.loc[va,features])
        subset=dev[va & known]
        pr=predictions(subset,risk[va & known],behavior[va & known],ev[ev.phase.eq('development')])
        met=components(truth_for(subset,pd.read_csv(root/'development_evidence.csv')),pr)
        folds.append({'fold':fold,'train_labeled':int(tr.sum()),'valid_labeled':len(subset),**met})
        print('Fold',fold,met,flush=True)
    pr=predictions(dev,risk,behavior,ev[ev.phase.eq('development')])
    pr.to_csv(out/'oof_predictions.csv',index=False)
    dev[['pair_id','table_id','fold']].assign(is_labeled=known).to_csv(out/'folds.csv',index=False)
    t=truth_for(dev,pd.read_csv(root/'development_evidence.csv'))
    metrics={'confirmed_labels':components(t[known.to_numpy()],pr[pr.pair_id.isin(dev.loc[known,'pair_id'])]),
             'unlabeled_as_negative_stress':{
                 'pair_ap':float(average_precision_score(dev.label,risk)),
                 'ap_definition':'sklearn threshold-grouped AP; unknown development pair IDs unavailable',
                 'behavior_and_composite':'Not reported: private labels and original unlabeled IDs unavailable'},'folds':folds,
             'limitations':['Stress labels are contaminated, not evaluation ground truth.',
               'Fixed seat-only heuristic evidence; no learned evidence model or stacking.',
               'No hidden-family detector; behavior always routes to a known family.',
               'Fixed hyperparameters, no outer-fold early stopping or threshold selection.'],
             'feature_columns':features}
    m=model();m.fit(dev.loc[known,features],dev.loc[known,'label'])
    b=model(True);pos=known & dev.label.eq(1);b.fit(dev.loc[pos,features],dev.loc[pos,'behavior_family'])
    m.booster_.save_model(str(out/'risk_model.txt'));b.booster_.save_model(str(out/'behavior_model.txt'))
    exposure=evaluation[['key','shared_hands']].merge(
        all_f[all_f.phase.eq('evaluation')][['key','shared_hands']],on='key',suffixes=('_given','_computed'),validate='one_to_one')
    assert len(exposure)==len(evaluation) and exposure.shared_hands_given.eq(exposure.shared_hands_computed).all()
    test=evaluation.merge(all_f[all_f.phase.eq('evaluation')].drop(columns='shared_hands'),on='key',validate='one_to_one')
    sub=predictions(test,m.predict_proba(test[features])[:,1],b.predict(test[features]),ev[ev.phase.eq('evaluation')])
    sample=pd.read_csv(root/'sample_submission.csv')
    sub=sample[['pair_id']].merge(sub,on='pair_id',validate='one_to_one')[sample.columns]
    sub.to_csv(out/'submission.csv',index=False,float_format='%.17g')
    validate(root,out)
    metrics['rows']={'labeled':int(known.sum()),'unlabeled_stress':int((~known).sum()),'evaluation':len(sub)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    c=metrics['confirmed_labels'];stress=metrics['unlabeled_as_negative_stress']['pair_ap']
    report=['# Baseline validation report','',
        'Measured locally. Five table-disjoint outer folds; fixed hyperparameters; trusted P/N training only.',
        'Evidence is a fixed seat-based heuristic. There is no learned evidence stacking, threshold tuning, or early stopping.',
        '', '| Population / metric | Value |','|---|---:|',
        *[f'| Confirmed-label {k} | {v:.6f} |' for k,v in c.items()],
        f'| Unlabeled-as-negative stress pair AP (sklearn) | {stress:.6f} |','',
        '## Outer folds','', '| Fold | Training labels | Validation labels | Pair AP | Evidence MAP@5 | Behavior MAP | Composite |',
        '|---:|---:|---:|---:|---:|---:|---:|',
        *[f"| {f['fold']} | {f['train_labeled']} | {f['valid_labeled']} | {f['pair_ap']:.6f} | {f['evidence_map5']:.6f} | {f['behavior_map']:.6f} | {f['composite']:.6f} |" for f in folds],
        '', '## Interpretation','',
        'The labeled composite is a development benchmark, not a leaderboard forecast. The stress set includes all eligible development background pairs, including pairs involving positive players; it does not exactly reproduce evaluation selection. Unknown pairs are provisionally negative, so stress AP is contaminated. No stress behavior/composite is reported.',
        'All behavior assignments use held-out predictions. All three known families are eligible at any risk; no hidden-family detector is implemented.',
        'The evidence heuristic emphasizes chip-flow proxies and may miss soft play and isolation. Net-flow proxies do not identify exact bilateral transfers in multiway pots.',
        '', '## Submission checks','',
        f'Passed: {len(sub):,} unique required pairs; finite risks in [0,1]; valid behaviors; five distinct shared evaluation hands per pair; recomputed exposure equals the supplied shared_hands.',
        '', 'No Kaggle upload was performed. See manifest.json for source/data hashes and versions.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    return metrics


def validate(root,out):
    sub=pd.read_csv(out/'submission.csv',keep_default_na=False)
    pairs=pd.read_csv(root/'evaluation_pairs.csv')
    assert len(sub)==len(pairs) and sub.pair_id.is_unique and set(sub.pair_id)==set(pairs.pair_id)
    assert sub[list(EVIDENCE_COLUMNS)].ne('NO_EVIDENCE').all().all()
    fake=sub.copy();fake.risk_score=0;fake.predicted_behavior='none'
    score(fake,sub,'pair_id')
    long=sub.melt(id_vars='pair_id',value_vars=list(EVIDENCE_COLUMNS),value_name='hand_id')
    hands=pd.read_parquet(root/'hands.parquet',columns=['hand_id','phase'])
    assert long.merge(hands,on='hand_id',validate='many_to_one').phase.eq('evaluation').all()
    assert long.hand_id.isin(hands.hand_id).all()
    seats=pd.read_parquet(root/'seats.parquet',columns=['hand_id','player_id'])
    lookup=pd.MultiIndex.from_frame(seats)
    target=long.merge(pairs,on='pair_id',validate='many_to_one')
    for side in ['player_1','player_2']:
        query=pd.MultiIndex.from_arrays([target.hand_id,target[side]])
        assert (lookup.get_indexer(query)>=0).all()
    print(f'Validated {len(sub):,} pairs and {len(long):,} shared evaluation evidence entries',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.data_dir.resolve();out=(args.output_dir or root/'artifacts/baseline').resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    inputs=['hands.parquet','seats.parquet','development_labels.csv','development_evidence.csv','evaluation_pairs.csv','sample_submission.csv']
    fingerprint={f:digest(root/f) for f in inputs}
    feature_source=inspect.getsource(build_features)+inspect.getsource(evidence_predictions)
    cache_signature={'inputs':fingerprint,'feature_code_sha256':hashlib.sha256(feature_source.encode()).hexdigest()}
    cache=out/'cache_signature.json'
    if args.rebuild or not cache.exists() or json.loads(cache.read_text())!=cache_signature:
        build_features(root,out);cache.write_text(json.dumps(cache_signature,indent=2)+'\n')
    result=train(root,out)
    manifest={**cache_signature,'pipeline_sha256':digest(Path(__file__)), 'official_metric_sha256':digest(Path(__file__).with_name('official_metric.py')),
              'seed':SEED,'elapsed_seconds':time.time()-start,'python':platform.python_version(),
              'versions':{x.__name__:x.__version__ for x in [np,pd,pyarrow,sklearn,lgb]},
              'submission_sha256':digest(out/'submission.csv')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
