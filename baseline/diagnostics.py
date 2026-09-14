"""Frozen diagnostic comparisons. Never select a model using these outer-fold results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from run import model, SEED
from official_metric import _average_precision


def diagnose(out):
    out=Path(out)
    features=pd.read_parquet(out/'features.parquet')
    labels=pd.read_parquet(out/'labels.parquet')
    folds=pd.read_csv(out/'folds.csv')
    dev=labels.merge(features[features.phase.eq('development')],on='key',validate='one_to_one')
    dev=dev.merge(folds[['pair_id','fold']],on='pair_id',validate='one_to_one').sort_values('pair_id').reset_index(drop=True)
    columns=json.loads((out/'metrics.json').read_text())['feature_columns']
    oof=pd.read_csv(out/'oof_predictions.csv').set_index('pair_id').loc[dev.pair_id]
    reports=[]
    for fold in range(5):
        tr=dev.fold.ne(fold);va=~tr
        assert set(dev.loc[tr,'table_id']).isdisjoint(dev.loc[va,'table_id'])
        training_players=set(dev.loc[tr,'player_1'])|set(dev.loc[tr,'player_2'])
        validation_players=set(dev.loc[va,'player_1'])|set(dev.loc[va,'player_2'])
        assert training_players.isdisjoint(validation_players)
        # Preserve original training row order to reproduce the frozen baseline exactly.
        train_order=dev.loc[tr].sort_values(['table_id','key'])
        X=train_order[columns];y=train_order.label
        m=model();m.fit(X,y)
        for rounds in [50,125,250]:
            ptr=m.predict_proba(X,num_iteration=rounds)[:,1]
            pva=m.predict_proba(dev.loc[va,columns],num_iteration=rounds)[:,1]
            if rounds==250:
                assert np.allclose(pva,oof.risk_score.to_numpy()[va],atol=1e-12,rtol=1e-10), 'Diagnostic refit differs from frozen OOF predictions'
            # Exact host tie order for training rows too.
            perm=np.argsort(train_order.pair_id.to_numpy(),kind='stable')
            train_ap=_average_precision(y.to_numpy()[perm],ptr[perm])
            val_ap=_average_precision(dev.loc[va,'label'].to_numpy(),pva)
            reports.append({'fold':fold,'model':'LightGBM','trees':rounds,
                'train_ap':train_ap,'validation_ap':val_ap,'ap_gap':train_ap-val_ap,
                'train_logloss':log_loss(y,ptr,labels=[0,1]),
                'validation_logloss':log_loss(dev.loc[va,'label'],pva,labels=[0,1]),
                'validation_prevalence':float(dev.loc[va,'label'].mean())})
        simple=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=SEED))
        simple.fit(X,y)
        ptr=simple.predict_proba(X)[:,1];pva=simple.predict_proba(dev.loc[va,columns])[:,1]
        train_ap=_average_precision(y.to_numpy()[perm],ptr[perm])
        val_ap=_average_precision(dev.loc[va,'label'].to_numpy(),pva)
        reports.append({'fold':fold,'model':'LogisticRegression','trees':0,
            'train_ap':train_ap,'validation_ap':val_ap,'ap_gap':train_ap-val_ap,
            'train_logloss':log_loss(y,ptr,labels=[0,1]),
            'validation_logloss':log_loss(dev.loc[va,'label'],pva,labels=[0,1]),
            'validation_prevalence':float(dev.loc[va,'label'].mean())})
    frame=pd.DataFrame(reports);frame.to_csv(out/'fit_diagnostics.csv',index=False)
    summary=frame.groupby(['model','trees'])[['train_ap','validation_ap','ap_gap','train_logloss','validation_logloss']].mean().reset_index()
    summary.to_csv(out/'fit_summary.csv',index=False)
    # Descriptive table bootstrap of fixed OOF scores; no model refitting.
    table_rows=[g.index.to_numpy() for _,g in dev.groupby('table_id',sort=True)]
    rng=np.random.default_rng(SEED);boot=[]
    for _ in range(300):
        ix=np.sort(np.concatenate([table_rows[i] for i in rng.integers(len(table_rows),size=len(table_rows))]))
        boot.append(_average_precision(dev.label.to_numpy()[ix],oof.risk_score.to_numpy()[ix]))
    final=summary[(summary.model=='LightGBM') & summary.trees.eq(250)].iloc[0]
    earlier=summary[(summary.model=='LightGBM') & summary.trees.eq(125)].iloc[0]
    alerts=[]
    if final.ap_gap>.10:alerts.append('Train/validation AP gap exceeds 0.10: investigate overfitting or distribution differences.')
    if final.validation_ap < earlier.validation_ap and final.train_ap > earlier.train_ap:
        alerts.append('More trees improve training AP but reduce held-out AP: a possible overfitting signal.')
    if final.validation_ap <= summary[summary.model=='LogisticRegression'].validation_ap.iloc[0]:
        alerts.append('LightGBM does not beat the simpler comparator on mean fold AP; investigate features/capacity.')
    alerts.append('These checks cannot certify absence of overfitting or underfitting; outer results are diagnostics, not tuning targets.')
    report={'table_and_player_overlap':0,'diagnostics_only_no_model_selection':True,
        'fixed_oof_pair_ap_table_bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap_note':'Descriptive 300-table-bootstrap replicates of fixed OOF predictions, not private-LB uncertainty or full training uncertainty.',
        'alerts':alerts,'summary':summary.to_dict('records')}
    (out/'validation_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary.to_string(index=False));print('\n'.join(alerts))
    return frame,report
