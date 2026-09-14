import unittest
import numpy as np
import pandas as pd
from official_metric import score, _average_precision, EVIDENCE_COLUMNS, ParticipantVisibleError
from run import evidence_predictions, components


class BaselineChecks(unittest.TestCase):
    def frames(self):
        truth=pd.DataFrame({'pair_id':['P1','P2'],'risk_score':[1,0],
                            'predicted_behavior':['soft_play','none']})
        for i,c in enumerate(EVIDENCE_COLUMNS):
            truth[c]=['H1' if i==0 else 'NO_EVIDENCE','NO_EVIDENCE']
        pred=truth.copy();pred.risk_score=[.8,.2]
        return truth,pred

    def test_exact_ties_and_components(self):
        self.assertEqual(_average_precision(np.array([1,0]),np.zeros(2)),1.)
        t,p=self.frames();m=components(t,p)
        self.assertAlmostEqual(m['evidence_map5'],1.)
        self.assertAlmostEqual(m['composite'],score(t,p,'pair_id'))

    def test_duplicate_evidence_rejected(self):
        t,p=self.frames();p.loc[0,'evidence_hand_2']='H1'
        with self.assertRaises(ParticipantVisibleError):score(t,p,'pair_id')

    def test_evidence_ranking_and_padding(self):
        c=pd.DataFrame({'key':[1,1,1,2],'hand_id':['H1','H2','H2','H3'],
                        'heuristic':[1.,3.,2.,0.]})
        r=evidence_predictions(c).set_index('key')
        self.assertEqual(r.loc[1,'evidence_hand_1'],'H2')
        self.assertEqual(r.loc[1,'evidence_hand_2'],'H1')
        self.assertEqual(r.loc[1,'evidence_hand_3'],'NO_EVIDENCE')


if __name__=='__main__':unittest.main()
