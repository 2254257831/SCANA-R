import unittest
import numpy as np
from scana_experiments.audited_metrics import direction_metrics,equal_group_weights,weighted_mmd

class AuditedEvaluationTests(unittest.TestCase):
    def test_stationary_windows_are_not_direction_errors(self):
        a=np.array([[[0.],[0.],[0.]],[[0.],[1.],[2.]]])
        report,values=direction_metrics(a,a)
        self.assertEqual(report['static_static'],1)
        self.assertAlmostEqual(report['moving_cosine'],1,12)
        self.assertTrue(np.isnan(values[0]))
        b=a[::-1].copy();report,_=direction_metrics(a,b)
        self.assertEqual(report['static_moving'],1)
        self.assertEqual(report['moving_static'],1)
        self.assertIsNone(report['moving_cosine'])

    def test_anchor_repetition_has_no_extra_mass(self):
        x=np.array([[0.],[2.]]);y=np.array([[1.],[4.]])
        v=weighted_mmd(x,y,.3)
        ids=[0,0,0,1]
        repeated=weighted_mmd(x[ids],y[ids],.3,equal_group_weights(ids))
        self.assertAlmostEqual(v,repeated,12)

    def test_identity_mmd_is_zero(self):
        x=np.random.default_rng(9).normal(size=(9,16,6))
        self.assertAlmostEqual(weighted_mmd(x,x,.4),0,12)

    def test_policy_input_is_invariant_to_future_and_targets(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legacy/tools'))
        from scana_public_policy_physics_closure_20260714 import features
        r=np.random.default_rng(8)
        d={'state':r.normal(size=(8,16,14)),'action':r.normal(size=(8,16,14)),'condition':r.normal(size=(8,8))}
        before=features(d)
        d['state'][:,1:]=np.nan;d['action'][:]=np.inf;d['condition'][:]=-99999
        np.testing.assert_array_equal(features(d),before)
        self.assertEqual(before.shape,(8,14))

if __name__=='__main__':unittest.main()
