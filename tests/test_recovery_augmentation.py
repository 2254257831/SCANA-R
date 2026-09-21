import unittest
import numpy as np
from scana_experiments.recovery_augmentation import *


class RecoveryContract(unittest.TestCase):
    def test_success_before_pulse_does_not_validate_failed_recovery(self):
        self.assertFalse(terminally_successful([0,1,4,2,0]))
        self.assertTrue(terminally_successful([0,1,2,4,4]))
        self.assertFalse(terminally_successful([]))
        self.assertFalse(terminally_successful([float('nan'),4]))

    def test_zero_backtracking_preserves_any_source_action(self):
        a=np.random.default_rng(1).normal(size=(400,14)).astype('float32');p=np.ones((8,14))
        np.testing.assert_array_equal(apply_pulse(a,p,80,0),a)

    def test_pulse_cannot_change_future_reference_or_grippers(self):
        a=np.zeros((400,14));p=np.ones((8,14));b=apply_pulse(a,p,80)
        np.testing.assert_array_equal(b[:,[6,13]],a[:,[6,13]])
        np.testing.assert_array_equal(b[:72],a[:72]);np.testing.assert_array_equal(b[80:],a[80:])
        self.assertGreater(b[72:80].sum(),0)

    def test_recovery_windows_pair_actual_current_observations(self):
        a=np.arange(400*14,dtype=float).reshape(400,14);obs=np.arange(400*18).reshape(400,18);b=apply_pulse(a,np.ones((8,14)),368)
        x,y,t=recovery_windows(obs,b,a,368)
        np.testing.assert_array_equal(t,[368,376,384]);np.testing.assert_array_equal(x,obs[t]);np.testing.assert_array_equal(y[0],a[368:384].reshape(-1))
        b[380,0]+=1
        with self.assertRaises(ValueError):recovery_windows(obs,b,a,368)

    def test_unsupported_phase_falls_back_to_identity(self):
        errors=np.ones((5,16,14));phase=np.arange(5)*8
        p=draw_pulse(errors,phase,300,np.random.default_rng(1));np.testing.assert_array_equal(p,0)

    def test_both_sources_obey_trust_region(self):
        errors=np.random.default_rng(1).normal(size=(49,16,14));phase=np.arange(49)*8
        for mode in ['calibrated','gaussian']:
            p=draw_pulse(errors,phase,160,np.random.default_rng(2),mode=mode)
            self.assertLessEqual(np.abs(p).max(),.09+1e-12);np.testing.assert_array_equal(p[:,[6,13]],0)

if __name__=='__main__':unittest.main()
