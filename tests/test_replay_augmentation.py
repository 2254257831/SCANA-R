import unittest
import numpy as np
from scana_experiments.replay_augmentation import bounded_replay_actions, ReplayNoiseConfig


class ReplayAugmentationContract(unittest.TestCase):
    def setUp(self):
        self.a=np.random.default_rng(1).normal(size=(400,14)).astype('float32')
        self.a[:,[6,13]]=0
        self.errors=np.random.default_rng(2).normal(size=(100,16,14)).astype('float32')
        self.starts=np.tile(np.arange(0,400,16),4)

    def draw(self,cfg=ReplayNoiseConfig(),mode='calibrated'):
        return bounded_replay_actions(self.a,self.errors,self.starts,np.random.default_rng(3),cfg,mode)

    def test_zero_is_identity_even_outside_empirical_bounds(self):
        self.a[100,0]=1000
        np.testing.assert_array_equal(self.draw(ReplayNoiseConfig(amplitude=0)),self.a)

    def test_gripper_events_and_source_are_preserved(self):
        self.a[200:,[6,13]]=1;old=self.a.copy();out=self.draw()
        np.testing.assert_array_equal(out[:,[6,13]],old[:,[6,13]])
        np.testing.assert_array_equal(self.a,old)
        np.testing.assert_array_equal(out[:8],old[:8]);np.testing.assert_array_equal(out[-8:],old[-8:])

    def test_absolute_trust_region_for_both_noise_sources(self):
        for mode in ['calibrated','gaussian']:
            out=self.draw(mode=mode)
            self.assertLessEqual(np.max(np.abs(out-self.a)),.03+1e-6)
            self.assertGreater(np.std(out-self.a),0)

    def test_no_supported_phase_means_no_extrapolated_perturbation(self):
        out=bounded_replay_actions(self.a,self.errors,self.starts+10000,np.random.default_rng(4))
        np.testing.assert_array_equal(out,self.a)

    def test_seed_controls_reproducibility_not_source_content(self):
        np.testing.assert_array_equal(self.draw(),self.draw())
        out=bounded_replay_actions(self.a,self.errors,self.starts,np.random.default_rng(5))
        self.assertFalse(np.array_equal(out,self.draw()))

    def test_short_sequences(self):
        out=bounded_replay_actions(self.a[:10],self.errors,self.starts,np.random.default_rng(6))
        np.testing.assert_array_equal(out,self.a[:10])

if __name__=='__main__':unittest.main()
