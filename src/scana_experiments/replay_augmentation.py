"""Bounded simulation action perturbations for a research SCANA revision.

This module does NOT produce observation-compatible labels by itself. Its
output must be executed in the simulator and paired with the observations
actually recorded there. A successful rollout is a task-specific acceptance
check, not a collision-safety certificate. DART is a relevant prior method.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplayNoiseConfig:
    amplitude: float = 0.01
    chunk: int = 16
    phase_radius: int = 16
    gripper_channels: tuple[int, ...] = (6, 13)
    temporal_window: int = 5
    max_amplitude_multiplier: float = 3.0
    event_threshold: float = 0.005
    event_radius: int = 8
    event_scale: float = 0.15
    boundary_steps: int = 8


def bounded_replay_actions(action, errors, starts, rng, config=ReplayNoiseConfig(), mode='calibrated'):
    """Draw phase-local cross-fitted error shapes or a correlated Gaussian.

    ``errors`` must contain out-of-source-episode policy prediction errors from
    training episodes only. Phase and gripper scheduling are augmentation-time
    information; they are never inputs to the deployed policy. Zero amplitude
    is exactly the identity, including source actions outside empirical bounds.
    """
    a=np.asarray(action)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('finite T-by-D actions required')
    if config.amplitude<0:raise ValueError('amplitude must be nonnegative')
    if mode not in ('calibrated','gaussian'):raise ValueError('unknown noise mode')
    if config.temporal_window%2!=1 or config.temporal_window<1:raise ValueError('odd positive temporal window required')
    if config.amplitude==0:return a.copy()
    t,d=a.shape;grip=np.asarray(config.gripper_channels);joints=np.setdiff1d(np.arange(d),grip)
    e_all=np.asarray(errors);phase=np.asarray(starts)
    if mode=='calibrated' and (e_all.ndim!=3 or e_all.shape[1:]!=(config.chunk,d) or len(phase)!=len(e_all) or not np.isfinite(e_all).all()):
        raise ValueError('invalid calibration bank')
    noise=np.zeros_like(a)
    for start in range(0,t,config.chunk):
        if mode=='calibrated':
            pool=np.flatnonzero(np.abs(phase-start)<=config.phase_radius)
            if not len(pool):continue  # unsupported phase: identity, no extrapolation
            e=e_all[rng.choice(pool)].copy()-e_all[pool].mean(0)
            e*=rng.choice([-1,1])
        else:
            e=rng.standard_normal((config.chunk,d))
            for k in range(1,config.chunk):e[k]=.9*e[k-1]+np.sqrt(1-.9**2)*e[k]
        e[:,grip]=0
        rms=np.sqrt(np.mean(e[:,joints]**2))
        n=min(config.chunk,t-start)
        noise[start:start+n]=e[:n]/max(rms,1e-8)*config.amplitude
    half=config.temporal_window//2
    p=np.pad(noise,((half,half),(0,0)),mode='edge')
    noise=sum(p[k:k+t] for k in range(config.temporal_window))/config.temporal_window
    cap=config.max_amplitude_multiplier*config.amplitude
    noise=np.clip(noise,-cap,cap);noise[:,grip]=0
    changing=np.max(np.abs(np.diff(a[:,grip],axis=0,prepend=a[:1,grip])),axis=1)>config.event_threshold
    # full convolution with explicit centered slice supports short trajectories.
    filt=np.ones(2*config.event_radius+1)
    protect=np.convolve(changing.astype(float),filt,mode='full')[config.event_radius:config.event_radius+t]>0
    noise[protect]*=config.event_scale
    n=min(config.boundary_steps,t);noise[:n]=0;noise[t-n:]=0
    return a+noise
