"""Simulation-verified recovery augmentation primitives.

Research SCANA-R revision. Calibrate from same-task, out-of-episode policy
errors; inject a bounded short pulse; execute and record the reference tail.
Successful terminal replay is required before accepting recovery supervision.
This changes the original fixed-context action-label augmentation contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    pulse_steps: int = 8
    chunk_steps: int = 16
    phase_radius: int = 8
    amplitude: float = .03
    cap_multiplier: float = 3.
    gripper_channels: tuple[int,...] = (6,13)
    backtracking: tuple[float,...] = (1.,.5,.25,0.)


def draw_pulse(errors, phase, endpoint, rng, cfg=RecoveryConfig(), mode='calibrated'):
    errors=np.asarray(errors);phase=np.asarray(phase)
    if cfg.amplitude<0:raise ValueError('negative amplitude')
    if errors.ndim!=3 or len(errors)!=len(phase) or errors.shape[1]<cfg.pulse_steps:raise ValueError('invalid calibration bank')
    if not np.isfinite(errors).all():raise ValueError('nonfinite calibration errors')
    d=errors.shape[2];joints=np.setdiff1d(np.arange(d),cfg.gripper_channels)
    if mode=='calibrated':
        pool=np.flatnonzero(np.abs(phase-(endpoint-cfg.pulse_steps))<=cfg.phase_radius)
        if not len(pool):return np.zeros((cfg.pulse_steps,d),dtype=errors.dtype)
        e=errors[rng.choice(pool),:cfg.pulse_steps].copy()
    elif mode=='gaussian':
        e=rng.normal(size=(cfg.pulse_steps,d))
        for j in range(1,cfg.pulse_steps):e[j]=.9*e[j-1]+np.sqrt(1-.9**2)*e[j]
    else:raise ValueError('unknown noise source')
    e[:,cfg.gripper_channels]=0
    e=e/max(np.sqrt(np.mean(e[:,joints]**2)),1e-8)*cfg.amplitude
    return np.clip(e,-cfg.cap_multiplier*cfg.amplitude,cfg.cap_multiplier*cfg.amplitude)


def apply_pulse(reference,pulse,endpoint,scale=1.,cfg=RecoveryConfig()):
    a=np.asarray(reference).copy();p=np.asarray(pulse).copy()
    if p.shape!=(cfg.pulse_steps,a.shape[1]) or not cfg.pulse_steps<=endpoint<=len(a):raise ValueError('pulse outside trajectory')
    if not 0<=scale<=1 or not np.isfinite(p).all():raise ValueError('invalid pulse scale or values')
    p[:,cfg.gripper_channels]=0
    a[endpoint-cfg.pulse_steps:endpoint]+=scale*p
    return a


def terminally_successful(rewards,target=4.):
    r=np.asarray(rewards)
    return bool(r.ndim==1 and len(r)>0 and np.isfinite(r).all() and r[-1]==target)


def recovery_windows(observations,executed,reference,endpoint,cfg=RecoveryConfig()):
    obs=np.asarray(observations);a=np.asarray(executed);base=np.asarray(reference)
    if len(obs)!=len(a) or a.shape!=base.shape:raise ValueError('observation/action length mismatch')
    times=np.arange(endpoint,min(endpoint+32,len(a)-cfg.chunk_steps+1),8)
    if not len(times):raise ValueError('no complete recovery chunks')
    for t in times:
        if not np.array_equal(a[t:t+cfg.chunk_steps],base[t:t+cfg.chunk_steps]):raise ValueError('reference labels were not executed in the recovery window')
    return obs[times].copy(),np.asarray([a[t:t+cfg.chunk_steps].reshape(-1) for t in times]),times
