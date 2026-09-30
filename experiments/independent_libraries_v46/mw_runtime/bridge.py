"""Pinned Meta-World adapter: current state, official physics and success rules."""
from pathlib import Path
import os, shutil, tempfile
import numpy as np
import metaworld
from metaworld import policies
from metaworld import asset_path_utils

HORIZON=250
TASKS={
    'reach-v3':'SawyerReachV3Policy',
    'push-v3':'SawyerPushV3Policy',
    'pick-place-v3':'SawyerPickPlaceV3Policy',
    'door-open-v3':'SawyerDoorOpenV3Policy',
    'drawer-open-v3':'SawyerDrawerOpenV3Policy',
    'button-press-v3':'SawyerButtonPressV3Policy',
}

def prepare_assets():
    """MuJoCo's Windows file API requires an ASCII asset path; bytes are unchanged."""
    source=asset_path_utils.ENV_ASSET_DIR_V3
    if os.name=='nt' and not str(source).isascii():
        cache=Path(os.environ.get('SCANA_MW_ASSET_CACHE',str(Path(tempfile.gettempdir())/'scana_v46_metaworld_assets')))
        if not (cache/'COPY_COMPLETE').exists():
            cache.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,cache,dirs_exist_ok=True)
            (cache/'COPY_COMPLETE').write_text('Meta-World 3.1.1 assets copied unchanged.\n')
        asset_path_utils.ENV_ASSET_DIR_V3=cache

prepare_assets()

def make_env(task,seed,render=False):
    return metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-goal-observable'](
        seed=int(seed),render_mode='rgb_array' if render else None)

def current_input(obs,t):
    # Exclude frame-stacked history 18:36. Goal coordinates are currently observable.
    return np.r_[obs[:18],obs[-3:],t/HORIZON].astype(np.float32)

def execute(env,actions=None,expert=None):
    obs,_=env.reset();inputs=[];raw=[];executed=[];rewards=[];success=[]
    stopped=False;held_grip=0.
    for t in range(HORIZON):
        inputs.append(current_input(obs,t));raw.append(obs.copy())
        if expert is not None:
            # A common demonstration endpoint rule: hold xyz after first success.
            # Deployment policies never receive this success signal or stopping rule.
            action=np.array([0.,0.,0.,held_grip]) if stopped else expert.get_action(obs.copy())
        else: action=actions[t]
        action=np.clip(np.asarray(action,dtype=np.float32),-1,1)
        executed.append(action.copy())
        obs,reward,terminated,truncated,info=env.step(action)
        rewards.append(float(reward));success.append(int(info['success']))
        if expert is not None and info['success']:
            stopped=True;held_grip=float(action[3])
    return dict(input=np.asarray(inputs),raw_obs=np.asarray(raw),action=np.asarray(executed),
                reward=np.asarray(rewards),success=np.asarray(success,dtype=np.int8),
                initial_rand_vec=env._last_rand_vec.copy())

def reference(task,seed):
    env=make_env(task,seed)
    try:
        return execute(env,expert=getattr(policies,TASKS[task])())
    finally:env.close()
