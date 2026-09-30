"""ACT physics unchanged; collect current privileged state without RGB."""
from common import *
sys.path.insert(0,str(ACT))
import constants
import tempfile, shutil
import os
asset_dir=Path(os.environ['SCANA_ASSET_CACHE'])/str(os.getpid())/'assets'
shutil.copytree(ACT/'assets',asset_dir)
constants.XML_DIR=str(asset_dir)
import types
try:import IPython
except ImportError:
    ip=types.ModuleType('IPython');ip.embed=lambda:None;sys.modules['IPython']=ip
import sim_env, ee_sim_env
from scripted_policy import PickAndTransferPolicy,InsertionPolicy
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN

def observation(self,p):
    obs=dict(qpos=self.get_qpos(p),qvel=self.get_qvel(p),env_state=self.get_env_state(p),images={})
    if p.model.nmocap:
        obs['mocap_pose_left']=np.r_[p.data.mocap_pos[0],p.data.mocap_quat[0]].copy()
        obs['mocap_pose_right']=np.r_[p.data.mocap_pos[1],p.data.mocap_quat[1]].copy()
        obs['gripper_ctrl']=p.data.ctrl.copy()
    return obs
sim_env.BimanualViperXTask.get_observation=observation
ee_sim_env.BimanualViperXEETask.get_observation=observation

def initial_pose(task,seed):
    from utils import sample_box_pose,sample_insertion_pose
    np.random.seed(seed)
    return sample_box_pose() if task=='transfer_cube' else np.concatenate(sample_insertion_pose())

def causal_observation(obs,step):
    # Current simulator object poses are privileged state, not future observations.
    pos=np.asarray(obs['env_state']).reshape(-1,7)[:,:3].reshape(-1)
    return np.r_[obs['qpos'],pos,step/400].astype(np.float32)

def collect(task,seed):
    np.random.seed(seed);env=ee_sim_env.make_ee_sim_env('sim_'+task+'_scripted');ts=env.reset()
    pose=ts.observation['env_state'].copy();policy=(PickAndTransferPolicy if task=='transfer_cube' else InsertionPolicy)(False)
    actions=[]
    for step in range(400):
        obs=ts.observation;q=obs['qpos'].copy();ctrl=obs['gripper_ctrl']
        q[6]=PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0]);q[13]=PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2]);actions.append(q)
        ts=env.step(policy(ts))
    env.close()
    sim_env.BOX_POSE[0]=pose;env=sim_env.make_sim_env('sim_'+task);ts=env.reset();xs=[];states=[];rew=[]
    for i,a in enumerate(actions):
        xs.append(causal_observation(ts.observation,i));states.append(ts.observation['qpos'].copy())
        ts=env.step(a);rew.append(ts.reward)
    env.close()
    return dict(action=np.asarray(actions,dtype=np.float32),state=np.asarray(states,dtype=np.float32),input=np.asarray(xs),pose=pose,rewards=np.asarray(rew),seed=seed,success=int(max(rew)==4))
