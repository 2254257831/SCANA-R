"""Local-only paths and simulator adapters. No changes to upstream physics."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'libero_local_v1_20261002'/'vendor'/'LIBERO-8f1084e3132a39270c3a13ebe37270a43ece2a01'
os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'libero_config')
os.environ['MUJOCO_GL']='glfw'
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'numba_cache')
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,str(SOURCE))
import yaml
from functools import lru_cache
(ROOT/'libero_config').mkdir(exist_ok=True)
_paths={'benchmark_root':SOURCE/'libero/libero','bddl_files':SOURCE/'libero/libero/bddl_files',
        'init_states':SOURCE/'libero/libero/init_files','datasets':ROOT/'data','assets':SOURCE/'libero/libero/assets'}
(ROOT/'libero_config/config.yaml').write_text(yaml.safe_dump({k:str(v) for k,v in _paths.items()}),encoding='utf8')

@lru_cache(maxsize=1)
def task_suite():
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)

@lru_cache(maxsize=256)
def asset_bytes(path):
    return Path(path).read_bytes()

def install_unicode_vfs():
    # MuJoCo 2.3's Windows native file loader cannot resolve this workspace's
    # Unicode path. Pass the exact same mesh/texture bytes through its VFS.
    import mujoco
    import xml.etree.ElementTree as ET
    from robosuite.utils.binding_utils import MjSim
    @classmethod
    def from_xml_string(cls,xml):
        tree=ET.fromstring(xml); assets={}; names={}
        for elem in tree.findall('./asset/*'):
            p=elem.get('file')
            if not p:continue
            if p not in names:
                name=f'vfs_{len(names):04d}'+Path(p).suffix
                assets[name]=asset_bytes(p);names[p]=name
            elem.set('file',names[p])
        return cls(mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'),assets))
    MjSim.from_xml_string=from_xml_string

def make_env(task_id,images=True,size=96):
    from libero.libero.envs import OffScreenRenderEnv
    install_unicode_vfs()
    task=task_suite().get_task(task_id)
    return OffScreenRenderEnv(bddl_file_name=str(_paths['bddl_files']/task.problem_folder/task.bddl_file),
                             camera_heights=size,camera_widths=size,use_camera_obs=images)

def write_json(name,data):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(p)

def data_file(task_id):
    t=task_suite().get_task(task_id)
    return ROOT/'data'/'libero_spatial'/(t.name+'_demo.hdf5')

def localize_xml(xml):
    import xml.etree.ElementTree as ET
    import robosuite
    tree=ET.fromstring(xml)
    for e in tree.findall('./asset/*'):
        p=e.get('file')
        if not p:continue
        parts=p.replace('\\','/').split('/')
        if 'robosuite' in parts:
            i=max(i for i,v in enumerate(parts) if v=='robosuite')
            new=Path(robosuite.__file__).parent.joinpath(*parts[i+1:])
        elif 'assets' in parts:
            i=max(i for i,v in enumerate(parts) if v=='assets')
            new=_paths['assets'].joinpath(*parts[i+1:])
        else: new=Path(p)
        if not new.exists():raise FileNotFoundError(new)
        e.set('file',str(new))
    return ET.tostring(tree,encoding='unicode')

def reset_demo(env,group):
    # Reuse the exact compiled source only within one episode; the corrected
    # reset was validated against a fresh XML reset, including all saved states.
    key=(group.file.filename,group.name)
    if getattr(env,'_scana_demo_key',None)!=key:
        xml=localize_xml(group.attrs['model_file'])
        env.reset_from_xml_string(xml);env._scana_demo_key=key
    else:
        env.env.deterministic_reset=True
        try:env.env.reset()
        finally:env.env.deterministic_reset=False
    # The Panda gripper integrates commands in a Python-side accumulator;
    # reset_from_xml_string does not clear it. Match a newly constructed robot.
    import numpy as np
    for robot in env.robots:
        if hasattr(robot.gripper,'current_action'):
            robot.gripper.current_action=np.zeros_like(robot.gripper.current_action)
    env.sim.reset()
    return env.set_init_state(group['states'][0])

def proprio(obs):
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([obs['robot0_joint_pos'],obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype('float32')
