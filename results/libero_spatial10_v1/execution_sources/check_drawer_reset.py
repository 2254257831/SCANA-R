from runtime import *
import numpy as np, random

def main():
    task=4;env=make_env(task,images=False,size=128);initial=task_suite().get_task_init_states(task)[0]
    traces=[];models=[];extras=[]
    for repeat in range(3):
        np.random.seed(100000+task*100);random.seed(100000+task*100);env.seed(100000+task*100)
        env.reset();obs=env.set_init_state(initial)
        models.append(env.env.model.get_xml())
        extras.append({'body_pos':env.sim.model.body_pos.copy().tolist(),'body_quat':env.sim.model.body_quat.copy().tolist(),
                       'gripper':env.robots[0].gripper.current_action.tolist(),'ctrl':env.sim.data.ctrl.copy().tolist()})
        traj=[]
        for _ in range(10):
            traj.append(env.sim.get_state().flatten());obs,_,_,_=env.step(np.zeros(7))
        traces.append(np.asarray(traj))
    env.close()
    differences=[{'repeat':i,'exact':bool(np.array_equal(traces[0],traces[i])),
                  'step_max_abs':np.max(np.abs(traces[0]-traces[i]),axis=1).tolist(),
                  'max_indices':np.unravel_index(np.argmax(np.abs(traces[0]-traces[i])),traces[0].shape),
                  'xml_identical':models[0]==models[i],
                  'bodypos_max':float(np.max(np.abs(np.asarray(extras[0]['body_pos'])-extras[i]['body_pos'])))} for i in [1,2]]
    # Convert NumPy indices for JSON; preserve all raw diagnostic arrays.
    for x in differences:x['max_indices']=list(map(int,x['max_indices']))
    np.savez_compressed(ROOT/'preflight/drawer_reset_traces.npz',traces=traces)
    for i,xml in enumerate(models):(ROOT/f'preflight/drawer_reset_model_{i}.xml').write_text(xml,encoding='utf-8')
    write_json('preflight/drawer_reset_diagnostic.json',{'differences':differences,'extras':extras})
    print(differences,flush=True)

if __name__=='__main__':main()
