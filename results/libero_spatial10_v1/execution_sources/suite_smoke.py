"""All-task native reset/render/control checks, with no policy success claim."""
from runtime import *
import numpy as np, time, random


def main():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    task_map = {}; results = []
    for tid in range(10):
        task = task_suite().get_task(tid)
        states = task_suite().get_task_init_states(tid)
        assert len(states) >= 50
        trajectories = []; fixture_states = []
        for repeat in range(2):
            np.random.seed(100000+tid*100); random.seed(100000+tid*100)
            env = make_env(tid, size=128)
            env.seed(100000+tid*100)
            env.reset(); obs = env.set_init_state(states[0])
            fixture_states.append(np.concatenate([env.sim.model.body_pos.ravel(), env.sim.model.body_quat.ravel()]))
            native_dim = int(env.env.action_dim)
            traj = []
            for _ in range(10):
                traj.append(env.sim.get_state().flatten())
                obs, reward, done, info = env.step(np.zeros(7))
            assert proprio(obs).shape == (15,) and np.isfinite(proprio(obs)).all()
            assert obs['agentview_image'].shape == obs['robot0_eye_in_hand_image'].shape == (128,128,3)
            trajectories.append(np.asarray(traj))
            env.close()
        assert np.array_equal(*trajectories), f'Nonrepeatable native test reset on task {tid}'
        assert np.array_equal(*fixture_states), f'Nonrepeatable fixed-body pose on task {tid}'
        task_map[str(tid)] = {'name':task.name, 'language':task.language}
        row = {'task':tid, 'name':task.name, 'initial_states_available':len(states),
               'native_action_dimension':native_dim, 'two_camera_shape':[128,128,3],
               'fresh_environment_per_episode':True, 'repeated_body_pose_max_error':0.0,
               'repeated_reset_max_state_error':float(np.max(np.abs(trajectories[0]-trajectories[1])))}
        assert row['native_action_dimension'] == 7
        results.append(row); print(row,flush=True)
        write_json('preflight/suite_smoke_progress.json', results)
    write_json('task_map.json', task_map)
    write_json('preflight/suite_smoke.json', {'tasks':results, 'cuda_device':torch.cuda.get_device_name(0),
               'scope':'Native environment/reset/render/control interface checks only; no policy evaluation',
               'completed_at':time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__ == '__main__':
    main()
