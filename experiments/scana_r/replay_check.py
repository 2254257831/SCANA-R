from experiment import *
FINAL=WORK/'independent_test'
path=FINAL/'policies/transfer_cube/7/Calibrated recovery.pt'
m,ck=load_policy(path);actual=rollout('transfer_cube',92000,m,ck)
expected=dict(np.load(FINAL/'rollouts/transfer_cube/7/Calibrated recovery/92000.npz'))
err=float(np.max(np.abs(actual['action']-expected['action'])))
serr=float(np.max(np.abs(actual['state']-expected['state'])))
assert err<1e-5 and serr<1e-5 and actual['success']==int(expected['success'])
print(json.dumps(dict(relocated_directory_smoke_passed=True,action_max_error=err,state_max_error=serr,success=actual['success'],note='Same installed Python environment and host; not a fresh-environment reproduction, not an additional independent test.'),indent=2))
