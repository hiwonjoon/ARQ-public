import numpy as np
import gin
import ray

import gym
import arq.modules.robomimic_env
import d4rl

from arq.modules.env_utils import _env_info, list_of_tuple_to_traj
from arq.modules.utils import tqdm

def wrap(Obj, num_cpus=1, num_gpus=0, num_tf_threads=2):
    import ray
    import gin

    @ray.remote(num_cpus=num_cpus,num_gpus=num_gpus)
    class RayRemoteObj(Obj):
        def __init__(self, config_files=[], config_params=[]):
            pass

        def init(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return RayRemoteObj


@gin.configurable
class RayEnv(object):
    def __init__(self,env_id,seed=None):
        self.env = gym.make(env_id)
        
        np.random.seed(seed)
        try:
            self.env.wrapped_env.seed(seed)
        except:
            self.env.seed(seed)

        self.env_info = _env_info(self.env)

        self.context = None # current trajectory context

    def _unroll_till_end(self, render=False, clip=False):
        # executed in the remote process
        transition_tuples = []

        t, s, should_reset = 0, self.env.reset(), False

        while should_reset == False:
            a = yield s

            if clip:
                a = np.clip(a,self.env.action_space.low,self.env.action_space.high)

            t,(ś,r,should_reset,_) = t+1, self.env.step(a)
            f = self.env.render('rgb_array') if render else None

            done = False
            if should_reset and t != self.env_info['max_length']:
                done = True # only set true when it actually dies in a episodic task.

            transition_tuples.append((s,a,r,ś,done,f))
            s = ś

        return list_of_tuple_to_traj(transition_tuples)

    def step(self,action):
        # communicating method
        if action is None:
            self.context = self._unroll_till_end()
            state = self.context.send(action)
            return (False, state)
        else:
            try:
                state = self.context.send(action)
                return (False, state)
            except StopIteration as e:
                traj = e.value
                self.context = None
                return (True, traj)

@gin.configurable()
class BatchRemoteEnv(object):
    """
    Synchronous Batched Remote Env
    """
    def __init__(self, env_id, num_envs=20):
        RemoteEnv = wrap(RayEnv,num_cpus=0.01)

        self.test_env = gym.make(env_id)

        self.remote_envs = [RemoteEnv.remote() for _ in range(num_envs)]

        # initialize one by one.
        for env in self.remote_envs:
            _ = ray.get(env.init.remote(env_id))

        #_ = ray.get([env.init.remote(env_id) for env in self.remote_envs])

    @property
    def num_envs(self):
        return len(self.remote_envs)

    def get_normalized_score(self,R):
        try:
            return self.test_env.get_normalized_score(R)
        except:
            return R

    def unroll_till_end(self, pi, stochastic, num_trajs, debug=False):
        trajs = []
        _current_perf = 0.

        running_envs = self.remote_envs[:min(num_trajs,len(self.remote_envs))]
        remaining_runs = num_trajs - len(running_envs)

        actions = [None] * len(running_envs)

        pbar = tqdm(disable=not debug)
        pbar.set_description(f"{len(running_envs)} running, {remaining_runs} to go")
        while True:
            pbar.update()

            results = ray.get([env.step.remote(a) for (env,a) in zip(running_envs,actions)])

            states = []
            for env, (done, ret) in zip(running_envs[:],results):
                if done:
                    trajs += [ret]
                    _current_perf = np.mean([np.sum(traj.rewards) for traj in trajs])

                    if remaining_runs > 0:
                        states.append(ray.get(env.step.remote(None))[1])
                        remaining_runs -= 1
                    else:
                        running_envs.remove(env)

                    pbar.set_description(f"{len(trajs)} done (perf: {_current_perf:.2f}), {len(running_envs)} running, {remaining_runs} to go")
                else:
                    states.append(ret)
            
            assert len(states) == len(running_envs)

            if len(running_envs) == 0:
                assert remaining_runs == 0
                break

            actions, _ = pi(np.array(states),stochastic)
            try:
                actions = actions.numpy()
            except:
                pass
        pbar.close()

        return trajs