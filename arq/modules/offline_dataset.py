"""
Offline Dataset
"""

import gin
import numpy as np
import tensorflow as tf
from functools import partial

from arq.modules.utils import tqdm
from arq.modules.env_utils import Trajectory

@gin.configurable
class Dataset(object):
    def __init__(
        self,
        seed,
        trajs,
        #gin-configurables
        gamma,
        train_ratio,
        n_steps,
        sarsa,
        valid_next_only,
    ):
        self.seed = seed
        self.trajs = trajs
        self.gamma = gamma

        self.n_steps = n_steps
        self.sarsa = sarsa
        self.valid_next_only = valid_next_only

        self.ob_dim = self.trajs[0].states[0].shape
        self.ac_dim = self.trajs[0].actions[0].shape

        self.output_signature = (
            (tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)),
            (
                tf.TensorSpec(shape=self.ob_dim, dtype=tf.float32), tf.TensorSpec(shape=self.ac_dim, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=self.ob_dim, dtype=tf.float32), tf.TensorSpec(shape=self.ac_dim, dtype=tf.float32)
            ) if self.sarsa else\
            (
                tf.TensorSpec(shape=self.ob_dim, dtype=tf.float32), tf.TensorSpec(shape=self.ac_dim, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=self.ob_dim, dtype=tf.float32)
            )
        )

        rng = np.random.default_rng(seed=seed) if seed is not None else np.random

        self.all_trajs = [(idx,self.trajs[idx]) for idx in rng.permutation(len(self.trajs))]
        self.train_trajs = self.all_trajs[:int(len(self.trajs)*train_ratio)]
        self.valid_trajs = self.all_trajs[int(len(self.trajs)*train_ratio):]

        self.test_trajs = [(idx,self.trajs[idx]) for idx in range(len(self.trajs))[::10]]

        self.max_len = np.max([len(traj.states) for traj in self.trajs])

        ######### Debug
        print('-----------')
        print(f'Total # Train Transitions: {sum([len(traj.actions) for _,traj in self.train_trajs]):,}')
        print(f'Total # Valid Transitions: {sum([len(traj.actions) for _,traj in self.valid_trajs]):,}')
        print('-----------')

    @staticmethod
    def _parse(gamma, trajs, n_steps, sarsa, valid_next_only, t_0_only = False):
        gamma = [gamma**i for i in range(n_steps+1)]
        for idx,traj in trajs:
            states, actions, rewards, dones, *_ = traj
            timesteps = np.arange(len(states))

            #assert np.sum(dones) <= 1

            if sarsa and not dones[-1]:
                states = states[:-1]
                timesteps = timesteps[:-1]

            T = len(states) - 1

            # it will yield partial n-steps for the transitions at the end of the sequence.
            for t in range(T):
                s = states[t]
                a = actions[t]
                rs = rewards[t:t+n_steps]
                ds = dones[t:t+n_steps]
                ns = states[t+1:t+1+n_steps]
                na = actions[t+1:t+1+n_steps]
                nt = timesteps[t+1:t+1+n_steps]

                partial_R = np.sum(gamma[:len(rs)] * rs)
                discount = gamma[len(rs)] * (1-ds[-1])

                if valid_next_only and discount == 0.:
                    break

                if not sarsa:
                    yield (idx,t,nt[-1]), (s,a,partial_R,discount,ns[-1])
                else:
                    if discount == 0:
                        ns, na = np.zeros_like(s), np.zeros_like(a)
                    elif discount > 0. and len(ns) > 0:
                        assert len(ns) == len(na)
                        ns, na = ns[-1], na[-1]
                    else:
                        assert False

                    yield (idx,t,nt[-1]), (s,a,partial_R,discount,ns,na)

                if t_0_only:
                    break

    @gin.configurable
    def bc_epoch(self,type,include_idx=False):
        trajs = getattr(self,f'{type}_trajs')

        states = np.concatenate([traj.states[:-1] for _,traj in trajs],axis=0).astype(np.float32)
        actions = np.concatenate([traj.actions for _,traj in trajs],axis=0).astype(np.float32)

        if include_idx:
            traj_idx = np.concatenate([[traj_idx]*len(traj.states[:-1]) for traj_idx,traj in trajs],axis=0).astype(np.int32)
            t_idx = np.concatenate([np.arange(0,len(traj.states[:-1])) for _,traj in trajs],axis=0).astype(np.int32)

            assert len(states) == len(actions) == len(traj_idx) == len(t_idx)
            D = tf.data.Dataset.from_tensor_slices((states,actions,traj_idx,t_idx))
        else:
            assert len(states) == len(actions)
            D = tf.data.Dataset.from_tensor_slices((states,actions))

        return D

    @gin.configurable
    def epoch(self,type,n_steps=None,sarsa=None,valid_next_only=None,t_0_only=False,include_idx=False,allow_fast_parse=True):
        trajs = getattr(self,f'{type}_trajs')
        n_steps = self.n_steps if n_steps is None else n_steps
        sarsa = self.sarsa if sarsa is None else sarsa
        valid_next_only = self.valid_next_only if valid_next_only is None else valid_next_only

        if allow_fast_parse and n_steps == 1 and valid_next_only == False and t_0_only == False:
            if not sarsa:
                traj_idx = np.concatenate([[traj_idx]*len(traj.states[:-1]) for traj_idx,traj in trajs],axis=0).astype(np.int32)
                t_idx = np.concatenate([np.arange(0,len(traj.states[:-1])) for _,traj in trajs],axis=0).astype(np.int32)
                states = np.concatenate([traj.states[:-1] for _,traj in trajs],axis=0).astype(np.float32)
                actions = np.concatenate([traj.actions for _,traj in trajs],axis=0).astype(np.float32)
                rewards = np.concatenate([traj.rewards for _,traj in trajs],axis=0).astype(np.float32)
                next_states = np.concatenate([traj.states[1:] for _,traj in trajs],axis=0).astype(np.float32)
                dones = np.concatenate([traj.dones for _,traj in trajs],axis=0).astype(np.float32)
                discount = self.gamma * (1 - dones)

                D = tf.data.Dataset.from_tensor_slices(((traj_idx,t_idx,t_idx+1),(states,actions,rewards,discount,next_states)))
            else:
                idxes, items = [], []

                for i,traj in trajs:
                    #assert np.sum(traj.dones) <= 1

                    if traj.dones[-1] == False:
                        T = len(traj.states) - 2
                        items.append((
                            traj.states[:-2],
                            traj.actions[:-1],
                            traj.rewards[:-1],
                            traj.dones[:-1],
                            traj.states[1:-1],
                            traj.actions[1:]
                        ))
                    else:
                        T = len(traj.states) - 1
                        items.append((
                            traj.states[:-1],
                            traj.actions,
                            traj.rewards,
                            traj.dones,
                            traj.states[1:],
                            np.concatenate([traj.actions[1:],np.zeros_like(traj.actions[:1])],axis=0)
                        ))

                    idxes.append((
                        np.array([i]*T), np.arange(0,T)
                    ))
                
                traj_idx, t_idx = [np.concatenate(e).astype(np.int32) for e in zip(*idxes)]
                states, actions, rewards, dones, next_states, next_actions = \
                    [np.concatenate(e).astype(np.float32) for e in zip(*items)]
                discount = self.gamma * (1 - dones)

                D = tf.data.Dataset.from_tensor_slices(((traj_idx,t_idx,t_idx+1),(states,actions,rewards,discount,next_states,next_actions)))

            if not include_idx:
                D = D.map(lambda debug_info, data: data)
        else:
            parse = partial(self._parse,self.gamma,trajs,n_steps,sarsa,valid_next_only,t_0_only)

            # n-step Dataset
            D = tf.data.Dataset.from_generator(parse,output_signature=self.output_signature)
            if not include_idx:
                D = D.map(lambda debug_info, data: data)
            D = D.cache()

        return D

    @gin.configurable
    def prepare_dataset(self,
        type,batch_size,shuffle_size=100_000,repeat=True,take=-1,window_size=-1,shuffle_seed=None,
        prefetch=True,debug_info=False,
    ):
        parse = partial(self._parse,self.gamma,getattr(self,f'{type}_trajs'),self.n_steps,self.sarsa,self.valid_next_only,t_0_only=False)

        # n-step Dataset
        D = tf.data.Dataset.from_generator(parse,output_signature=self.output_signature)
        if not debug_info:
            D = D.map(lambda debug_info, data: data)
        if repeat:
            D = D.cache()

        # manipulate as you want. (for training in general)
        if shuffle_size == 'max':
            for i,_ in enumerate(tqdm(D,desc='caching', unit=' training samples', unit_scale=True)): pass
            D = D.shuffle(i,reshuffle_each_iteration=repeat,seed=shuffle_seed)
        elif shuffle_size > 0:
            D = D.shuffle(shuffle_size,reshuffle_each_iteration=repeat,seed=shuffle_seed)

        if repeat is True: D = D.repeat() # repeat indefinitely
        elif repeat > 0: D = D.repeat(count=repeat)

        if take > 0: D = D.take(take)

        if batch_size > 0:
            D = D.batch(batch_size)

        if window_size > 0:
            D = tf.data.Dataset.zip((D,)*window_size)

        if prefetch:
            D = D.prefetch(tf.data.experimental.AUTOTUNE)

        return D

    @gin.configurable
    def t_0_batch(self,
        type,batch_size,shuffle_size=1000,take=-1,shuffle_seed=None,
        prefetch=True,debug_info=False,
    ):
        parse = partial(self._parse,self.gamma,getattr(self,f'{type}_trajs'),self.max_len,self.sarsa,self.valid_next_only,t_0_only=True)

        # n-step Dataset
        D = tf.data.Dataset.from_generator(parse,output_signature=self.output_signature)
        if not debug_info:
            D = D.map(lambda debug_info, data: data)

        if shuffle_size > 0: D = D.shuffle(shuffle_size,reshuffle_each_iteration=False,seed=shuffle_seed)
        if take > 0: D = D.take(take)
        D = D.cache()

        if batch_size > 0: D = D.batch(batch_size)
        if prefetch: D = D.prefetch(tf.data.experimental.AUTOTUNE)

        return D

    @gin.configurable
    def eval_batch(self,
        type,batch_size,shuffle_size=100_000,take=-1,shuffle_seed=None,
        prefetch=True, debug_info=False,
    ):
        parse = partial(self._parse,self.gamma,getattr(self,f'{type}_trajs'),self.max_len,self.sarsa,self.valid_next_only,t_0_only=False)

        # n-step Dataset
        D = tf.data.Dataset.from_generator(parse,output_signature=self.output_signature)
        if not debug_info:
            D = D.map(lambda debug_info, data: data)

        if shuffle_size > 0: D = D.shuffle(shuffle_size,reshuffle_each_iteration=False,seed=shuffle_seed)
        if take > 0: D = D.take(take)
        D = D.cache()

        if batch_size > 0: D = D.batch(batch_size)
        if prefetch: D = D.prefetch(tf.data.experimental.AUTOTUNE)

        return D

@gin.configurable
class D4RL_Dataset(Dataset):
    @staticmethod
    def _parse_v0(env_id, shape_reward, clip_actions):
        import gym, d4rl

        env = gym.make(env_id)
        dataset = env.get_dataset()
        obs, acs, rs, dones =\
            dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals']

        if shape_reward == True:
            if 'antmaze' in env_id:
                rs -= 1.0 #https://github.com/ikostrikov/implicit_q_learning/blob/c1ec002681ff43ef14c3d38dec5881faeca8f624/train_offline.py#L69

        #assert np.all(acs >= env.action_space.low) and np.all(acs <= env.action_space.high)
        
        if clip_actions:
            acs = np.clip(acs,env.action_space.low + 1e-5, env.action_space.high - 1e-5)

        # hopper-medium-replay-v0 & hopper-medium-expert-v0 contains an action that is outside of possible action bounds. :(
        # so clip it.
        acs = np.clip(acs,env.action_space.low,env.action_space.high)

        def _parse(obs,actions,rewards,dones,trim_first_T,max_episode_steps):
            trajs = []
            start = trim_first_T
            while start < len(dones):
                end = start

                while end != 1000000 - 1 and end < len(dones) - 1 and \
                    (not dones[end] and end - start + 1 < max_episode_steps):
                    end += 1

                if dones[end]:
                    # the trajectory ends normally.
                    # since the next state will not be (should not be, actually) used by any algorithms,
                    # we add null states (zero-states) at the end.

                    traj = Trajectory(
                        states = np.concatenate([obs[start:end+1],np.zeros_like(obs[0])[None]],axis=0).astype(np.float32),
                        actions = actions[start:end+1].astype(np.float32),
                        rewards = rewards[start:end+1].astype(np.float32),
                        dones = dones[start:end+1].astype(np.bool),
                        frames = None,
                    )

                    assert np.all(traj.dones[:-1] == False) and traj.dones[-1]

                else:
                    # episodes end unintentionally (terminate due to timeout, cut-off when concateante two trajectories, or etc).
                    # since the next-state is not available, it drops the last action.

                    traj = Trajectory(
                        states = obs[start:end+1].astype(np.float32),
                        actions = actions[start:end].astype(np.float32),
                        rewards = rewards[start:end].astype(np.float32),
                        dones = dones[start:end].astype(np.bool),
                        frames = None,
                    )

                    assert np.all(traj.dones == False)

                if len(traj.states) > 1: # some trajectories are extremely short in -medium-replay dataset (due to unexpected timeout caused by RLKIT); https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671
                    trajs.append(traj)

                start = end + 1

            return trajs

        if env_id == 'halfcheetah-medium-replay-v0':
            trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
        elif env_id == 'halfcheetah-medium-v0':
            trajs = _parse(obs,acs,rs,dones,899,env._max_episode_steps-1) # why env._max_episode_stpes - 1? it is questionable, but it looks a valid thing to do.
        elif env_id == 'halfcheetah-expert-v0':
            trajs = _parse(obs,acs,rs,dones,996,env._max_episode_steps-1)
        elif env_id == 'halfcheetah-medium-expert-v0':
            trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],899,env._max_episode_steps-1) + \
                _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],996,env._max_episode_steps-1)
        elif env_id == 'hopper-medium-v0':
            trajs = _parse(obs,acs,rs,dones,211,env._max_episode_steps)
        elif env_id == 'hopper-expert-v0':
            trajs = _parse(obs,acs,rs,dones,309,env._max_episode_steps-1)
        elif env_id == 'hopper-medium-expert-v0': # actually, expert + mixed
            trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],309,env._max_episode_steps-1) + \
                _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],0,env._max_episode_steps-1)
        elif env_id == 'walker2d-medium-v0':
            trajs = _parse(obs,acs,rs,dones,644,env._max_episode_steps)
        elif env_id == 'walker2d-expert-v0':
            trajs = _parse(obs,acs,rs,dones,487,env._max_episode_steps-1)
        elif env_id == 'walker2d-medium-expert-v0': # actually, expert + mixed
            trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],644,env._max_episode_steps) + \
                _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],487,env._max_episode_steps-1)
        elif env_id in ['halfcheetah-random-v0', 'walker2d-random-v0', 'hopper-random-v0', 'walker2d-medium-replay-v0', 'hopper-medium-replay-v0']:
            trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps-1)
        elif env_id in ['pen-expert-v0', 'hammer-expert-v0', 'door-expert-v0', 'relocate-expert-v0']:
            trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
        elif env_id in ['door-human-v0','relocate-human-v0','hammer-human-v0']:
            # Note that `timeout` in -human type is not reliable!!!
            trajs = _parse(obs,acs,rs,dones,0,np.inf)
            #trim out the last item; (since the given trajectory is always unfinished one.); basically treat as a timeout
            trajs = [Trajectory(states=traj.states[:-1],actions=traj.actions[:-1],rewards=traj.rewards[:-1],dones=traj.dones[:-1],frames=None) for traj in trajs]
            for traj in trajs: assert np.all(~traj.dones) #human demo never finishes; always timeout
        elif env_id in ['pen-human-v0']:
            # Note that `timeout` in -human type is not reliable!!!
            dones[199::200] = True # the length of human demo is always 200.
            trajs = _parse(obs,acs,rs,dones,0,np.inf)
            #trim out the last item; (since the given trajectory is always unfinished one.); basically treat as a timeout
            trajs = [Trajectory(states=traj.states[:-1],actions=traj.actions[:-1],rewards=traj.rewards[:-1],dones=traj.dones[:-1],frames=None) for traj in trajs]
            for traj in trajs: assert np.all(~traj.dones) #human demo never finishes; always timeout
        elif env_id in ['door-cloned-v0','relocate-cloned-v0','hammer-cloned-v0']:
            # First half is human trajectories (repeated), and the rest half is the expert
            # It is pretty sure that the dataset is generated before the `hand_dapg_combined.py` script.
            # Since (1) in there case, the concatenation is [expert + human] not [human + expert], and there is no data overlap between `-expert` data and `-human` data.
            human_trajs = _parse(obs[:500000],acs[:500000],rs[:500000],dones[:500000],0,np.inf)
            human_trajs = [Trajectory(states=traj.states[:-1],actions=traj.actions[:-1],rewards=traj.rewards[:-1],dones=traj.dones[:-1],frames=None) for traj in human_trajs[:-1]] + human_trajs[-1:]

            expert_trajs = _parse(obs[500000:],acs[500000:],rs[500000:],np.zeros_like(dones[500000:]),0,env._max_episode_steps) # expert_trajs always timeout in door, relocate, hammer, but there is one location where done is set True.

            trajs = human_trajs + expert_trajs
            for traj in trajs: assert np.all(~traj.dones) # there is no early termination in those datasets
        elif env_id in ['pen-cloned-v0']:
            dones[:250_000][199::200] = True
            human_trajs = _parse(obs[:250000],acs[:250000],rs[:250000],dones[:250000],0,np.inf)
            human_trajs = [Trajectory(states=traj.states[:-1],actions=traj.actions[:-1],rewards=traj.rewards[:-1],dones=traj.dones[:-1],frames=None) for traj in human_trajs]

            expert_trajs = _parse(obs[250000:],acs[250000:],rs[250000:],dones[250000:],0,env._max_episode_steps)

            trajs = human_trajs + expert_trajs
            ### for traj in trajs: assert np.all(~traj.dones) # Early termination does exist in pen, and actually expert demo fails a few times. (94 times)
        else:
            trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)

        return trajs

    @staticmethod
    def _parse_rest(env_id, drop_trailings, shape_reward, clip_actions):
        import gym, d4rl

        env = gym.make(env_id)
        dataset = env.get_dataset()
        obs, actions, rewards, terminals, timeouts =\
            dataset['observations'],\
            dataset['actions'],\
            dataset['rewards'],\
            dataset['terminals'],\
            dataset['timeouts']

        if shape_reward == True:
            if 'antmaze' in env_id:
                rewards -= 1.0 #https://github.com/ikostrikov/implicit_q_learning/blob/c1ec002681ff43ef14c3d38dec5881faeca8f624/train_offline.py#L69
        
        if clip_actions:
            actions = np.clip(actions,env.action_space.low + 1e-5, env.action_space.high - 1e-5)

        assert len(obs) == len(actions) == len(rewards) == len(terminals) == len(timeouts)
        N = len(obs)

        trajs = []

        start = 0
        while start < N:
            end = start
            while not (terminals[end] or timeouts[end]) and end < N-1:
                end += 1

            if timeouts[end] or (end == N-1 and not drop_trailings):
                # the trajectory ends due to some external cut-offs
                # since the next-state is not available, it drops the last action.

                traj = Trajectory(
                    states = obs[start:end+1].astype(np.float32),
                    actions = actions[start:end].astype(np.float32),
                    rewards = rewards[start:end].astype(np.float32),
                    dones = terminals[start:end].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones == False)

            elif terminals[end]:
                # the trajectory ends normally.
                # since the next state will not be (should not be, actually) used by any algorithms,
                # we add null states (zero-states) at the end.

                traj = Trajectory(
                    states = np.concatenate([obs[start:end+1],np.zeros_like(obs[0])[None]],axis=0).astype(np.float32),
                    actions = actions[start:end+1].astype(np.float32),
                    rewards = rewards[start:end+1].astype(np.float32),
                    dones = terminals[start:end+1].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones[:-1] == False) and traj.dones[-1]

            elif end == N-1 and drop_trailings:
                break

            else:
                assert False

            if len(traj.states) > 1: # some trajectories are extremely short in -medium-replay dataset (due to unexpected timeout caused by RLKIT); https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671
                trajs.append(traj)

            start = end + 1

        return trajs

    @staticmethod
    def _parse_d4rl(env_id,shape_reward,clip_actions,normalize_reward):
        if env_id.split('-')[-1] == 'v0' and 'antmaze' not in env_id:
            trajs = D4RL_Dataset._parse_v0(env_id, shape_reward, clip_actions)
        else:
            trajs = D4RL_Dataset._parse_rest(env_id, False, shape_reward, clip_actions)

        if normalize_reward:
            # Only apply normalization for hopper, walker2d, halfcheetah environments.

            if ('hopper' in env_id) or ('walker2d' in env_id) or ('halfcheetah' in env_id):
                R = [np.sum(traj.rewards) for traj in trajs]
                min_R, max_R = min(R), max(R)

                scale = 1000. / (max_R - min_R) 

                for traj in trajs:
                    traj.rewards[:] = traj.rewards[:] * scale

        return trajs

    def __init__(self,seed=None,env_id=None,shape_reward=True,clip_actions=True,normalize_reward=True,**kwargs):
        trajs = self._parse_d4rl(env_id,shape_reward,clip_actions,normalize_reward)

        self.env_id = env_id

        super().__init__(seed=seed,trajs=trajs,**kwargs)

@gin.configurable
class RoboMimicDataset(Dataset):
    @staticmethod
    def parse(env_id,split,ignore_done,truncate_if_done,shape_reward):
        import gym, arq.modules.robomimic_env
        env = gym.make(env_id)
        trajs = env.get_dataset(split,ignore_done=ignore_done,truncate_if_done=truncate_if_done,shape_reward=shape_reward)
        return trajs

    def __init__(self,seed,env_id,split,ignore_done=False,truncate_if_done=False,shape_reward=True,**kwargs):
        trajs = self.parse(env_id,split,ignore_done,truncate_if_done,shape_reward)

        self.env_id = env_id
        super().__init__(seed=seed,trajs=trajs,**kwargs)