import argparse
import logging
import os
import random
import logging
import psutil

import gin
import numpy as np
import tensorflow as tf
import ray
from matplotlib import pyplot as plt
import tfplot

from arq.modules.parallel_batch_env import BatchRemoteEnv
from arq.modules.utils import setup_logger, tqdm, write_gin_config

@gin.configurable
def init_batch_env(
    env_id,
    num_parallel_envs=10,
):
    if not ray.is_initialized():
        num_cpus = min(len(os.sched_getaffinity(0)),num_parallel_envs)
        print(f'Parallel Envs Using {num_cpus} Cpus...')

        ray.init(include_dashboard=False,num_cpus=num_cpus,num_gpus=0) #,local_mode=True)

    # For some compute node whose memory is not enough...
    large_memory_footprint_envs = [
        'lift-low-mg-v0',
        'lift-low-mg-sparse-v0',
        'lift-low-ph-v0',
        'lift-low-mh-v0',
        'can-low-mg-v0',
        'can-low-mg-sparse-v0',
        'can-low-ph-v0',
        'can-low-mh-v0',
        'square-low-ph-v0',
        'square-low-mh-v0',
        'transport-low-ph-v0',
        'transport-low-mh-v0',
        'toolhang-low-mh-v0',
    ]
    device_memory = psutil.virtual_memory().total // 1024**3 #memory in GB

    if env_id in large_memory_footprint_envs and device_memory <= 16:
        num_parallel_envs = min(num_parallel_envs,10)

    batch_env = BatchRemoteEnv(env_id, num_envs=num_parallel_envs)
    return batch_env

@gin.configurable
def run_pi_parallel(
    pi, ### if pi is None, model.build_pi() will be used.
    batch_env, #should be singleton
    num_trajs,
    stochastic=False,
    debug=False,
    model=None, ### All it needs is build_pi function
    **kwargs,
):
    summary_writer = logging.getLogger('summary_writer')

    if pi is None:
        pi = model.build_pi()

    def eval_policy(u):
        trajs = batch_env.unroll_till_end(pi, stochastic, num_trajs=num_trajs, debug=debug)

        Ts = [len(traj.states) for traj in trajs]
        returns = [np.sum(traj.rewards) for traj in trajs]

        norm_returns = [batch_env.get_normalized_score(r) for r in returns]

        summary_writer.info('raw',f'eval.{__name__}/mean_eps_length',np.mean(Ts),u)
        summary_writer.info('raw',f'eval.{__name__}/mean_eps_return',np.mean(returns),u)
        summary_writer.info('raw',f'eval.{__name__}/mean_eps_norm_return',np.mean(norm_returns),u)

        if len(returns) > 1:
            summary_writer.info('histogram',f'eval.{__name__}/eps_return',returns,u)
            summary_writer.info('histogram',f'eval.{__name__}/eps_norm_return',norm_returns,u)

        return (Ts, returns, norm_returns)

    return eval_policy

@gin.configurable
def likelihood(
    model,
    dataset,
    ### gin configurable
    build_log_likelihood,
    write=False,
    write_key='',
    **kwargs,
):
    """
    Evaluation code for 2-dimensional distribution learning
    """
    sde = model
    log_likelihood = build_log_likelihood(sde)

    s,a = zip(*[(s.numpy(),a.numpy()) for s,a in dataset.epoch()])
    s,a = np.stack(s,axis=0), np.stack(a,axis=0)

    s_gap = 0.2 * (np.max(s) - np.min(s))   
    a_gap = 0.2 * (np.max(a) - np.min(a))
    s_test, a_test = np.meshgrid(
        np.linspace(np.min(s) - s_gap, np.max(s) + s_gap, num=100),
        np.linspace(np.min(a) - a_gap, np.max(a) + a_gap, num=100)
    )

    def eval(u):
        # calc log_p(x) for training items
        log_p_x = np.mean(log_likelihood(s,a))
        if write:
            logging.getLogger('summary_writer').info('raw',f'eval.{__name__}/log_likelihood',log_p_x,u+1)
        else:
            print(f'training log_p_x: {log_p_x}')

        # calc log_p(x) for general items
        fig,ax = tfplot.subplots()

        log_p_x = log_likelihood(s_test.ravel()[:,None], a_test.ravel()[:,None]).reshape(s_test.shape)
        p_x = np.exp(log_p_x)

        c = ax.pcolormesh(s_test,a_test,p_x,cmap='terrain',alpha=1.0,shading='auto',vmax=1.0)
        #c = ax.pcolormesh(s_test,a_test,np.clip(log_p_x,-100,100),cmap='terrain',alpha=1.0,shading='auto')
        fig.colorbar(c,ax=ax,location='left')

        # Ground-truth
        ax.scatter(s,a,marker='x',color='black')

        if write:
            logging.getLogger('summary_writer').info('img',f'eval.{__name__}/{write_key}sample',fig,u)
            plt.close(fig)
        else:
            return fig

    return eval