"""
Sample from score-based model & save it to a file. (For future use.)
"""
import logging
import os
import random
import argparse
from pathlib import Path
import pickle
import gzip

import gin
import numpy as np
import tensorflow as tf

from arq.modules.utils import setup_logger, tqdm, write_gin_config
from arq.algo.sde_conditional import build_pc_sampler, build_log_likelihood

@gin.configurable
def run(
    args,
    log_dir,
    ### gin configurables
    SDE = None,
    SDE_chkpt = None,
    Dataset = None, # should have `trajs` which is non-stochastic trajectories
    build_sampler = build_pc_sampler,
    build_log_likelihood = build_log_likelihood,
    batch_size = 100,
    num_samples = 30,
    do_action_probs=True,
    do_action_cands=True,
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')

    assert 'temp' in str(log_dir) or not os.path.exists(os.path.join(log_dir,'candidate_actions.pkl')), f'{log_dir}/candidate_actions.pkl exist!'

    if SDE is None:
        model = gin.query_parameter('arq.scripts.train_sde_conditional.run.SDE').scoped_configurable_fn()
    else:
        model = SDE()
    model.load_weights(SDE_chkpt)

    if Dataset is None:
        dataset = gin.query_parameter('arq.scripts.train_sde_conditional.run.Dataset').scoped_configurable_fn(seed=None)
    else:
        dataset = Dataset(seed=None)

    sampler = build_sampler(sde=model)
    log_likelihood = build_log_likelihood(sde=model)

    # Prepare Dataset
    ptr, slices, traj_states, traj_actions, = 0, [], [], []
    for traj in dataset.trajs:
        traj_states.append(traj.states[:-1].astype(np.float32))
        traj_actions.append(traj.actions.astype(np.float32))
        slices.append((ptr,ptr+len(traj_states[-1])))
        ptr += len(traj_states[-1])

    ## Put in a single sequence
    traj_states = np.concatenate(traj_states,axis=0)
    traj_actions = np.concatenate(traj_actions,axis=0)
    assert len(traj_states) == slices[-1][1]

    D = tf.data.Dataset.from_tensor_slices((traj_states,traj_actions)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Do Sampling & Calculate Log Prob
    traj_action_probs = []
    traj_action_cands = []
    traj_action_cand_probs = []
    for i, (s,original_a) in enumerate(tqdm(D)):
        while True:
            try:
                if do_action_probs:
                    original_log_prob = log_likelihood(s,original_a) #[B]
                else:
                    original_log_prob = np.zeros([batch_size]).astype(np.float32)

                if do_action_cands:
                    rs = np.repeat(np.expand_dims(s,axis=1),num_samples,axis=1) #[B,N] + s_dim
                    rs_flat = rs.reshape([-1] + list(s.shape[1:]))

                    ra = sampler(s=rs_flat).numpy() #[B*N] + a_dim
                    rlog_prob = log_likelihood(rs_flat, ra) #[B*N]

                    a = ra.reshape([len(s),num_samples] + list(ra.shape[1:])) #[B,N] + a_dim
                    log_prob = rlog_prob.reshape([len(s),num_samples]) #[B,N]
                else:
                    a = np.zeros([len(s),num_samples] + list(original_a.shape[1:])).astype(np.float32)
                    log_prob = np.zeros([len(s),num_samples]).astype(np.float32)
                break
            except AssertionError:
                print('Error occured! We will repeat the process once again...')

        traj_action_probs.append(original_log_prob)
        traj_action_cands.append(a)
        traj_action_cand_probs.append(log_prob)

        if i % 100 == 0:
            summary_writer.info('raw',f'{__name__}/num_processed',i*batch_size,i)

    traj_action_probs = np.concatenate(traj_action_probs,axis=0)
    traj_action_cands = np.concatenate(traj_action_cands,axis=0)
    traj_action_cand_probs = np.concatenate(traj_action_cand_probs,axis=0)

    assert len(traj_action_cand_probs) == len(traj_action_cands) == len(traj_states)

    # Put back to original `traj` unit.
    action_prob = []
    candidate_actions = []
    candidate_actions_prob = []
    for beg,end in slices:
        action_prob.append(traj_action_probs[beg:end])
        candidate_actions.append(traj_action_cands[beg:end])
        candidate_actions_prob.append(traj_action_cand_probs[beg:end])

    if do_action_probs:
        with gzip.open(os.path.join(log_dir,'action_log_prob.pkl'),'wb') as f:
            pickle.dump(action_prob,f)
    
    if do_action_cands:
        with gzip.open(os.path.join(log_dir,'candidate_actions.pkl'),'wb') as f:
            pickle.dump(candidate_actions,f)

        with gzip.open(os.path.join(log_dir,'candidate_actions_log_prob.pkl'),'wb') as f:
            pickle.dump(candidate_actions_prob,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file',required=True, nargs='+')
    parser.add_argument('--config_params', nargs='*', default='')

    args = parser.parse_args()

    config_params = '\n'.join(args.config_params)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.seed is not None:
        #os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_global_generator(tf.random.Generator.from_seed(args.seed))

    import arq.scripts.sampling as this
    gin.parse_config_files_and_bindings(args.config_file, config_params)
 
    this.run(args,**vars(args))