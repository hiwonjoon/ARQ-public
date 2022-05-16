"""
Behavior Cloning
"""
import os
import random
import argparse
import logging
from pathlib import Path
import gzip, pickle

import gin
import numpy as np
import tensorflow as tf

from arq.modules.utils import setup_logger, tqdm, write_gin_config

@gin.configurable
class Dataset(object):
    def __init__(
        self,
        seed,
        #### Gin configurable
        BaseDataset,
        Q,
        Q_chkpt,
        action_samples_pkl,
        action_samples_likelihood_pkl,
        action_likelihood_pkl=None,
    ):
        self.base_dataset = BaseDataset(seed=seed)

        self.q = Q()
        self.q.load_weights(Q_chkpt)

        with gzip.open(os.path.expanduser(action_samples_pkl),'rb') as f:
            self.action_samples = pickle.load(f)
            self.num_samples = self.action_samples[0].shape[1]

        with gzip.open(os.path.expanduser(action_samples_likelihood_pkl),'rb') as f:
            self.action_samples_likelihood = pickle.load(f)

        if action_likelihood_pkl is not None:
            with gzip.open(os.path.expanduser(action_likelihood_pkl),'rb') as f:
                self.action_likelihood = pickle.load(f)
        else:
            # if not given, treat as the best ll.
            self.action_likelihood = []
            for ll in self.action_samples_likelihood:
                self.action_likelihood.append(np.max(ll,axis=-1))

    @gin.configurable
    def awr_epoch(
        self,
        ll_threshold,
        never_filter_gt_acs=True,
        only_original_actions=True,
        **kwargs
    ):
        if never_filter_gt_acs:
            high = np.max(np.concatenate(self.action_likelihood))
            for ll in self.action_likelihood:
                ll[:] = high

        epoch = self.base_dataset.bc_epoch(include_idx=True,**kwargs)

        # Append Sample & Likelihoods
        def get_samples(traj_idx,t_idx):
            return self.action_likelihood[traj_idx][t_idx].astype(np.float32), self.action_samples[traj_idx][t_idx].astype(np.float32), self.action_samples_likelihood[traj_idx][t_idx].astype(np.float32)

        epoch = epoch.map(
            lambda s,a,traj_idx,t_idx: (s,a,*tf.numpy_function(func=get_samples,inp=[traj_idx,t_idx], Tout=[tf.float32,tf.float32,tf.float32]))
        )

        # Calculate Advantages for AWR
        @tf.function(input_signature=[
            tf.TensorSpec([None]+list(self.base_dataset.ob_dim), tf.float32),
            tf.TensorSpec([None]+list(self.base_dataset.ac_dim), tf.float32),
            tf.TensorSpec([None], tf.float32),
            tf.TensorSpec([None,self.num_samples]+list(self.base_dataset.ac_dim), tf.float32),
            tf.TensorSpec([None,self.num_samples], tf.float32),
        ])
        def calc_adv(s, a, action_log_likelihood, cand_actions, cand_actions_log_likelihoods):
            actions = tf.concat([tf.expand_dims(a,axis=1), cand_actions],axis=1)
            log_likelihoods = tf.concat([tf.expand_dims(action_log_likelihood,axis=1), cand_actions_log_likelihoods],axis=1)

            penalty = tf.cast(tf.where(log_likelihoods > ll_threshold, 0., float('inf')),tf.float32)

            rs = tf.repeat(tf.expand_dims(s,axis=1),tf.shape(actions)[1],axis=1)
            q_penalized = self.q(rs,actions) - penalty #[B,N]

            v = tf.reduce_mean(tf.ragged.boolean_mask(q_penalized, tf.math.is_finite(q_penalized)),axis=-1,keepdims=True)
            v = tf.where(tf.math.is_finite(v), v, 0.) # mask if every q value is -inf.

            adv = q_penalized - v #[B,N]

            return rs, actions, adv
        
        epoch = epoch.batch(
            1000 # for choose_best parallelism, batching
        ).map(
            calc_adv ## compute
        ).unbatch( ## then unbatch.
        )

        if only_original_actions:
            epoch = epoch.map(
                lambda rs, actions, adv: (rs[0], actions[0], adv[0])
            )
            epoch = epoch.cache()
        else:
            epoch = epoch.unbatch()

        return epoch

@gin.configurable(module=__name__)
def train_pi(
    args,
    log_dir,
    seed,
    ########## gin controlled.
    Policy,
    Dataset,
    # training loop
    num_updates,
    save_period, # in #updates
    Evals,
    eval_periods, # in #updates
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    chkpt_dir = Path(log_dir).resolve()/'chkpt'
    chkpt_dir.mkdir(parents=True,exist_ok=True)

    ########################
    # Define Dataset
    ########################
    dataset = Dataset(seed=seed)
    epoch = dataset.awr_epoch()

    # Define Network
    pi = Policy()
    update, report = pi.prepare_behavior_clone(epoch)

    # Prepare Eval
    evals = [e(seed=seed,pi=pi,dataset=dataset,model=None) for e in Evals]
    eval_periods = np.array(eval_periods)

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        for u in tqdm(range(num_updates)):
            _ = update()

            # log
            if (u+1) % 100 == 0:
                for name,item in report.items():
                    val = item.result().numpy()
                    summary_writer.info('raw',f'{__name__}/{name}',val,u+1)
                    item.reset_states()

            # eval
            for idx in np.where((u+1) % eval_periods == 0)[0]:
                evals[idx](u+1)

            # save
            if (u+1) % save_period == 0:
                pi.save_weights(os.path.join(log_dir,chkpt_dir,f'pi-{u+1}.tf'))

    except KeyboardInterrupt:
        pass

    pi.save_weights(os.path.join(log_dir,f'pi.tf'))

    logger.info('-------Gracefully finalized--------')
    logger.info('-------Bye Bye--------')

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

    gin.parse_config_files_and_bindings(args.config_file, config_params)

    import arq.scripts.bc as this
    this.train_pi(args,**vars(args))
