import random
import gzip
import os
import pickle
import argparse
import logging
from pathlib import Path

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
        action_likelihood_pkl,
        action_samples_pkl,
        action_samples_likelihood_pkl,
    ):
        self.base_dataset = BaseDataset(seed=seed)

        with gzip.open(os.path.expanduser(action_samples_pkl),'rb') as f:
            self.action_samples = pickle.load(f)
            self.num_samples = self.action_samples[0].shape[1] #[T,N] + ac_dim

        with gzip.open(os.path.expanduser(action_samples_likelihood_pkl),'rb') as f:
            self.action_samples_likelihood = pickle.load(f)

        with gzip.open(os.path.expanduser(action_likelihood_pkl),'rb') as f:
            self.action_likelihood = pickle.load(f)

        # Sanity Check
        assert self.action_samples_likelihood[0].shape[1] == self.num_samples
        assert len(self.base_dataset.trajs) == len(self.action_samples) == len(self.action_samples_likelihood)

        #for traj, actions, likelihoods in zip(self.base_dataset.trajs, self.action_samples, self.action_samples_likelihood):
        #    #print(len(traj.actions),len(actions),len(likelihoods))
        #    assert len(traj.actions) == len(actions) == len(likelihoods) # Robomimic Truncate Done --> length becomes different

    @gin.configurable
    def epoch(
        self,
        sarsa,
        ll_threshold,
        never_filter_gt_acs=True,
        **kwargs
    ):
        if never_filter_gt_acs:
            high = np.max(np.concatenate(self.action_likelihood))
            for ll in self.action_likelihood:
                ll[:] = high

        epoch = self.base_dataset.epoch(include_idx=True,sarsa=sarsa,**kwargs)

        if sarsa:
            def _get_samples(traj_idx, t_idx, nt_idx, discount):
                if discount == 0.:
                    return self.action_likelihood[traj_idx][t_idx].astype(np.float32),\
                        self.action_samples[traj_idx][t_idx].astype(np.float32),\
                        self.action_samples_likelihood[traj_idx][t_idx].astype(np.float32), \
                        np.zeros_like(self.action_likelihood[traj_idx][t_idx].astype(np.float32)),\
                        np.zeros_like(self.action_samples[traj_idx][t_idx].astype(np.float32)),\
                        np.zeros_like(self.action_samples_likelihood[traj_idx][t_idx].astype(np.float32))
                else:
                    return self.action_likelihood[traj_idx][t_idx].astype(np.float32),\
                        self.action_samples[traj_idx][t_idx].astype(np.float32),\
                        self.action_samples_likelihood[traj_idx][t_idx].astype(np.float32),\
                        self.action_likelihood[traj_idx][nt_idx].astype(np.float32),\
                        self.action_samples[traj_idx][nt_idx].astype(np.float32),\
                        self.action_samples_likelihood[traj_idx][nt_idx].astype(np.float32)

            def annotate(idxes, data):
                traj_idx, t_idx, nt_idx = idxes
                s, a, R, discount, ns, na = data

                a_log_prob, a_cands, a_cands_log_prob, na_log_prob, na_cands, na_cands_log_prob = tf.numpy_function(func=_get_samples,inp=[traj_idx,t_idx,nt_idx,discount], Tout=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

                s_beta_acs = tf.concat([a[None], a_cands],axis=0)
                s_beta_ll = tf.concat([a_log_prob[None], a_cands_log_prob],axis=0)

                ns_beta_acs = tf.concat([na[None], na_cands],axis=0)
                ns_beta_ll = tf.concat([na_log_prob[None], na_cands_log_prob],axis=0)

                # ideal penalty function we imagined.
                s_beta_acs_penalty = tf.cast(tf.where(s_beta_ll > ll_threshold, 0., float('inf')),tf.float32)
                ns_beta_acs_penalty = tf.cast(tf.where(ns_beta_ll > ll_threshold, 0., float('inf')),tf.float32)

                return s, a, s_beta_acs, s_beta_ll, s_beta_acs_penalty, R, discount, ns, ns_beta_acs, ns_beta_acs_penalty

            epoch = epoch.map(annotate).cache()

            return epoch
        else:
            def _get_samples(traj_idx, t_idx):
                return self.action_likelihood[traj_idx][t_idx].astype(np.float32),\
                    self.action_samples[traj_idx][t_idx].astype(np.float32),\
                    self.action_samples_likelihood[traj_idx][t_idx].astype(np.float32)

            def annotate(idxes, data):
                traj_idx, t_idx, nt_idx = idxes
                s, a, R, discount, ns = data

                a_log_prob, a_cands, a_cands_log_prob = tf.numpy_function(func=_get_samples,inp=[traj_idx,t_idx], Tout=[tf.float32,tf.float32,tf.float32])

                s_beta_acs = tf.concat([a[None], a_cands],axis=0)
                s_beta_ll = tf.concat([a_log_prob[None], a_cands_log_prob],axis=0)

                # ideal penalty function we imagined.
                s_beta_acs_penalty = tf.cast(tf.where(s_beta_ll > ll_threshold, 0., float('inf')),tf.float32)

                return s, a, s_beta_acs, s_beta_ll, s_beta_acs_penalty, R, discount, ns

            epoch = epoch.map(annotate).cache()

            return epoch

@gin.configurable
def prepare_update(
    Qs,
    pi,
    epoch,
    # gin configurable
    batch_size,
    shuffle_size,
    # Q-related settings
    K,
):
    reports= {}

    if pi is not None:
        pi_update_fn, pi_reports = pi.prepare_behavior_clone(epoch,batch_size=None)
        for key,val in pi_reports.items(): reports[f'pi/{key}'] = val

    q_update_fn, q_reports = Qs[0].prepare_update(friends=Qs[1:],bootstrap=False)
    for key,val in q_reports.items(): reports[f'q/{key}'] = val

    # Dataset
    if shuffle_size == 'max':
        try:
            D = epoch.shuffle(epoch.cardinality(),reshuffle_each_iteration=True)
        except:
            for i,_ in enumerate(tqdm(epoch,desc='caching', unit=' training samples', unit_scale=True)): pass
            D = epoch.shuffle(i,reshuffle_each_iteration=True)
    else:
        D = epoch.shuffle(shuffle_size,reshuffle_each_iteration=True)

    D = D.repeat()
    D = D.batch(batch_size)
    D = D.prefetch(tf.data.experimental.AUTOTUNE)
    D_samples = iter(D)

    def Q(ob,ac):
        qs = tf.stack([Q(ob,ac,use_target=True) for Q in Qs],axis=-1)
        return tf.reduce_min(qs,axis=-1)

    def get_quantile(v, K):
        return tf.reduce_min(tf.nn.top_k(v, K, sorted=False).values, axis=-1)

    @tf.function
    def _update(s,a,s_beta_acs,s_beta_ll,s_beta_acs_penalty,R,discount,ś,ś_beta_acs,ś_beta_acs_penalty):
        # Get value
        ś_q_candidates = Q(tf.repeat(tf.expand_dims(ś,axis=1),tf.shape(ś_beta_acs)[1],axis=1), ś_beta_acs) #[B, #beta_acs]
        ś_q_penalized = ś_q_candidates - ś_beta_acs_penalty #[B, #beta_acs]

        next_q_ś = get_quantile(ś_q_penalized,K)
        next_q_ś = tf.where(tf.math.greater(next_q_ś,ś_q_candidates[:,0]), next_q_ś, ś_q_candidates[:,0]) # use Q(s',a') if top_k quantile is smaller than Q(s',a')

        target_q = R + discount * next_q_ś

        # update Q
        q_update_fn(s,a,target_q)

        # Update pi w/ advantage-based BC
        if pi is not None:
            rs = tf.repeat(tf.expand_dims(s,axis=1),tf.shape(s_beta_acs)[1],axis=1)
            s_q_beta_penalized = Q(rs,s_beta_acs) - s_beta_acs_penalty #[B,N]

            s_q_beta_penalized_mean = tf.reduce_mean(tf.ragged.boolean_mask(s_q_beta_penalized, tf.math.is_finite(s_q_beta_penalized)),axis=-1,keepdims=True)
            s_q_beta_penalized_mean = tf.where(tf.math.is_finite(s_q_beta_penalized_mean), s_q_beta_penalized_mean, 0.) # mask if every q value is -inf.

            adv = s_q_beta_penalized - s_q_beta_penalized_mean #[B,N]

            pi_update_fn(s,s_beta_acs[:,0],adv[:,0])

    def update():
        _update(*next(D_samples))

    return update, reports

@gin.configurable(module=__name__)
def train(
    args,
    log_dir,
    seed,
    ########## gin controlled.
    ActionValue,
    num_qs,
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
    D = Dataset(seed=seed)
    epoch = D.epoch()

    # Define Network
    Qs = [ActionValue() for _ in range(num_qs)]
    pi = Policy() if Policy is not None else None

    # Prepare Update
    update, report = prepare_update(Qs, pi, epoch)

    # Prepare Eval
    evals = [e(seed=seed,Qs=Qs,pi=pi,dataset=D,model=None) for e in Evals]
    eval_periods = np.array(eval_periods)

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        for u in tqdm(range(num_updates)):
            update()

            # log
            if (u+1) % 100 == 0:
                for name,item in report.items():
                    val = item.result().numpy()
                    summary_writer.info('raw',f'{__name__}/{name}',val,u+1)
                    item.reset_states()

            # save
            if (u+1) % save_period == 0:
                if pi is not None:
                    pi.save_weights(os.path.join(log_dir,chkpt_dir,f'pi-{u+1}.tf'))
                for i,q in enumerate(Qs):
                    q.save_weights(os.path.join(log_dir,chkpt_dir,f'q{i}-{u+1}.tf'))

            # eval
            for idx in np.where((u+1) % eval_periods == 0)[0]:
                evals[idx](u+1)

    except KeyboardInterrupt:
        pass

    for i,q in enumerate(Qs):
        q.save_weights(os.path.join(log_dir,f'q{i}.tf'))

    if pi is not None:
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

    import arq.scripts.train_arq as this
    this.train(args,**vars(args))
