"""
Train Diffusion Model -- Stochastic Differential Equation(SDE)
"""
import os
import random
import logging
import argparse
from pathlib import Path

import gin
import numpy as np
import tensorflow as tf

from arq.modules.utils import setup_logger, tqdm, write_gin_config

@gin.configurable(module=__name__)
def run(
    args,
    log_dir,
    seed,
    ########## gin controlled.
    SDE,
    Dataset,
    epoch_fn_name,
    Evals,
    eval_periods, # generate 100 trajectories
    num_updates,
    log_period,
    save_period,
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    chkpt_dir = Path(log_dir).resolve()/'chkpt'
    chkpt_dir.mkdir(parents=True,exist_ok=True)

    # Define Dataset
    dataset = Dataset(seed=seed)
    epoch = getattr(dataset,epoch_fn_name)()

    # Define algorithm
    model = SDE()
    
    update, reports = model.prepare_update(epoch)
    
    try:
        evals = [e(seed=seed,model=model,dataset=dataset,pi=None) for e in Evals]
        eval_periods = np.array(eval_periods)

        assert len(evals) == len(eval_periods)
    except Exception as e:
        raise e

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        for u in tqdm(range(num_updates)):
            update()

            # log
            if (u+1) % log_period == 0:
                for name,item in reports.items():
                    val = item.result().numpy()
                    summary_writer.info('raw',f'{__name__}/{name}',val,u+1)
                    item.reset_states()

            # save
            if (u+1) % save_period == 0:
                model.save_weights(os.path.join(chkpt_dir,f'model-{u+1}.tf'))

            # eval
            for idx in np.where((u+1) % eval_periods == 0)[0]:
                evals[idx](u+1)

    except KeyboardInterrupt:
        pass

    model.save_weights(os.path.join(log_dir,f'model.tf'))

    logger.info('-------Gracefully finalized--------')
    logger.info('-------Bye Bye--------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file', nargs='*')
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

    import arq.scripts.train_sde_conditional as this
    this.run(args,**vars(args))