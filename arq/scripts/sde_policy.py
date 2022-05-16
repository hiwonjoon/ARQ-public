import argparse
import random
import logging

import gin
import numpy as np
import tensorflow as tf

from arq.modules.utils import setup_logger, write_gin_config
from arq.modules.eval import run_pi_parallel

@gin.configurable
def test(
    args,
    log_dir,
    seed,
    ##### gin controlled
    SDE,
    SDE_chkpt,
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)

    if SDE is None:
        model = gin.query_parameter('arq.scripts.train_sde_conditional.run.SDE').scoped_configurable_fn()
    else:
        model = SDE()
    model.load_weights(SDE_chkpt)

    pi = model.build_pi()
    
    eval = run_pi_parallel(pi=pi)
    #eval = run_pi(model=model,env_id=env_id,num_trajs=num_trajs,debug=True)

    Ts, returns, norm_returns = eval(0)
    print(returns, np.mean(returns))
    print(norm_returns, np.mean(norm_returns))

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

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

    import arq.scripts.sde_policy as this
    this.test(args,**vars(args))
