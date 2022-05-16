import gin
import tensorflow as tf

@gin.configurable
def build_penalized_Q(
    sde,
    ### Gin configured
    build_penalty_fn,
    Q,
    Q_chkpt,
    alpha,
): 
    p = build_penalty_fn(sde)
    
    q = Q()
    q.load_weights(Q_chkpt)
    
    def penalized_Q(s,a):
        return alpha * q(s,a) + p(s,a)

    return penalized_Q

@gin.configurable
def build_penalty_fn(
    sde,
    ### Gin configured
    build_log_likelihood,
    threshold=None, # if threshold is None, quantile will be used.
): 
    log_likelihood = build_log_likelihood(sde)

    def p(s,a): # ideal penalty function we imagined.
        ll = tf.cast(log_likelihood(s,a),tf.float32)
        masked_ll = tf.cast(tf.where(ll > threshold, 0., float('-inf')),tf.float32)

        return masked_ll

    return p