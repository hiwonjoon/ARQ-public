import gin
import tensorflow as tf
from tensorflow.keras import Model

@gin.configurable(module=__name__)
class ActionValue(Model):
    def __init__(
        self,
        Net,
        build_target_net=False,
        name=None,
    ):
        super().__init__()

        self.net = Net(name=None if name is None else name)

        if build_target_net:
            self.target_net = Net(name=None if name is None else f'{name}_target')
            self.update_target(0.)
        else:
            self.target_net = self.net

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    @property
    def trainable_decay_variables(self):
        return self.net.decay_vars

    @tf.function
    def __call__(self,ob,ac,use_target=True): # default is using target network
        if use_target:
            return tf.squeeze(self.target_net((ob,ac),training=False),axis=-1)
        else:
            return tf.squeeze(self.net((ob,ac),training=True),axis=-1)

    def sample(self,ob,ac,num_samples=1,use_target=True): # default is using target network
        return tf.repeat(tf.expand_dims(self(ob,ac,use_target),axis=-1),num_samples,axis=-1)

    #@tf.function
    def update_target(self,τ):
        main_net_vars = sorted(self.net.variables,key = lambda v: v.name)
        target_net_vars = sorted(self.target_net.variables,key = lambda v: v.name)
        assert len(main_net_vars) > 0 and len(target_net_vars) > 0 and len(main_net_vars) == len(target_net_vars), f'{len(main_net_vars)} != {len(target_net_vars)}'

        for v_main,v_target in zip(main_net_vars,target_net_vars):
            v_target.assign(τ*v_target + (1-τ)*v_main)

    @gin.configurable(module=f'{__name__}.ActionValue')
    def prepare_update(
        self,
        polyak,
        Optimizer,
        friends = [],
        feature_reg = 0.,
        bootstrap = True,
    ):
        """
        This function is for scripts/policy_evaluation.py
        """
        Qs = [self] + friends

        optimizer = Optimizer(
            [v for Q in Qs for v in Q.trainable_variables],
            [dv for Q in Qs for dv in Q.trainable_decay_variables]
        )

        reports= {
            'gradient_scale': tf.keras.metrics.Mean()
        }
        reports.update(optimizer.reports)

        if feature_reg > 0.:
            reports['feature_product'] = tf.keras.metrics.Mean()

        if bootstrap:
            @tf.function
            def update(s,a,R,discount,ś,á):
                target_q = R + discount * tf.reduce_min(tf.stack([Q(ś,á,use_target=True) for Q in Qs],axis=-1),axis=-1)

                with tf.GradientTape() as tape:
                    # TD loss
                    loss = []
                    for Q in Qs:
                        q = Q(s,a,use_target=False)

                        L = tf.reduce_mean(0.5 * (q - target_q)**2)
                        loss.append(L)

                        if feature_reg > 0.:
                            phi_1 = self.net.fv((s,a))
                            phi_2 = self.net.fv((ś,á))

                            L_reg = tf.matmul(tf.expand_dims(phi_1,axis=-1),tf.expand_dims(phi_2,axis=-1),transpose_a=True)

                            mask = tf.cast(discount > 0, tf.float32)
                            L_reg = tf.reduce_sum(mask*L_reg) / tf.reduce_sum(mask)
                            reports['feature_product'](L_reg)

                            loss.append(feature_reg * L_reg)

                    loss = tf.math.accumulate_n(loss)

                optimizer.minimize(tape,loss)

                # Calcualte gradient scale (for debugging)
                for Q in Qs:
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(a)
                        q = Q(s,a,use_target=False)
                    grad_scale = tf.reduce_mean(tf.linalg.norm(tape.gradient(q,a),axis=-1))
                    reports['gradient_scale'](grad_scale)

                for Q in Qs:
                    if Q.net != Q.target_net:
                        Q.update_target(polyak)

                return loss

            return update, reports
        else:
            @tf.function
            def update(s,a,target_q):
                with tf.GradientTape() as tape:
                    # TD loss
                    loss = []
                    for Q in Qs:
                        q = Q(s,a,use_target=False)

                        L = tf.reduce_mean(0.5 * (q - target_q)**2)
                        loss.append(L)

                    loss = tf.math.accumulate_n(loss)

                optimizer.minimize(tape,loss)

                for Q in Qs:
                    if Q.net != Q.target_net:
                        Q.update_target(polyak)

                return loss

            return update, reports