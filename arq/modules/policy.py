# Policy Implementations
import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from arq.modules.utils import tqdm

class Policy(Model):
    def __init__(
        self,
        scale,
    ):
        super().__init__()

        self.scale  = scale

    def __call__(self,ob,stochastic=True):
        if ob.ndim == 1:
            ob = ob[None]
            flatten = True
        else:
            flatten = False

        a, log_prob = self.action(ob,stochastic)
        a = tf.clip_by_value(a,-self.scale,self.scale)

        if flatten:
            a = a[0].numpy()

        return a, log_prob

    def action(self,ob,stochastic):
        raise NotImplementedError()

    def action_sample(self,ob,num_samples):
        raise NotImplementedError()

@gin.configurable()
class DeterministicPolicy(Policy):
    def __init__(
        self,
        Net,
        Preprocessor,
        scale=1.,
        act_noise=0.0,
        squash_output=True,
        **kwargs
    ):
        super().__init__(scale,**kwargs)

        self.net = Net()
        self.pp = Preprocessor()

        self.act_noise = act_noise
        self.squash_output = squash_output

    @tf.function
    def action(self,ob,stochastic=False):
        x = self.pp.to_x(ob)

        if self.squash_output:
            y = self.scale * tf.nn.tanh(self.net(x))
        else:
            y = self.scale * self.net(x)
        
        a = self.pp.to_a(y)

        if stochastic:
            noise = tf.random.get_global_generator().normal(tf.shape(a),stddev=self.act_noise)
            a += noise

            a = tf.clip_by_value(a,-self.scale,self.scale)

        return a, None

    @tf.function
    def action_sample(self,ob,num_samples):
        x = self.pp.to_x(ob)

        if self.squash_output:
            y = self.scale * tf.nn.tanh(self.net(x))
        else:
            y = self.scale * self.net(x)

        a = self.pp.to_a(y)

        noise = tf.random.get_global_generator().normal(tf.concat([[num_samples],tf.shape(a)],axis=0),stddev=self.act_noise)
        a = a[None] + noise

        a = tf.clip_by_value(a,-self.scale,self.scale)

        return a

    @gin.configurable(module=f'{__name__}.DeterministicPolicy')
    def prepare_behavior_clone(
        self,
        epoch,
        ### gin configurables
        Optimizer,
        batch_size,
        update_pp=False,
        beta=None,
        max_weight=100.,
        shuffle_size=None,
    ):
        if update_pp:
            self.pp.prepare(epoch.batch(100))
            optimizer = Optimizer(
                self.pp.trainable_variables + self.net.trainable_variables,
                self.pp.decay_vars + self.net.decay_vars)
        else:
            optimizer = Optimizer(
                self.net.trainable_variables,
                self.net.decay_vars,
            )

        reports = {}
        if beta is not None:
            reports['weight'] = tf.keras.metrics.Mean()

        reports.update(optimizer.reports)

        def _update(ob,gt,adv,_beta=beta):
            importance = tf.minimum(tf.exp(tf.math.multiply_no_nan(adv,_beta)),max_weight)
            reports['weight'](importance)

            with tf.GradientTape() as tape:
                a,_ = self.action(ob,stochastic=False)
                loss = 0.5 * tf.reduce_mean(importance * tf.reduce_mean((a-gt)**2,axis=-1))
            
            optimizer.minimize(tape,loss)
            return loss

        if batch_size is None:
            return _update, reports

        if shuffle_size is None:
            shuffle_size = int(epoch.cardinality())
        if shuffle_size < 0:
            for shuffle_size,_ in enumerate(tqdm(epoch, desc='counting', unit=' training samples', unit_scale=True)): pass

        D = epoch.shuffle(shuffle_size,reshuffle_each_iteration=True)
        D = D.repeat()
        D = D.batch(batch_size)
        D_samples = iter(D)

        if beta is None:
            @tf.function
            def update():
                s,a = next(D_samples)
                adv = tf.zeros([len(s)],tf.float32)
                return _update(s,a,adv,0.)
        else:
            @tf.function
            def update():
                s,a,adv = next(D_samples)
                return _update(s,a,adv)

        return update, reports

@gin.configurable
class EnsemblePolicy(Policy):
    def __init__(
        self,
        Base,
        num_ensembles,
    ):
        super(Policy, self).__init__()

        self.policies = [Base() for _ in range(num_ensembles)]
        for pi in self.policies[1:]:
            pi.pp = self.policies[0].pp # Use only the single preprocessor.
        
        self.scale = self.policies[0].scale

    @tf.function
    def action(self,ob,stochastic=False):
        action_candidates = tf.stack([pi.action(ob,stochastic)[0] for pi in self.policies],axis=1) #[B,N] + ac_dim
        idx = tf.random.get_global_generator().uniform(shape=[len(ob)],minval=0,maxval=len(self.policies),dtype=tf.int32)

        a = tf.gather_nd(
            action_candidates, #[B,N,y_dim]
            idx[:,None], #[B,1]
            batch_dims=1
        ) #[B,y_dim]

        return a, None

    @tf.function
    def action_sample(self,ob,num_samples):
        return np.concatenate([
            pi.action_sample(
                ob,
                num_samples//len(self.policies) + (1 if i < num_samples%len(self.policies) else 0))
            for i, pi in enumerate(self.policies)],axis=0)

    @gin.configurable(module=f'{__name__}.EnsemblePolicy')
    def prepare_behavior_clone(
        self,
        epoch,
        ### gin configurables
        train_split_ratio,
    ):
        cardinality = int(epoch.cardinality())
        if cardinality <= 0:
            for cardinality,_ in enumerate(tqdm(epoch, desc='counting', unit=' training samples', unit_scale=True)): pass

        updates, reports = [], []
        for i, pi in enumerate(self.policies):
            epoch_split_cardinality = int(cardinality * train_split_ratio)

            epoch_split = epoch.shuffle(cardinality,reshuffle_each_iteration=False).take(epoch_split_cardinality)
            update, report = pi.prepare_behavior_clone(epoch_split, update_pp = (i==0), shuffle_size=epoch_split_cardinality)

            updates.append(update)
            reports.append(report)

        agg_report = {key:type(item)() for key, item in reports[0].items()}

        @tf.function
        def update():
            for update in updates:
                update()

            for key, item in agg_report.items():
                for report in reports:
                    item(report[key].result())
                    report[key].reset_states()

        return update, agg_report

@gin.configurable(module=__name__)
class StochasticPolicy(Policy):
    def __init__(
        self,
        Net,
        Preprocessor, # preprocessor only be used for 'state'
        squash_distribution=False,
        tanh_mu=False, # even when distribution is squashed, apply tanh on `mu` can stabilize training
        scale=1.,
        log_std_min=-5.0,
        log_std_max=2.0,
        **kwargs,
    ):
        super().__init__(scale,**kwargs)

        self.net = Net()
        self.pp = Preprocessor()

        self.squash_distribution = squash_distribution
        self.tanh_mu = tanh_mu
        self.scale = scale

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def trainable_variables(self,include_pp):
        if include_pp:
            return self.pp.trainable_variables + self.net.trainable_variables
        else:
            return self.net.trainable_variables
        
    def decay_variables(self,include_pp):
        if include_pp:
            return self.pp.decay_vars + self.net.decay_vars
        else:
            return self.net.decay_vars
    
    def mean_std(self,ob):
        x = self.pp.to_x(ob)
        o = self.net(x)

        mu, log_std = tf.split(o,2,axis=-1)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mu, tf.math.exp(log_std)

    def _action(self,ob,stochastic=True):
        mu, std = self.mean_std(ob)

        if self.squash_distribution:
            base_dist = tfd.MultivariateNormalDiag(
                loc = mu,
                scale_diag = std
            )
            action_dist = tfd.TransformedDistribution(
                distribution = base_dist,
                bijector = tfb.Scale(scale=self.scale)(tfb.Tanh())
            )

            a = action_dist.sample() if stochastic else self.scale * tf.nn.tanh(mu)
        else:
            # sample from this distirbution could be out of action limits even with tanh.
            base_dist = tfd.MultivariateNormalDiag(
                loc = tf.nn.tanh(mu) if self.tanh_mu else mu,
                scale_diag = std
            )
            action_dist = tfd.TransformedDistribution(
                distribution = base_dist,
                bijector = tfb.Scale(scale=self.scale)
            )

            a = action_dist.sample() if stochastic else self.scale * (tf.nn.tanh(mu) if self.tanh_mu else mu)

        logp_a = action_dist.log_prob(a)

        return action_dist, (a, logp_a)

    @tf.function
    def action(self,ob,stochastic=True):
        _, (a, logp_a)= self._action(ob,stochastic)
        return a, logp_a

    @tf.function
    def action_sample(self,ob,num_samples):
        # Note that num_samples are PREpended, not appended
        action_dist, _ = self._action(ob)
        return action_dist.sample(num_samples)

    @gin.configurable(module=f'{__name__}.StochasticPolicy')
    def prepare_behavior_clone(
        self,
        epoch,
        ### gin configurables
        Optimizer,
        batch_size,
        update_pp=False,
        beta=None,
        max_weight=100.,
        shuffle_size=None,
    ):
        if update_pp:
            self.pp.prepare(epoch.batch(100))
            optimizer = Optimizer(
                self.trainable_variables(include_pp=True),
                self.decay_variables(include_pp=True))
        else:
            optimizer = Optimizer(
                self.trainable_variables(include_pp=False),
                self.decay_variables(include_pp=False))

        reports = {}
        reports.update(optimizer.reports)

        if beta is not None:
            reports['weight'] = tf.keras.metrics.Mean()

        @tf.function
        def _update(ob,gt,adv,_beta=beta):
            importance = tf.minimum(tf.exp(tf.math.multiply_no_nan(adv,_beta)),max_weight)
            reports['weight'](importance)

            with tf.GradientTape() as tape:
                action_dist, _ = self._action(ob,stochastic=True)
                logp_a = action_dist.log_prob(gt)
                loss = -tf.reduce_mean(importance * logp_a)
            
            optimizer.minimize(tape,loss)
            return loss

        if batch_size is None:
            return _update, reports

        if shuffle_size is None:
            shuffle_size = int(epoch.cardinality())
        if shuffle_size < 0:
            for shuffle_size,_ in enumerate(tqdm(epoch, desc='counting', unit=' training samples', unit_scale=True)): pass

        D = epoch.shuffle(shuffle_size,reshuffle_each_iteration=True)
        D = D.repeat()
        D = D.batch(batch_size)
        D = D.prefetch(tf.data.experimental.AUTOTUNE)
        D_samples = iter(D)

        if beta is None:
            @tf.function
            def update():
                s,a = next(D_samples)
                adv = tf.zeros([len(s)],tf.float32)
                return _update(s,a,adv,0.)
        else:
            @tf.function
            def update():
                s,a,adv = next(D_samples)
                return _update(s,a,adv)

        return update, reports

@gin.configurable(module=__name__)
class StateIndependentStochasticPolicy(StochasticPolicy):
    def __init__(
        self,
        ac_dim,
        tanh_mu=True,
        **kwargs,
    ):
        super().__init__(tanh_mu=tanh_mu,**kwargs)

        self.log_std = self.add_weight(
            name='log_std',
            shape=(1, ac_dim),
            initializer=tf.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32
        )

    def mean_std(self,ob):
        x = self.pp.to_x(ob)

        mu = self.net(x)
        log_std = tf.clip_by_value(self.log_std, self.log_std_min, self.log_std_max)

        return mu, tf.math.exp(log_std)

    def trainable_variables(self,include_pp):
        return super().trainable_variables(include_pp) + [self.log_std]

@gin.configurable(module=__name__)
class GaussianMixture(StochasticPolicy):
    def __init__(
        self,
        Net,
        Preprocessor, # preprocessor only be used for 'state'
        ac_dim,
        num_mixtures,
        squash_distribution=False,
        scale=1.,
        std_min=1e-4,
        **kwargs,
    ):
        Policy.__init__(self,scale,**kwargs)

        self.net = Net(out_dim = num_mixtures + 2 * num_mixtures * ac_dim)
        self.pp = Preprocessor()

        self.ac_dim = ac_dim
        self.num_mixtures = num_mixtures

        self.squash_distribution = squash_distribution
        self.scale = scale

        self.std_min = std_min
    
    def logits_mean_std(self,ob,stochastic):
        x = self.pp.to_x(ob)
        o = self.net(x)

        batch_dims = tf.shape(o)[:-1]

        logits, mus, std_logits = tf.split(o,[self.num_mixtures,self.num_mixtures*self.ac_dim,self.num_mixtures*self.ac_dim],axis=-1)

        mus = tf.reshape(mus,tf.concat([batch_dims,[self.num_mixtures,self.ac_dim]],axis=-1))
        std_logits = tf.reshape(std_logits,tf.concat([batch_dims,[self.num_mixtures,self.ac_dim]],axis=-1))

        if stochastic:
            stds = tf.nn.softplus(std_logits) + self.std_min
        else:
            stds = tf.ones_like(std_logits) * self.std_min

        return logits, mus, stds

    def _action(self,ob,stochastic=True):
        logits, mus, stds = self.logits_mean_std(ob,stochastic)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.MultivariateNormalDiag(mus,scale_diag=stds)
        )

        if self.squash_distribution:
            action_dist = tfd.TransformedDistribution(
                distribution = base_dist,
                bijector = tfb.Scale(scale=self.scale)(tfb.Tanh())
            )
        else:
            action_dist = tfd.TransformedDistribution(
                distribution = base_dist,
                bijector = tfb.Scale(scale=self.scale)
            )

        a = action_dist.sample()
        logp_a = action_dist.log_prob(a)

        return action_dist, (a, logp_a)