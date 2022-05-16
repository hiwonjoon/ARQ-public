"""
Conditional SDE;
    Solely dedicated for our purpose: we always want to model p(a|s).
    Therefore, we build a score function s(s,a,t)
        in the form of : R^(|s|+|a|+1) -> R^(|a|),
        which represent s(s,a,t) ~= \nabla log p(a|s,t)
"""
import gin
import numpy as np
import tensorflow as tf
from scipy import integrate

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
import tensorflow_probability as tfp
tfd = tfp.distributions

from arq.modules.utils import tqdm

gin.external_configurable(tf.nn.silu, 'swish')
gin.external_configurable(tf.nn.relu, 'relu')

@gin.configurable
class GaussianFourierProjection(Layer):
    """
        R --> R^|embed_dim|
    """
    def __init__(
        self,
        embed_dim,
        ### Gin configurable
        scale = 30.0, # size of projection vector
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.W = self.add_weight( # random projection vector
            name='W',
            shape=(1, embed_dim//2),
            initializer=tf.initializers.RandomNormal(stddev=scale),
            trainable=False,
            dtype=tf.float32,
        )

    @tf.function
    def call(self,t,training=None):
        """
        input:
            t: [B], [B,C,...]
        ouptut:
            embed: [B,W], [B,C,...,W]
        """
        t = tf.expand_dims(t,axis=-1)

        rank = t.shape.rank
        if rank == 2 or rank is None:
            embed = tf.matmul(t, self.W)
        else:
            embed = tf.tensordot(t, self.W, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = t.shape.as_list()
                output_shape = shape[:-1] + [self.W.shape[-1]]
                embed.set_shape(output_shape)

        embed = 2 * 3.1415 * embed

        return tf.concat([tf.math.sin(embed), tf.math.cos(embed)], axis=-1)

@gin.configurable(module=__name__)
class ScoreNet(Layer):
    """
    Conditional Score function

    features:
        1. Skip-connection: res-net style
        2. Gaussian Fourier Projection
    """
    def __init__(
        self,
        x_dim, # target dimension to infer
        c_dim, # conditional item dimension
        ### Gin configurable
        num_res_blocks,
        embed_dim,
        activation,
        use_bias=True
    ):
        super().__init__()

        self.activation = activation

        self.gf = GaussianFourierProjection(embed_dim)
        self._t_projection = Dense(
            embed_dim,
            activation = None,
            use_bias = use_bias,
        )
        self._t_projection.build((embed_dim,))

        self._x_projection = Dense(
            embed_dim,
            activation = None,
            use_bias = use_bias,
        )
        self._x_projection.build((x_dim+c_dim,))

        self._first = []
        self._second = []
        for l in range(num_res_blocks):
            _l1 = Dense(
                embed_dim,
                activation = None,
                use_bias=use_bias,
            )
            _l1.build((embed_dim,))

            self._first.append(_l1)

            _l2 = Dense(
                embed_dim,
                activation = None,
                use_bias=use_bias,
            )
            _l2.build((embed_dim,))

            self._second.append(_l2)

        self._out = Dense(
            x_dim,
            activation = None,
            use_bias = use_bias,
        )
        self._out.build((embed_dim,))

    #@tf.function
    def call(self,x,c,t,training=None):
        o = self._x_projection(tf.concat([x,c],axis=-1),training=training)
        t_e = self._t_projection(self.gf(t),training=training)

        for l1,l2 in zip(self._first,self._second):
            o_init = tf.identity(o)

            o += t_e
            o = l1(self.activation(o), training=training)

            o += t_e
            o = l2(self.activation(o), training=training)

            o = o_init + o

        o = self._out(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        layers = [self._x_projection, self._t_projection, self._out] + self._first + self._second

        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in layers
            ]

@gin.configurable
class SDE(Model):
    def __init__(
        self,
        Preprocessor,
        Score,
        eps_t=1e-5, # start of SDE
        T=1, # end of SDE
        ema=0., # exponential moving average
    ):
        super().__init__()

        self.pp = Preprocessor()
        self.x_dim = np.prod(self.pp.y_dims) # action-representation shape
        self.c_dim = np.prod(self.pp.x_dims) # state-representation shape

        self.score_net = Score(x_dim=self.x_dim,c_dim=self.c_dim)

        self.ema = ema
        if ema > 0.:
            self.ema_score_net = Score(x_dim=self.x_dim,c_dim=self.c_dim)
            self.update_ema_weights(0.)
        else:
            self.ema_score_net = self.score_net

        self.eps_t, self.T = eps_t, T

    def update_ema_weights(self,weight):
        if self.score_net == self.ema_score_net:
            return

        main_net_vars = sorted(self.score_net.variables, key = lambda v: v.name)
        ema_net_vars = sorted(self.ema_score_net.variables, key = lambda v: v.name)
        assert len(main_net_vars) > 0 and len(ema_net_vars) > 0 and len(main_net_vars) == len(ema_net_vars), f'{len(main_net_vars)} != {len(ema_net_vars)}'

        for v_main,v_ema in zip(main_net_vars,ema_net_vars):
            v_ema.assign(weight*v_ema + (1.-weight)*v_main)

    def score(self,x,c,t,training=False):
        if training:
            return self.score_net(x,c,t)
        else:
            return self.ema_score_net(x,c,t)

    def sde(self, x, t):
        """
        Stochastic Differential Equation.
        should provide drift (coefficient for dt ) and diffusion (coefficient for dw)
        """
        # we consider SDE that does not change given the context.
        raise NotImplementedError()

    def marginal_prob_std(self, x, t):
        # std of p(x_t|x_0); since SDE does not change given the context.
        raise NotImplementedError()

    def prior_dist(self):
        # p(x_1)
        raise NotImplementedError()

    def reverse_sde(self, x, c, t):
        """
        When SDE is in this form:
            dx = f(x,t) dt + g(t) dw
        the reverse diffusion process is:
            dx = [f(x,t) - g^2(t) score(x,c,t)] dt + g(t) dw
        """
        drift, diffusion = self.sde(x,t)
        drift_r = drift - tf.expand_dims(diffusion**2,axis=-1) * self.score(x,c,t)

        return drift_r, diffusion

    def reverse_ode(self, x, c, t):
        """
        When SDE is in this form:
            dx = f(x,t) dt + g(t) dw
        The associated reverse ODE is:
            dx = [f(x,t) - 0.5 * g^2(t) score(x,c,t)] dt
        """
        drift, diffusion = self.sde(x,t)
        drift = drift - 0.5 * tf.expand_dims(diffusion**2,axis=-1) * self.score(x,c,t)

        return drift, tf.zeros_like(diffusion)

    @gin.configurable
    def build_pi(
        self,
        ### gin configurable
        build_sampler,
        build_log_likelihood=None, #depreciated
        num_samples=1,
        build_grader=None, # (s,a) -> R; provide some number to select among multiple samples; higher grade will be selected.
        scale=float('inf'),
        advantage=False, # advantage formulation instead of raw Q value for stochastic policy
    ):
        sampler = build_sampler(self)

        if num_samples == 1:
            def pi(
                s,
                stochastic=False
            ):
                a = sampler(s=s)
                a = tf.clip_by_value(a,-scale,scale)

                return a.numpy(), None
        else:
            if build_grader is not None:
                grader = build_grader(sde=self)
            elif build_log_likelihood is not None:
                grader = build_log_likelihood(sde=self)
            else:
                assert False

            #@tf.function
            def pi(
                s,
                stochastic=False
            ):
                rs = np.repeat(np.expand_dims(s,axis=1),num_samples,axis=1) #[B,N] + x_dim
                rs_flat = rs.reshape([-1] + list(s.shape[1:]))

                ra_flat = sampler(s=rs_flat) #[B*N] + y_dim
                ra_flat = tf.clip_by_value(ra_flat,-scale,scale)

                # faster approximation of log_px?
                #drift, _ = self.reverse_sde(x,tf.ones(len(x)) * 1e-3)
                #log_px = -tf.linalg.norm(drift,axis=-1)

                grade = grader(rs_flat, ra_flat.numpy()) #[B*N]

                ra = tf.reshape(ra_flat, [len(s),num_samples] + ra_flat.get_shape()[1:])
                grade = tf.reshape(grade, [len(s),num_samples])

                if stochastic:
                    if advantage:
                        masked_mean = tf.reduce_mean(tf.ragged.boolean_mask(grade, tf.math.is_finite(grade)),axis=-1,keepdims=True)
                        masked_mean = tf.where(tf.math.is_finite(masked_mean), masked_mean, 0.)

                        grade = grade - masked_mean

                    best_idx = tf.random.categorical(logits=grade,num_samples=1)[:,0] #[B]
                else:
                    best_idx = tf.argmax(grade,axis=-1) #[B]
                    #print(tf.reduce_max(grade,axis=-1).numpy())

                best_a = tf.gather_nd(
                    ra, #[B,N,y_dim]
                    best_idx[:,None], #[B,1]
                    batch_dims=1
                ) #[B,y_dim]

                return best_a.numpy(), None
        
        return pi

    @gin.configurable(module=f'{__name__}')
    def prepare_update(
        self,
        epoch,
        #### Gin configurable
        Optimizer,
        batch_size,
        weighting, # \lambda(t) in denoising score matching objective; [None, 'marginal_std', 'likelihood']
        update_pp=True,
        shuffle_size=None,
    ):
        """
        We have Gaussian as marginal dist:
             p_0t(x(t) | x(0), c) = N( mu_at_t^2, sigma_at_t^2)
        Therefore,
            \nabla_{x_t} p_0t(x(t)|x(0), c) = - z(t) / sigma_at_t where z(t) ~ N(0,I)
            since \nabla log p(x) = - (x-mu) / sigma, where x ~ N(mu,sigma) 
        """
        if update_pp:
            self.pp.prepare(epoch.batch(100))
            optimizer = Optimizer(
                self.pp.trainable_variables + self.score_net.trainable_variables,
                self.pp.decay_vars + self.score_net.decay_vars)
        else: #for ensemble
            optimizer = Optimizer(
                self.score_net.trainable_variables,
                self.score_net.decay_vars)

        reports = {}
        reports.update(optimizer.reports)

        def _update(s,a):
            """
            input:
                s: [B] + s_dims
                a: [B] + a_dims
            """
            B = tf.shape(s)[0]

            # pick perturbing step t.
            t = tf.random.get_global_generator().uniform(shape=[B],minval=self.eps_t,maxval=self.T) # t ~ Uni(0,1); [B]

            with tf.GradientTape() as tape:
                x_0 = self.pp.to_y(a)
                c = self.pp.to_x(s)

                # sample x_t; perturbed version
                mean, std = self.marginal_prob_std(x_0, t)
                noise = tf.random.get_global_generator().normal(shape=tf.shape(x_0),mean=0.,stddev=1.) # shape of x_0
                x_t = mean + tf.expand_dims(std,axis=-1) * noise # due to the given SDE form, x_t can be calculated without following a path.

                # Score
                score = self.score(x_t,c,t,training=True)

                # loss
                if weighting is None:
                    loss = tf.reduce_mean(tf.reduce_sum((score + noise / std)**2,axis=-1))
                elif weighting == 'marginal_std':
                    loss = tf.reduce_mean(tf.reduce_sum((score * tf.expand_dims(std,axis=-1) + noise)**2,axis=-1))
                elif weighting == 'likelihood':
                    _, diffusion = self.sde(tf.zeros_like(x_t),t)
                    loss = tf.reduce_mean(diffusion**2 * tf.reduce_sum((score * tf.expand_dims(std,axis=-1) + noise)**2,axis=-1))
                else:
                    assert False

            optimizer.minimize(tape,loss)
            self.update_ema_weights(self.ema)
            return loss

        # Dataset
        if shuffle_size is None:
            shuffle_size = int(epoch.cardinality())
        if shuffle_size < 0:
            for shuffle_size,_ in enumerate(tqdm(epoch, desc='counting', unit=' training samples', unit_scale=True)): pass

        D = epoch.shuffle(shuffle_size,reshuffle_each_iteration=True)
        D = D.repeat()
        D = D.batch(batch_size)
        D = D.prefetch(tf.data.experimental.AUTOTUNE)
        D_samples = iter(D)

        @tf.function
        def update():
            s,a = next(D_samples)
            return _update(s,a)

        return update, reports

@gin.configurable
class VPSDE(SDE):
    """
    Variance Preserving SDE
        when it is discretized, it is same as Denoising diffusion probabilistic moeling (DDPM)
    
    Implemented SDE:
        dx = -0.5 \beta(t) dt + \beta(t) ** 0.5 dw
    where beta(t) = beta_0 + t * (beta_1 - beta_0); linear schedule

    Since drift coefficient is affine, we know that the marginal distribution is Gaussian:
        p_0t(x(t) | x(0) )
            = N(
                x(0) \exp[-0.5 * \int_0^t \beta(t)],
                (1 - \exp[- \int_0^t \beta(t)]) I
            )
    """
    def __init__(
        self,
        beta_min=0.1,
        beta_max=20.,
        eps_t= 1e-5, #to avoid small time-step region (numerical unstability)
        **kwargs,
    ):
        super().__init__(eps_t=eps_t,T=1.,**kwargs)

        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def score(self,x,c,t,training=False):
        raw_score = super().score(x,c,t,training)

        _, std = self.marginal_prob_std(tf.zeros_like(x), t)

        return raw_score / tf.expand_dims(std,axis=-1)

    def sde(self,x,t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * tf.expand_dims(beta_t,axis=-1) * x
        diffusion = beta_t ** 0.5

        return drift, diffusion

    def marginal_prob_std(self, x, t):
        coeff = 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0
        mean = tf.expand_dims(tf.math.exp(-0.5 * coeff),axis=-1) * x
        std = (1 - tf.math.exp(-coeff))**0.5

        return mean, std

    def prior_dist(self):
        return tfd.MultivariateNormalDiag(
            loc = [0.] * self.x_dim,
            scale_diag = [1.] * self.x_dim
        )

@gin.configurable
class SDE_Ensemble(SDE):
    def __init__(
        self,
        Base,
        num_ensembles,
    ):
        super(SDE, self).__init__()

        self.sdes = [Base() for _ in range(num_ensembles)]

        self.pp = self.sdes[0].pp
        self.x_dim = self.sdes[0].x_dim
        self.c_dim = self.sdes[0].c_dim

        for sde in self.sdes:
            sde.pp = self.pp # Use only the single preprocessor.

        self.eps_t, self.T = self.sdes[0].eps_t, self.sdes[0].T

    def score(self,x,c,t,training=False):
        return tf.add_n([sde.score(x,c,t,training) for sde in self.sdes]) / len(self.sdes) # take mean score

    def sde(self, x, t):
        return self.sdes[0].sde(x,t)

    def marginal_prob_std(self, x, t):
        return self.sdes[0].marginal_prob_std(x,t)

    def prior_dist(self):
        return self.sdes[0].prior_dist()

    @gin.configurable(module=f'{__name__}.SDE_Ensemble')
    def prepare_update(
        self,
        epoch,
        ### Gin configurable
        train_split_ratio,
    ):
        cardinality = int(epoch.cardinality())
        if cardinality <= 0:
            for cardinality,_ in enumerate(tqdm(epoch, desc='counting', unit=' training samples', unit_scale=True)): pass

        updates, reports = [], []
        for i, sde in enumerate(self.sdes):
            epoch_split_cardinality = int(cardinality * train_split_ratio)

            epoch_split = epoch.shuffle(cardinality,reshuffle_each_iteration=False).take(epoch_split_cardinality)
            update, report = sde.prepare_update(epoch_split, update_pp = (i==0), shuffle_size = epoch_split_cardinality)

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

@gin.configurable
def build_pc_sampler(
    sde,
    ### gin configurables
    num_t_steps,
    num_mcmc_steps = 1,
    gradient_snr = 0.16,
    eps_t = 1e-3,
    batch_independent_step_size = False,
): 
    @tf.function(jit_compile=True, input_signature=[tf.TensorSpec(shape=[None,sde.pp.ob_dim],dtype=tf.float32)])
    def predictor_corrector(
        s,
    ):
        # NOTE: Be aware that the quality of PC sampling can be affected by the batch-size.
        # It actively adjust the step_size, based on the gradient norm computed across the batch.
        """
        perform Langevin MCMC (corrector), Euler-Maruyama (predictor) repeatedly
            Langevin MCMC:
                we adaptively adjust the langevin step size using gradient norm across batch.
            Euler-Maruyama SDE solver:
                Euler-Maruyama method is an discretized method;

        if mcmc_steps == 0, then it becomes Euler-Maruyama SDE solver.
        """
        num_samples = len(s)
        c = sde.pp.to_x(s)

        shape = [num_samples,sde.x_dim]

        times = tf.linspace(sde.T,eps_t,num_t_steps)
        delta_t = times[0] - times[1] # this is positive.

        # sample from prior p_1(x_1)
        prior = sde.prior_dist()
        x = prior.sample(num_samples) #[N, x_dim]
        x_mean = x

        for t in times:
            b_t = t * tf.ones([num_samples])

            # Langevin MCMC (Corrector)
            for _ in tf.range(num_mcmc_steps):
                # get gradient
                noise = tf.random.get_global_generator().normal(shape=shape,mean=0.,stddev=1.) #[N, x_dim]
                score = sde.score(x,c,b_t) #[N, x_dim]

                # get step_size
                if batch_independent_step_size:
                    grad_norm = tf.linalg.norm(score,axis=-1,keepdims=True)
                    noise_norm = tf.linalg.norm(noise,axis=-1,keepdims=True)
                else:
                    grad_norm = tf.reduce_mean(tf.linalg.norm(score,axis=-1))
                    noise_norm = tf.reduce_mean(tf.linalg.norm(noise,axis=-1))

                step_size = 2 * (gradient_snr * noise_norm / grad_norm) ** 2

                # do MCMC
                x = x + step_size * score + (2 * step_size)**0.5 * noise

            # Euler-Maruyama (Predictor)
            drift, diffusion = sde.reverse_sde(x,c,b_t)
            x_mean = x - drift * delta_t

            z = tf.random.get_global_generator().normal(shape=shape,mean=0.,stddev=1.) #dw approximation
            x = x_mean + delta_t**0.5 * (tf.expand_dims(diffusion,axis=-1) * z)

            #tf.debugging.assert_all_finite(x, f'{step_size}, {grad_norm}, {noise_norm}, {tf.math.is_finite(diffusion)}, {tf.math.is_finite(drift)}')

        return sde.pp.to_a(x_mean)

    return predictor_corrector

@gin.configurable
def build_ode_sampler(
    sde,
    ### gin configurables
    atol=1e-5,
    rtol=1e-5,
    eps_t=1e-3, # smallest time t; for numerical stability
    method='RK45'
):
    def ode_sampler(
        s,
    ):
        """
        Sovle the ODE using blackbox ODE solver
            specifically, solving initial condition problem!
        """
        num_samples = len(s)
        c = sde.pp.to_x(s)

        shape = [num_samples,sde.x_dim]

        prior = sde.prior_dist()
        init_x = prior.sample(num_samples) #[N, x_dim]

        def _ode_func(t,x):
            """
            input:
                t: scalar
                x: [num_samples*x_dim]
            output:
                dx: [num_samples*x_dim]
            """
            b_t = tf.ones([num_samples]) * t
            x = tf.cast(x.reshape(shape),tf.float32)

            drift, _ = sde.reverse_ode(x, c, b_t)

            return drift.numpy().ravel()

        ret = integrate.solve_ivp(
            _ode_func,
            (sde.T,eps_t),
            init_x.numpy().ravel(),
            rtol=rtol,atol=atol,method=method
        )

        last_x = ret.y[:,-1].reshape(shape)
        return sde.pp.to_a(last_x)

    return ode_sampler

@gin.configurable
def build_log_likelihood(
    sde,
    ### gin configurables
    num_random_projections=1,
    hutchinson_type='Rademacher', # way to predict the divergence using projection
    atol=1e-5,
    rtol=1e-5,
    eps_t=1e-5, # smallest time t; for numerical stability
    method='RK45',
):
    """
    ODE transforms one distribution to another (here, p_0 <-> p_1), i.e. there is one-to-one mapping (although, mapping is quite complex) exist.
    Therefore, we can calculate the likelihood using change-of-variable formula.

    For ODEs of the form:
        dx = h(x,t) dt
    There is an "instanteous" change-of-variable formula (that means, we can write the change-of-variable formula using only p_0 and p_1, abbreviating all the intermediate steps, I guess):
        p_0(x_0) = exp(\int_0^1 div h(x(t),t) dt) p_1(x_1), or,
        log p_0(x_0) = log p_1(x_1) + \int_0^1 div h(x(t),t) dt
    
    In SDE we are dealing with, h(x,t) is:
        h(x,t) = [f(x,t) - 0.5 * g^2(t) score(x,t)]; drift
    Therefore, all we need to calculate is the integral part since log p_1(x_1) is known.
        
    And, here, divergence (:= trace-sum of the Jacobian of score function) can be efficiently approximated by Skilling-Hutchinson estimattor. The idea is, to randomly projet the Jacobian function into lower-space.
        div[h(x(t),t)]
            = \E_\eps~N(0,I) [\eps^t \nabla_x h(x(t),t) \eps]
    This is efficient since it allow us to skip the Jacobian calculation.

    Now, we can combine it together via ODE solver;
        we now solve the forward ODE; integrate 0 to 1.
        we infer both x(t) and div[h(x(t),t)] at the same time; find x(1) from x(0) and its associated divergence.
    """
    rp_shape = [num_random_projections, sde.x_dim]
    if hutchinson_type == 'Gaussian':
        rp = tf.random.get_global_generator().normal(shape=rp_shape,mean=0.,stddev=1.) #random projection
    elif hutchinson_type == 'Rademacher':
        rp = tf.cast(tf.random.get_global_generator().uniform(shape=rp_shape,minval=0,maxval=2,dtype=tf.int32),tf.float32) * 2 - 1
    else:
        assert False
    
    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None,sde.x_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None,sde.c_dim],dtype=tf.float32),
            tf.TensorSpec(shape=[None],dtype=tf.float32),
        ])
    def _divergence(x,c,t):
        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape:
            tape.watch(x)

            drift, _ = sde.reverse_ode(x,c,t) #[B, x_dim]
            projected = tf.reduce_sum(tf.expand_dims(drift,axis=1) * rp[None],axis=-1) #[B,1,x_dim], [1,#rp,x_dim] -> [B,#rp]
            projected = tf.split(projected,num_random_projections,axis=1)

        gradients = tf.stack([tape.gradient(proj,x) for proj in projected],axis=1) #[B,#rp,x_dim]
        del tape

        div = tf.reduce_mean(tf.reduce_sum(gradients*rp[None],axis=-1),axis=1) #[B]
        return drift, div

    def log_likelihood(
        s,a
    ):
        x = sde.pp.to_y(a).numpy()
        c = sde.pp.to_x(s).numpy()
        num_items = len(s)

        shape = [num_items,sde.x_dim]

        def _ode_func(t,x):
            """
            input:
                t: scalar
                x(t) and log_p(t): [num_items*x_dim + num_items]
            output:
                dx, d_log_p: [num_items*x_dim + num_items];
            """
            assert np.all(np.isfinite(x)), 'Non-finite value detected'

            x = tf.cast(x[:np.prod(shape)].reshape(shape),dtype=tf.float32)
            b_t = tf.ones([num_items]) * t

            drift, div = _divergence(x,c,b_t)
            return np.concatenate([drift.numpy().ravel(), div.numpy().ravel()])

        ret = integrate.solve_ivp(
            _ode_func,
            (eps_t,sde.T),
            np.concatenate([x.ravel(),np.zeros([num_items]).astype(np.float32)]),
            rtol=rtol,atol=atol,method=method
        )

        x_1, integral = ret.y[:,-1][:np.prod(shape)].reshape(shape), ret.y[:,-1][np.prod(shape):]

        prior_dist = sde.prior_dist()
        log_p1_x1 = prior_dist.log_prob(x_1).numpy()

        log_p0_x0 = log_p1_x1 + integral
        return log_p0_x0
    return log_likelihood