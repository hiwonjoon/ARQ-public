from collections.abc import Iterable
import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Wrapper, TimeDistributed

@gin.configurable
class SpectralNormalization(Wrapper):
    # Don't use the tf.addons.Spectral Normalization
    # https://github.com/tensorflow/addons/issues/2414

    def __init__(self, layer, power_iterations=1, eps=1e-12, **kwargs):
        assert power_iterations > 0

        super().__init__(layer,**kwargs)

        self.power_iterations = power_iterations
        self.eps = eps

    def build(self, input_shape):
        super().build(input_shape)

        self.W = self.layer.kernel
        self.out_dim = self.W.shape.as_list()[-1]

        self.u = self.add_weight(
            name='u',
            shape=(1, self.out_dim),
            initializer=tf.initializers.RandomNormal(stddev=1.0),
            trainable=False,
            dtype=self.layer.kernel.dtype,
        )

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        W = tf.reshape(self.W,[-1,self.out_dim])

        for _ in range(self.power_iterations): # do not use tf.range here. Better to unroll it since power_iterations usually set very small numbers.
            v = tf.math.l2_normalize(tf.matmul(self.u,W,transpose_b=True),epsilon=self.eps) #[1,in_dim]
            u = tf.math.l2_normalize(tf.matmul(v,W),epsilon=self.eps) #[1,out_dim]
        
        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        sigma = tf.matmul(tf.matmul(v, W),u,transpose_b=True)

        if training:
            self.u.assign(u)

        ## Version 1
        #self.layer.kernel = self.W / sigma
        #output = self.layer.call(inputs)
        #self.layer.kernel = self.W

        ## Version 2
        #W_original = tf.identity(self.W)
        #W_normalized = self.W / sigma

        #self.W.assign(W_normalized)
        #output = self.layer.call(inputs)
        #self.W.assign(W_original)

        ## Version 3
        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            output = tf.matmul(inputs, self.W/sigma)
        else:
            output = tf.tensordot(inputs, self.W/sigma, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.W.shape[-1]]
                output.set_shape(output_shape)

        if self.layer.use_bias:
            output = tf.nn.bias_add(output, self.layer.bias)
        
        if self.layer.activation is not None:
            output = self.layer.activation(output)

        return output
    
    def get_config(self):
        config = {"power_iterations": self.power_iterations, 'eps':self.eps}
        base_config = super().get_config()
        return {**base_config, **config}

@gin.configurable(module=__name__)
class MLP(Layer):
    def __init__(self,num_layers,dim,out_dim,activation='relu',name=None,in_dim=None,spectral_norm=False,use_bias=True,time_distributed=False,last_activation=None,last_spectral_norm=False,orthogonal_init=False):
        super().__init__()

        if isinstance(in_dim,Iterable):
            in_dim = sum(in_dim)

        if isinstance(out_dim,Iterable):
            out_dim = sum(out_dim)

        self.layers = []
        for l in range(num_layers):
            l = Dense(
                dim,
                activation = activation,
                name = None if name is None else f'{name}_{l}',
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=2**0.5) if orthogonal_init else 'glorot_uniform',
                bias_initializer='zeros',
            )

            if spectral_norm:
                l = SpectralNormalization(l)

            if in_dim is not None:
                l.build((in_dim,))
                in_dim = dim

            if time_distributed:
                l = TimeDistributed(l)

            self.layers.append(l)

        l = Dense(
            out_dim,
            activation = last_activation,
            name = None if name is None else f'{name}_{num_layers}',
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1e-2) if orthogonal_init else 'glorot_uniform',
            bias_initializer='zeros',
        )

        if spectral_norm and last_spectral_norm:
            l = SpectralNormalization(l)
        
        if in_dim is not None:
            l.build((in_dim,))

        if time_distributed:
            l = TimeDistributed(l)

        self.layers.append(l)

        self.in_dim = in_dim
        self.out_dim = out_dim

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for i,l in enumerate(self.layers):
            o = l(o,training=training)
        return o

    @tf.function
    def fv(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for i,l in enumerate(self.layers[:-1]):
            o = l(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in self.layers
            ]

@gin.configurable(module=__name__)
class MLPResNet(Layer):
    def __init__(self,in_dim,num_res_blocks,dim,out_dim,activation='relu',spectral_norm=False,use_bias=True,last_spectral_norm=False,last_activation=None):
        super().__init__()

        if isinstance(in_dim,Iterable): in_dim = sum(in_dim)
        self.in_dim = in_dim

        if isinstance(out_dim,Iterable): out_dim = sum(out_dim)
        self.out_dim = out_dim

        self.activation = activation

        # projection layer
        self._projection = Dense(
            dim,
            activation = None,
            use_bias = use_bias,
        )
        if spectral_norm: self._projection = SpectralNormalization(self._projection)
        self._projection.build((in_dim,))

        self._first = []
        self._second = []
        for _ in range(num_res_blocks):
            _l1 = Dense(
                dim,
                activation = None,
                use_bias=use_bias,
            )
            if spectral_norm: _l1 = SpectralNormalization(_l1)
            _l1.build((dim,))

            self._first.append(_l1)

            _l2 = Dense(
                dim,
                activation = None,
                use_bias=use_bias,
            )
            if spectral_norm: _l2 = SpectralNormalization(_l2)
            _l2.build((dim,))

            self._second.append(_l2)

        self._out = Dense(
            out_dim,
            activation = last_activation,
            use_bias = use_bias,
        )
        if spectral_norm and last_spectral_norm: self._out = SpectralNormalization(self._out)
        self._out.build((dim,))

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)

        o = self._projection(o,training=training)

        for l1,l2 in zip(self._first,self._second):
            o_init = tf.identity(o)
            if self.activation == 'relu':
                o = tf.nn.relu(o)
            else: assert False
            o = l1(o, training=training)

            if self.activation == 'relu':
                o = tf.nn.relu(o)
            else: assert False
            o = l2(o, training=training)
            o = o_init + o

        o = self._out(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        layers = [self._projection, self._out] + self._first + self._second

        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in layers
            ]

@gin.configurable
class Dropout(Wrapper):
    # apply dropout on its "input"
    # Always apply dropout since it is for MC
    def __init__(self, layer, drop_rate):
        super().__init__(layer)
        self.drop_rate = drop_rate

    @tf.function
    def call(self,inputs,training=None):
        x = tf.nn.dropout(inputs,self.drop_rate)
        return self.layer(x,training=training)

@gin.configurable
class MLPDropout(Layer):
    def __init__(self,Dropout,num_layers,dim,out_dim,activation='relu',name=None,in_dim=None,skip_first=False):
        super().__init__()

        self.layers = []
        for l in range(num_layers):
            l = Dense(
                dim,
                activation = activation,
                name = None if name is None else f'{name}_{l}',
            )
            if in_dim is not None:
                l.build((in_dim,))
                in_dim = dim

            if skip_first:
                self.layers.append(l)
            else:
                self.layers.append(Dropout(l))

        l = Dense(
            out_dim,
            name = None if name is None else f'{name}_{num_layers}',
        )
        if in_dim is not None:
            l.build((in_dim,))

        self.layers.append(Dropout(l))

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in self.layers
            ]
