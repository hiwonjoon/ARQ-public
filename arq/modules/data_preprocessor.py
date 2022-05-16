import numpy as np
import tensorflow as tf
import gin

from tensorflow.keras.models import Model

@gin.configurable
class EmptyPreprocessor(Model):
    def __init__(
        self,
        ob_dim,
        ac_dim,
        ac_scale,
    ):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.x_dims = [ob_dim]
        self.y_dims = [ac_dim]
        self.z_dims = [ob_dim + ac_dim]

        self.ac_scale = ac_scale # action limits

        def _create_weight(shape, init_val, name):
            return self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.constant_initializer(init_val * np.ones(shape,dtype=np.float32)),
                dtype=tf.float32,
                trainable=False
            )

        self.y_min = _create_weight([1,ac_dim],-float('inf'),'y_min')
        self.y_max = _create_weight([1,ac_dim],float('inf'),'y_max')
    
    @property
    def decay_vars(self):
        return []

    def prepare(
        self,
        D, # for input normalization. (iterator that has signature of (s,a,*))
    ):
        a_min, a_max = np.zeros(self.y_dims,np.float64), np.zeros(self.y_dims,np.float64)

        for s,a,*_ in D:
            s,a = s.numpy().astype(np.float64), a.numpy().astype(np.float64)

            a_min = np.minimum(a_min, np.amin(a,axis=0,keepdims=True))
            a_max = np.maximum(a_max, np.amax(a,axis=0,keepdims=True))

        a_min, a_max = a_min - 0.05 * (a_max - a_min), a_max + 0.05 * (a_max - a_min)
        
        self.y_min.assign(self.to_y(a_min))
        self.y_max.assign(self.to_y(a_max))

    def to_x(self,s):
        return tf.cast(s,tf.float32)

    def to_y(self,a):
        return tf.cast(a,tf.float32)

    def clip_y(self,y):
        return tf.clip_by_value(y, self.y_min, self.y_max)

    def to_s(self,x):
        return x

    def to_a(self,y):
        return y

    def to_z(self,x,y):
        return tf.concat([x,y],axis=-1)

@gin.configurable
class Preprocessor(Model):
    def __init__(
        self,
        ob_dim,
        ac_dim,
        ac_scale,
        clip_std = None,
    ):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.x_dims = [ob_dim]
        self.y_dims = [ac_dim]
        self.z_dims = [ob_dim + ac_dim]

        self.ac_scale = ac_scale # action limits

        def _create_weight(shape, init_val, name):
            return self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.constant_initializer(init_val * np.ones(shape,dtype=np.float32)),
                dtype=tf.float32,
                trainable=False
            )

        self.clip_std = clip_std

        self.s_mu = _create_weight([1,ob_dim],0.,'s_mu')
        self.s_std = _create_weight([1,ob_dim],1.,'s_std')
        self.a_mu = _create_weight([1,ac_dim],0.,'a_mu')
        self.a_std = _create_weight([1,ac_dim],1.,'a_std')
        
        self.y_min = _create_weight([1,ac_dim],-1.,'y_min')
        self.y_max = _create_weight([1,ac_dim],1.,'y_max')

    @property
    def decay_vars(self):
        return []

    @gin.configurable(module=f'{__name__}.Preprocessor')
    def prepare(
        self,
        D, # for input normalization. (iterator that has signature of (s,a,*))
        ######### Gin Configurable
        ddof = 0,
        a_margin = 0.05
    ):
        # Welford's algorithm (https://stackoverflow.com/a/5544108,https://stackoverflow.com/a/56407442)
        n = 0

        s_mu = np.zeros(self.s_mu.shape,np.float64)
        s_M2 = np.zeros(self.s_std.shape,np.float64)

        a_mu = np.zeros(self.a_mu.shape,np.float64)
        a_M2 = np.zeros(self.a_std.shape,np.float64)
        a_min = np.zeros(self.a_mu.shape,np.float64)
        a_max = np.zeros(self.a_mu.shape,np.float64)

        for s,a,*_ in D:
            s,a = s.numpy().astype(np.float64), a.numpy().astype(np.float64)

            n += len(s)

            s_delta = s - s_mu
            s_mu += np.sum(s_delta,axis=0,keepdims=True) / n
            s_M2 += np.sum(s_delta * (s - s_mu),axis=0,keepdims=True)

            a_delta = a - a_mu
            a_mu += np.sum(a_delta,axis=0,keepdims=True) / n
            a_M2 += np.sum(a_delta * (a - a_mu),axis=0,keepdims=True)

            a_min = np.minimum(a_min, np.amin(a,axis=0,keepdims=True))
            a_max = np.maximum(a_max, np.amax(a,axis=0,keepdims=True))
        
        s_std = (s_M2 / (n - ddof))**0.5
        a_std = (a_M2 / (n - ddof))**0.5

        if self.clip_std is not None:
            s_std[np.logical_and(s_std < self.clip_std, s_std > 0)] = self.clip_std
            a_std[np.logical_and(a_std < self.clip_std, a_std > 0)] = self.clip_std

        a_min, a_max = a_min - a_margin * (a_max - a_min), a_max + a_margin * (a_max - a_min)
        a_min = np.clip(a_min, -self.ac_scale, self.ac_scale)
        a_max = np.clip(a_max, -self.ac_scale, self.ac_scale)

        self.s_mu.assign(s_mu)
        self.s_std.assign(s_std)

        self.a_mu.assign(a_mu)
        self.a_std.assign(a_std)

        self.y_min.assign(self.to_y(a_min))
        self.y_max.assign(self.to_y(a_max))

        print(f'{self.s_mu.numpy()=}')
        print(f'{self.s_std.numpy()=}')
        print(f'{self.a_mu.numpy()=}')
        print(f'{self.a_std.numpy()=}')
        print(f'{self.y_min.numpy()=}')
        print(f'{self.y_max.numpy()=}')

    def to_x(self,s):
        return tf.math.divide_no_nan((tf.cast(s,tf.float32) - self.s_mu),self.s_std)

    def to_y(self,a):
        return tf.math.divide_no_nan((tf.cast(a,tf.float32) - self.a_mu),self.a_std)

    def clip_y(self,y):
        return tf.clip_by_value(tf.cast(y,tf.float32), self.y_min, self.y_max)

    def to_s(self,x):
        return (x * self.s_std) + self.s_mu

    def to_a(self,y):
        return (y * self.a_std) + self.a_mu

    def to_z(self,x,y):
        return tf.concat([x,y],axis=-1)

    def to_xy(self,z):
        return tf.split(z,self.x_dims+self.y_dims,axis=-1)