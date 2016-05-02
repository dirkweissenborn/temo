import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import *
import random
import functools

_operations = {"max": lambda s, v: tf.maximum(s, v),
               "keep": lambda s, v: s,
               "replace": lambda s, v: v,
               "mul": lambda s, v: tf.mul(s, v),
               "min": lambda s, v: tf.minimum(s, v),
               "diff": lambda s, v: 0.5 * tf.abs(s - v),
               "forget": lambda s, v: tf.zeros_like(s),
               "sqr_diff": lambda s, v: 0.25 * (s - v)**2}


class MORUCell(RNNCell):

    def __init__(self, num_units, input_size=None,
                 op_controller_size=None,
                 ops=(_operations["keep"], _operations["replace"], _operations["mul"]),
                 op_biases=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._op_controller_size = 0 if op_controller_size is None else op_controller_size
        self._op_biases = list(op_biases)
        self._ops = ops if ops is not None else list(map(lambda _: 0.0, ops))
        self._num_ops = len(ops)

    @staticmethod
    def from_op_names(operations, biases, num_units, input_size=None, op_controller_size=None):
        if biases is None:
            biases = map(lambda _: 0.0, operations)
        assert len(list(biases)) == len(operations), "Operations and operation biases have to have same length."
        ops = list(map(lambda op: _operations[op], operations))
        return MORUCell(num_units, input_size, op_controller_size, ops, biases)

    def _op_weights(self, inputs):
        t = tf.reshape(linear(inputs, self._num_units * self._num_ops, True), [-1, self._num_units, self._num_ops])
        weights = tf.split(2, self._num_ops, t)
        #op_sharpening = tf.get_variable("gamma", (), tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)
        for i, w in enumerate(weights):
            if self._op_biases and self._op_biases[i] != 0.0:
                weights[i] = tf.exp((w + self._op_biases[i]))
            else:
                weights[i] = tf.exp(w)
        #tf.softmax is incredibly slow, as well as reduce_sum
        acc = weights[0]
        for w in weights[1:]:
            acc += w
        weights = [tf.reshape(weights[i]/acc, [-1, self._num_units], name="op_weight_%d"%i) for i in range(len(weights))]
        return weights

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units + max(self._op_controller_size, 0)

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            s, op_ctr = None, None
            if self._op_controller_size > 0:
                s = tf.slice(state, [0, 0], [-1, self._num_units])
                op_ctr = tf.slice(state, [0, self._num_units], [-1, self._op_controller_size])
            else:
                s = state
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r = linear([inputs, s], self._num_units, True, 1.0)
                r = sigmoid(r)
            with vs.variable_scope("Feature"):
                f = tanh(linear([inputs, r * s], self._num_units, True))
            new_op_ctr = None
            if self._op_controller_size > 0:
                with vs.variable_scope("Op_controller"):
                    new_op_ctr = [tanh(linear([inputs, s, op_ctr], self._op_controller_size, True))]
            else:
                new_op_ctr = [inputs, s]
            with vs.variable_scope("Op"):
                op_weights = self._op_weights(new_op_ctr)
                new_cs = [o(s, f) * w for (o, w) in zip(self._ops, op_weights)]
                new_c = tf.reduce_sum(tf.pack(new_cs), [0])
        if self._op_controller_size > 0:
            return new_c, tf.concat(1, [new_c, new_op_ctr])
        else:
            return new_c, new_c


class AssociativeGRUCell(RNNCell):

    def __init__(self, num_units, num_copies=1, input_size=None, num_read_keys=0, read_only=False, rng=None):
        if rng is None:
            rng = random.Random(123)
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._num_copies = num_copies
        self._num_read_keys = num_read_keys
        self._read_only = read_only
        self._permutations = [list(range(0, int(num_units/2))) for _ in range(self._num_copies-1)]
        for perm in self._permutations:
            rng.shuffle(perm)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units * (self._num_copies+1)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_h = tf.slice(state, [0, 0], [-1, self._num_units])
            old_ss = tf.slice(state, [0, self._num_units], [-1,-1])
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs, old_h], (1+self._num_read_keys)*self._num_units, False)))
                w_k_real, w_k_imag = _comp_real(key), _comp_imag(key)
                r_keys = []
                if self._num_copies > 1:
                    k = tf.transpose(tf.concat(0, [_comp_real(key), _comp_imag(key)]), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    w_k_real, w_k_imag = tf.split(0, 2, tf.transpose(k, [1, 0]))
                    if self._num_read_keys > 0:
                        k_real = tf.split(1, 1+self._num_read_keys, w_k_real)
                        k_imag = tf.split(1, 1+self._num_read_keys, w_k_imag)
                        w_k_real = k_real[0]
                        w_k_imag = k_imag[0]
                        r_k_real = k_real[1:]
                        r_k_imag = k_imag[1:]
                        r_keys = list(zip(r_k_real, r_k_imag))
                w_key = (w_k_real, w_k_imag)
                conj_w_key = _comp_conj(w_key)

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(conj_w_key, c_ss), "retrieved")
                r_hs = []
                for i, k in enumerate(r_keys):
                    r_hs.append(uncomplexify(self._read(k, c_ss), "read_%d" % i))

            if not self._read_only:
                with vs.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    r, u = array_ops.split(1, 2, linear(r_hs + [inputs, h],
                                                         2 * self._num_units, True, 1.0))
                    r, u = sigmoid(r), sigmoid(u)
                with vs.variable_scope("Candidate"):
                    c = tanh(linear(r_hs + [inputs, r * h], self._num_units, True))

                to_add = u * (c - h)
                to_add_r, to_add_i = tf.split(1, 2, to_add)
                c_to_add = (tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
                new_ss = old_ss + uncomplexify(_comp_mul(w_key, c_to_add))
                new_h = tf.add(h, to_add, "out")
            else:
                new_h = h
                new_ss = old_ss

        return new_h, tf.concat(1, [new_h, new_ss])

    def _read(self, keys, redundant_states):
        read = _comp_mul(keys, redundant_states)
        if self._num_copies > 1:
            xs_real = tf.split(1, self._num_copies, _comp_real(read))
            xs_imag = tf.split(1, self._num_copies, _comp_imag(read))
            read = (tf.add_n(xs_real)/self._num_copies, tf.add_n(xs_imag)/self._num_copies)
        return read


class DualAssociativeGRUCell(AssociativeGRUCell):

    def __init__(self, num_units, num_copies=1, input_size=None, num_read_keys=0, share=False, rng=None):
        AssociativeGRUCell.__init__(self, num_units, num_copies=num_copies, input_size=input_size,
                                    num_read_keys=num_read_keys, read_only=False, rng=rng)
        self._share = share

    @property
    def state_size(self):
        return self._num_units * (self._num_copies*2+1)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_h = tf.slice(state, [0, 0], [-1, self._num_units])
            old_ss = tf.slice(state, [0, self._num_units], [-1, self._num_units * self._num_copies])
            read_mem = tf.slice(state, [0, self._num_units * (self._num_copies+1)], [-1, -1])
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs, old_h], (1+self._num_read_keys)*self._num_units, False)))
                w_k_real, w_k_imag = _comp_real(key), _comp_imag(key)
                r_keys = []
                if self._num_copies > 1:
                    k = tf.transpose(tf.concat(0, [_comp_real(key), _comp_imag(key)]), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    w_k_real, w_k_imag = tf.split(0, 2, tf.transpose(k, [1, 0]))
                    if self._num_read_keys > 0:
                        k_real = tf.split(1, 1+self._num_read_keys, w_k_real)
                        k_imag = tf.split(1, 1+self._num_read_keys, w_k_imag)
                        w_k_real = k_real[0]
                        w_k_imag = k_imag[0]
                        r_k_real = k_real[1:]
                        r_k_imag = k_imag[1:]
                        r_keys = list(zip(r_k_real, r_k_imag))
                w_key = (w_k_real, w_k_imag)
                conj_w_key = _comp_conj(w_key)

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(conj_w_key, c_ss), "retrieved")
                r_hs = []
                for i, k in enumerate(r_keys):
                    r_hs.append(uncomplexify(self._read(k, c_ss), "read_%d" % i))

            with vs.variable_scope("Read_Given"):
                h2 = uncomplexify(self._read(conj_w_key, complexify(read_mem)), "retrieved")

            with vs.variable_scope("Gates"):
                if self._share:
                    tf.get_variable_scope().reuse_variables()
                gs = linear(r_hs + [inputs, h], 2 * self._num_units, True, 1.0)
            with vs.variable_scope("DualGates"):
                gs = sigmoid(gs + linear([h2], 2 * self._num_units, False))
            r, u = tf.split(1, 2, gs)

            with vs.variable_scope("Candidate"):
                if self._share:
                    tf.get_variable_scope().reuse_variables()
                c = linear(r_hs + [inputs, r * h], self._num_units, True)

            with vs.variable_scope("DualCandidate"):
                c = tanh(c + linear([h2], self._num_units, False))

            to_add = u * (c - h)
            to_add_r, to_add_i = tf.split(1, 2, to_add)
            c_to_add = (tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
            new_ss = old_ss + uncomplexify(_comp_mul(w_key, c_to_add))
            new_h = tf.add(h, to_add, "out")

        return new_h, tf.concat(1, [new_h, new_ss, read_mem])


def complexify(v, name=None):
    v_r, v_i = tf.split(1, 2, v)
    return (v_r, v_i)


def uncomplexify(v, name=None):
    return tf.concat(1, v, name)


def bound(v, name=None):
    im_v = _comp_imag(v)
    re_v = _comp_real(v)
    v_modulus = tf.maximum(1.0, tf.sqrt(im_v * im_v + re_v * re_v))
    return (re_v / v_modulus, im_v / v_modulus, name)


# Much faster than using native tensorflow complex datastructure
def _comp_conj(x):
    return (_comp_real(x), -_comp_imag(x))

def _comp_add(x, y):
    return (_comp_real(x)+_comp_real(y), _comp_imag(x)+_comp_imag(y))

def _comp_add_n(xs):
    xs_real = [_comp_real(x) for x in xs]
    xs_imag = [_comp_imag(x) for x in xs]
    return (tf.add_n(xs_real), tf.add_n(xs_imag))

def _comp_mul(x, y):
    return (_comp_real(x) * _comp_real(y) - _comp_imag(x) * _comp_imag(y),
            _comp_real(x) * _comp_imag(y) + _comp_imag(x) * _comp_real(y))

def _comp_real(x):
    return x[0]

def _comp_imag(x):
    return x[1]


class RectifierRNNCell(BasicRNNCell):
    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            output = tf.maximum(0, linear([inputs, state], self._num_units, True))
        return output, output