import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import *
import random

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
        weights = [tf.reshape(weights[i]/acc, [-1, self._num_units], name="op_weight_%d"%i) for i in xrange(len(weights))]
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

    def __init__(self, num_units, num_copies=1, input_size=None, read_only=False):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._num_copies = num_copies
        self._read_only = read_only
        self._permutations = [list(xrange(0, num_units/2)) for _ in xrange(self._num_copies)]
        for perm in self._permutations:
            random.shuffle(perm)

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
            with vs.variable_scope("Permutations"):
                perms = reduce(lambda x, y: x+y, self._permutations)
                perms = tf.constant(perms)

            old_key = tf.slice(state, [0, 0],[-1, self.output_size])
            old_ss = tf.slice(state, [0, self.output_size], [-1,-1])
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs, old_key], self._num_units, True)))
                k = tf.transpose(tf.concat(0, [tf.real(key), tf.imag(key)]))
                k_real, k_imag = tf.split(0, 2, tf.transpose(tf.nn.embedding_lookup(k, perms)))
                ks = tf.complex(k_real, k_imag)

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(tf.conj(ks), c_ss))

            if not self._read_only:
                with vs.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    r, u = array_ops.split(1, 2, linear([inputs, h],
                                                         2 * self._num_units, True, 1.0))
                    r, u = sigmoid(r), sigmoid(u)
                with vs.variable_scope("Candidate"):
                    c = tanh(linear([inputs, r * h], self._num_units, True))

                to_add = u * (c - h)
                to_add_r, to_add_i = tf.split(1, 2, to_add)
                c_to_add = tf.complex(tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
                new_ss = uncomplexify(c_ss + ks * c_to_add)
                new_h = h + to_add
            else:
                new_h = h
                new_ss = old_ss

        return new_h, tf.concat(1, [uncomplexify(key, "key"), new_ss])

    def _read(self, keys, redundant_states):
        read = keys * redundant_states
        if self._num_copies > 1:
            reads = tf.split(1, self._num_copies, read)
            read = reads[0]
            for i in xrange(1, self._num_copies):
                read += reads[i]
            read /= self._num_copies
        return read


class ControlledAssociativeGRUCell(AssociativeGRUCell):
    @property
    def state_size(self):
        return self._num_units * (self._num_copies+2)

    def __call__(self, inputs, state, scope=None):
        h = tf.slice(state, [0, 0], [-1, self._num_units])
        s = tf.slice(state, [0, self._num_units], [-1, -1])
        assoc_h, assoc_s = AssociativeGRUCell.__call__(self, inputs, s, scope)

        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Controller"):
                with vs.variable_scope("Gates"):
                    u, c = tf.split(1, 2, linear([assoc_h, inputs, h], 2 * self._num_units, True))
                    u, c = sigmoid(u), tanh(c)
                new_h = u * h + (1 - u) * c

        return new_h, tf.concat(1, [new_h, assoc_s])



def complexify(v, name=None):
    v_r, v_i = tf.split(1, 2, v)
    return tf.complex(v_r, v_i, name)


def uncomplexify(v, name=None):
    return tf.concat(1, [tf.real(v), tf.imag(v)], name)


def bound(v, name=None):
    im_v = tf.imag(v)
    re_v = tf.real(v)
    v_sqrt = tf.maximum(1.0, tf.sqrt(im_v * im_v + re_v * re_v))
    return tf.complex(re_v / v_sqrt, im_v / v_sqrt, name)

