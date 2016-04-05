import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import *
import numpy as np

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
        self._op_biases = op_biases
        self._ops = ops if ops is not None else map(lambda _: 0.0, ops)

    @staticmethod
    def from_op_names(operations, biases, num_units, input_size=None, op_controller_size=None):
        if biases is None:
            biases = map(lambda _: 0.0, operations)
        assert len(biases) == len(operations), "Operations and operation biases have to have same length."
        ops = map(lambda op: _operations[op], operations)
        return MORUCell(num_units, input_size, op_controller_size, ops, biases)

    def _op_weights(self, inputs):
        t = tf.reshape(linear(inputs, self._num_units * (len(self._ops)), True), [-1, self._num_units, len(self._ops)])
        weights = tf.split(2, len(self._ops), t)
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
        weights = map(lambda i: tf.reshape(weights[i]/acc, [-1, self._num_units], name="op_weight_%d"%i), xrange(len(weights)))
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
                new_cs = map(lambda (o, w): o(s, f) * w, zip(self._ops, op_weights))
                new_c = tf.reduce_sum(tf.pack(new_cs), [0])
        if self._op_controller_size > 0:
            return new_c, tf.concat(1, [new_c, new_op_ctr])
        else:
            return new_c, new_c


class AssociativeMORUCell(RNNCell):

    def __init__(self, num_units, input_size=None,
                 op_controller_size=None,
                 ops=(_operations["keep"], _operations["replace"], _operations["mul"]),
                 op_biases=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._op_controller_size = 0 if op_controller_size is None else op_controller_size
        self._op_biases = op_biases
        self._ops = ops if ops is not None else map(lambda _: 0.0, ops)
        self._num_reads = 3

    @staticmethod
    def from_op_names(operations, biases, num_units, input_size=None, op_controller_size=None):
        if biases is None:
            biases = map(lambda _: 0.0, operations)
        assert len(biases) == len(operations), "Operations and operation biases have to have same length."
        ops = map(lambda op: _operations[op], operations)
        return MORUCell(num_units, input_size, op_controller_size, ops, biases)

    def _op_weights(self, inputs):
        t = tf.reshape(linear(inputs, self._num_units * (len(self._ops)), True), [-1, self._num_units, len(self._ops)])
        weights = tf.split(2, len(self._ops), t)
        for i, w in enumerate(weights):
            if self._op_biases and self._op_biases[i] != 0.0:
                weights[i] = tf.exp((w + self._op_biases[i]))
            else:
                weights[i] = tf.exp(w)
        acc = weights[0]
        for w in weights[1:]:
            acc += w
        weights = map(lambda i: tf.reshape(weights[i]/acc, [-1, self._num_units], name="op_weight_%d"%i), xrange(len(weights)))
        return weights

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            s = complexify(state)
            with vs.variable_scope("key"):
                print("Associative MORU")
                key = bound(linear([inputs, s], self._num_units, True))
                old_f = key * s
            with vs.variable_scope("Gates"):
                r = sigmoid(linear([inputs, old_f], self._num_units, True, 1.0))
            with vs.variable_scope("Feature"):
                f = complexify(bound(linear([inputs, r * old_f], self._num_units, True)))

            with vs.variable_scope("Op"):
                op_weights = self._op_weights([f, old_f])
                new_fs = map(lambda (o, w): o(old_f, f) * w, zip(self._ops, op_weights))
                new_f = tf.reduce_sum(tf.pack(new_fs), [0])

            s = s - old_f + new_f
            s = tf.concat(1, [tf.real(s), tf.imag(s)])
        return s, s


class AssociativeGRUCell(RNNCell):

    def __init__(self, num_units, num_copies=1, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._num_copies = num_copies
        self._permutations = [np.random.permutation(xrange(0, num_units)) for _ in xrange(self._num_copies)]
        # permutations with transpose -> gather -> transpose

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
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with vs.variable_scope("Permutations"):
                perms = map(lambda perm: tf.constant(perm), self._permutations)

            split = tf.split(1, 1+self._num_copies, state)
            h = split[0]
            ss = [complexify(s) for s in split[1:]]
            with vs.variable_scope("Keys"):
                k_tr = tf.transpose(bound(linear([inputs, h], self._num_units, True)))
                #k_tr_r, k_tr_w = tf.split(1, 2, k_tr)
                #k_rs = []
                ks = []
                for perm in perms:
                    #k_rs.append(complexify(tf.transpose(tf.gather(k_tr_r, perm))))
                    ks.append(complexify(tf.transpose(tf.gather(k_tr, perm))))

            with vs.variable_scope("Read"):
                old_f = self._read(map(tf.conj, ks), ss)

            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, linear([inputs, old_f],
                                                     2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = bound(linear([inputs, r * old_f, h], self._num_units, True))

            to_add = u * (c - old_f)
            c_to_add = complexify(to_add)
            new_ss = [uncomplexify(s + k_w * c_to_add) for k_w, s in zip(ks, ss)]
            new_h = old_f + to_add

        return new_h, tf.concat(1, [new_h] + new_ss)

    def _read(self, keys, redundant_states):
        read = uncomplexify(keys[0] * redundant_states[0])
        for i in xrange(1, self._num_copies):
            read = read + uncomplexify(tf.conj(keys[i]) * redundant_states[i])
        if self._num_copies > 1:
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
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"



            with vs.variable_scope("Controller"):  # "GRUCell"
                with vs.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    r, u = array_ops.split(1, 2, linear([assoc_h, h],
                                                        2 * self._num_units, True, 1.0))
                    r, u = sigmoid(r), sigmoid(u)
                with vs.variable_scope("Candidate"):
                    c = tanh(linear([assoc_h, r * h], self._num_units, True))
                new_h = u * h + (1 - u) * c
        return new_h, tf.concat(1, [new_h, assoc_s])



def complexify(v):
    v_r, v_i = tf.split(1, 2, v)
    return tf.complex(v_r, v_i)

def uncomplexify(v):
    return tf.concat(1, [tf.real(v), tf.imag(v)])


def bound(v):
    v_sqr = v * v
    v1, v2 = tf.split(1, 2, v_sqr)
    v_sqrt = tf.concat(1, [tf.sqrt(v1 + v2)])
    v_sqrt = tf.concat(1, [v_sqrt, v_sqrt])
    return v/v_sqrt

