import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import *


_operations = {"max": lambda s, f: tf.maximum(s, f),
               "keep": lambda s, f: s,
               "replace": lambda s, f: f,
               "mul": lambda s, f: tf.mul(s, f),
               "min": lambda s, f: tf.minimum(s, f),
               "diff": lambda s, f: 0.5 * tf.abs(s - f),
               "forget":lambda s, f: tf.zeros_like(s),
               "sqr_diff": lambda s, f: 0.25 * (s - f)**2}


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


def complexify(v):
    v_r, v_i = tf.split(1,2,v)
    return tf.complex(v_r, v_i)


def bound(v):
    v_sqr = v * v
    v1, v2 = tf.split(1, 2, v_sqr)
    v_sqrt = tf.concat(1, [tf.sqrt(v1 + v2)])
    v_sqrt = tf.concat(1, [v_sqrt, v_sqrt])
    return v/v_sqrt

