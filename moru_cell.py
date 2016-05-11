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
        t = linear(inputs, self._num_units * self._num_ops, True)
        weights = tf.split(1, self._num_ops, t)
        for i, w in enumerate(weights):
            if self._op_biases and self._op_biases[i] != 0.0:
                weights[i] = tf.exp((w + self._op_biases[i]))
            else:
                weights[i] = tf.exp(w)
        acc = tf.add_n(weights)
        weights = [tf.div(weights[i], acc, name="op_weight_%d" % i) for i in range(len(weights))]
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
                    new_op_ctr = [linear([inputs, s, op_ctr], self._op_controller_size, True)]
            else:
                new_op_ctr = [inputs, s]
            with vs.variable_scope("Op"):
                op_weights = self._op_weights(new_op_ctr)
                new_cs = [o(s, f) * w for (o, w) in zip(self._ops, op_weights)]
                new_c = tf.add_n(new_cs)
        if self._op_controller_size > 0:
            return new_c, tf.concat(1, [new_c] + new_op_ctr)
        else:
            return new_c, new_c
