import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import *


class MORUCell(RNNCell):

    def __init__(self, num_units, input_size=None,
                 op_controller_size=None,
                 op_biases=(3.0, 1.0, 0, 0, 0),
                 ops=(lambda s, f: tf.maximum(s, f),
                      lambda s, f: s,
                      lambda s, f: f,
                      lambda s, f: tf.mul(s, f)
                      #lambda s, f: 0.5 * tf.abs(s - f),
                      #lambda s, f: tf.minimum(s, f),
                      )):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._op_controller_size = max(10, int(num_units / 5)) if op_controller_size is None else op_controller_size
        self._op_biases = op_biases
        self._ops = ops

    def _op_weights(self, inputs):
        t = tf.reshape(linear(inputs, self._num_units * (len(self._ops)), True), [-1, self._num_units, len(self._ops)])
        weights = tf.split(2, len(self._ops), t)

        for i,w in enumerate(weights):
            if self._op_biases and self._op_biases[i] != 0.0:
                weights[i] = tf.exp(w + self._op_biases[i])
            else:
                weights[i] = tf.exp(w)
        #tf.softmax is incredibly slow, as well as reduce_sum
        acc = weights[0]
        for w in weights[1:]:
            acc += w
        weights = map(lambda w: tf.reshape(w/acc, [-1, self._num_units]), weights)
        return weights

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units + self._op_controller_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            s = tf.slice(state, [0, 0], [-1, self._num_units])
            op_ctr = tf.slice(state, [0, self._num_units],[-1, self._op_controller_size])
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r = linear([inputs, s], self._num_units, True, 1.0)
                r = sigmoid(r)
            with vs.variable_scope("Feature"):
                f = tanh(linear([inputs, r * s], self._num_units, True))
            with vs.variable_scope("Op_controller"):
                new_op_ctr = tanh(linear([inputs, r * s, op_ctr], self._op_controller_size, True))
            with vs.variable_scope("Op"):
                op_weights = self._op_weights([new_op_ctr])
                new_cs = map(lambda (o, w): o(s, f) * w, zip(self._ops, op_weights))
                new_c = tf.reduce_sum(tf.pack(new_cs), [0])

        return new_c, tf.concat(1, [new_c, new_op_ctr])