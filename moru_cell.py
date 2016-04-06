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


class AssociativeGRUCell(RNNCell):

    def __init__(self, num_units, num_copies=1, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._num_copies = num_copies
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
        return self._num_units * (self._num_copies+2)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Permutations"):
                perms = reduce(lambda x,y: x+y, self._permutations)
                perms = tf.constant(perms)

            split = tf.split(1, 2+self._num_copies, state)
            h = tf.slice(state, [0,0],[-1,self.output_size])
            old_key = tf.slice(state, [0,self.output_size],[-1,self.output_size]) 
            ss = complexify(tf.slice(state, [0,self.output_size*2], [-1,-1]))
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs, old_key], self._num_units, True)))
               # with tf.device("/cpu:0"):
                k = tf.transpose(tf.concat(0, [tf.real(key), tf.imag(key)]))
                k_real, k_imag = tf.split(0, 2, tf.transpose(tf.nn.embedding_lookup(k, perms)))
                ks = tf.complex(k_real, k_imag)
		#ks_real = self._num_copies
		#ks_imag = tf.split(1, self._num_copies, k_imag)
 		#ks = [tf.complex(r,i) for r,i in zip(ks_real, ks_imag)]

            with vs.variable_scope("Read"):
                old_f = uncomplexify(self._read(tf.conj(ks), ss))

            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, linear([inputs, old_f],
                                                     2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = tanh(linear([inputs, r * old_f], self._num_units, True))

            to_add = u * (c - old_f)
            to_add_r, to_add_i = tf.split(1, 2, to_add)
            c_to_add = tf.complex(tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
            new_ss = uncomplexify(ss + ks * c_to_add)
            new_h = old_f + to_add

        return new_h, tf.concat(1, [new_h, uncomplexify(key), new_ss])

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
        return self._num_units * (self._num_copies+3)

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
    im_v = tf.imag(v)
    re_v = tf.real(v)
    v_sqrt = tf.maximum(1.0, tf.sqrt(im_v * im_v + re_v * re_v))
    return tf.complex(re_v / v_sqrt, im_v / v_sqrt)

