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
        return self._num_units * self._num_copies

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_ss = state
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs], (1+self._num_read_keys)*self._num_units, False)))
                k = [_comp_real(key), _comp_imag(key)]
                if self._num_copies > 1:
                    if self._num_read_keys > 0:
                        k = tf.transpose(tf.concat(0, tf.split(1, 1+self._num_read_keys, k[0]) + tf.split(1, 1+self._num_read_keys, k[1])), [1, 0])
                    else:
                        k = tf.transpose(tf.concat(0, k), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    k = tf.split(0, 2*(1+self._num_read_keys), tf.transpose(k, [1, 0]))

                w_k_real = k[0]
                w_k_imag = k[1+self._num_read_keys]
                r_k_real = k[1:1+self._num_read_keys]
                r_k_imag = k[2+self._num_read_keys:]
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

        return new_h, new_ss

    def _read(self, keys, redundant_states):
        read = _comp_mul(keys, redundant_states)
        if self._num_copies > 1:
            xs_real = tf.split(1, self._num_copies, _comp_real(read))
            xs_imag = tf.split(1, self._num_copies, _comp_imag(read))
            read = (tf.add_n(xs_real)/self._num_copies, tf.add_n(xs_imag)/self._num_copies)
        return read


class ControllerWrapper(RNNCell):

    def __init__(self, controller_cell, cell, output_proj=None, out_size=None):
        self._cell = cell
        self._controller_cell = controller_cell
        self._output_proj = output_proj
        self._out_size = out_size

    @property
    def output_size(self):
        if self._out_size is None:
            return self._controller_cell.output_size + self._cell.output_size
        else:
            return self._out_size

    def __call__(self, inputs, state, scope=None):
        ctr_state = tf.slice(state, [0, 0], [-1,self._controller_cell.state_size])
        inner_state = None
        if self._cell.state_size > 0:
            inner_state = tf.slice(state, [0, self._controller_cell.state_size], [-1, self._cell.state_size])
        inner_out = tf.slice(state, [0, self._controller_cell.state_size+self._cell.state_size], [-1,-1])
        inputs = tf.concat(1, [inputs, inner_out])
        ctr_out, ctr_state = self._controller_cell(inputs, ctr_state)
        inner_out, inner_state = self._cell(ctr_out, inner_state)
        out = tf.concat(1, [ctr_out, inner_out])
        if self._output_proj is not None:
            with tf.variable_scope("Output_Projection"):
                out = self._output_proj(out, self.output_size)
        if self._cell.state_size > 0:
            return out, tf.concat(1, [ctr_state, inner_state, inner_out])
        else:
            return out, tf.concat(1, [ctr_state, inner_out])

    def zero_state(self, batch_size, dtype):
        if self._cell.state_size > 0:
            return tf.concat(1, [self._controller_cell.zero_state(batch_size, dtype), self._cell.zero_state(batch_size, dtype),
                             tf.zeros([batch_size,self._cell.output_size]),tf.float32])
        else:
            return tf.concat(1, [self._controller_cell.zero_state(batch_size, dtype), tf.zeros([batch_size,self._cell.output_size]),tf.float32])

    @property
    def input_size(self):
        return self._controller_cell.input_size

    @property
    def state_size(self):
        return self._controller_cell.state_size + self._cell.state_size + self._cell.output_size


class SelfControllerWrapper(RNNCell):

    def __init__(self, cell, input_size, output_proj=None, out_size=None):
        self._cell = cell
        self._output_proj = output_proj
        self._input_size = input_size
        self._out_size = out_size

    @property
    def output_size(self):
        if self._out_size is None:
            return self._cell.output_size
        else:
            return self._out_size

    def __call__(self, inputs, state, scope=None):
        prev_state = None
        if self._cell.state_size > 0:
            prev_state = tf.slice(state, [0, 0], [-1, self._cell.state_size])
        prev_out = tf.slice(state, [0, self._cell.state_size], [-1,self._cell.output_size])
        inputs = tf.concat(1, [inputs, prev_out])
        new_out, prev_state = self._cell(inputs, prev_state)
        out = new_out
        if self._output_proj is not None:
            with tf.variable_scope("Output_Projection"):
                out = self._output_proj(out, self.output_size)
        if self._cell.state_size > 0:
            return out, tf.concat(1, [prev_state, new_out])
        else:
            return out, new_out

    def zero_state(self, batch_size, dtype):
        if self._cell.state_size > 0:
            return tf.concat(1, [self._cell.zero_state(batch_size, dtype), tf.zeros(tf.pack([batch_size,self._cell.output_size]),tf.float32)])
        else:
            return tf.zeros(tf.pack([batch_size,self._cell.output_size]),tf.float32)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._cell.state_size + self._cell.output_size

class DualAssociativeGRUCell(AssociativeGRUCell):

    def __init__(self, num_units, num_read_mems=1, num_copies=1, input_size=None, num_read_keys=0, read_only=False, rng=None):
        self._num_read_mems = num_read_mems
        AssociativeGRUCell.__init__(self, num_units, num_copies, input_size, num_read_keys, read_only, rng)

    @property
    def state_size(self):
        return self._num_units * self._num_copies * (1+self._num_read_mems)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_ss = tf.slice(state, [0, 0], [-1, self._num_units * self._num_copies])
            dual_mem = tf.slice(state, [0, self._num_units * self._num_copies], [-1, -1])
            read_mems = tf.split(1, self._num_read_mems, dual_mem)
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(linear([inputs], (1+self._num_read_keys)*self._num_units, False)))
                k = [_comp_real(key), _comp_imag(key)]
                if self._num_copies > 1:
                    if self._num_read_keys > 0:
                        k = tf.transpose(tf.concat(0, tf.split(1, 1+self._num_read_keys, k[0]) + tf.split(1, 1+self._num_read_keys, k[1])), [1, 0])
                    else:
                        k = tf.transpose(tf.concat(0, k), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    k = tf.split(0, 2*(1+self._num_read_keys), tf.transpose(k, [1, 0]))
                w_k_real = k[0]
                w_k_imag = k[1+self._num_read_keys]
                r_k_real = k[1:1+self._num_read_keys]
                r_k_imag = k[2+self._num_read_keys:]
                r_keys = list(zip(r_k_real, r_k_imag))
                w_key = (w_k_real, w_k_imag)
                conj_w_key = _comp_conj(w_key)

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(conj_w_key, c_ss), "retrieved")
                r_hs = []
                for i, k in enumerate(r_keys):
                    r_hs.append(uncomplexify(self._read(k, c_ss), "read_%d" % i))

            with vs.variable_scope("Read_Given"):
                if self._num_read_mems > 1:
                    h2_s = [uncomplexify(self._read(conj_w_key, complexify(read_mem))) for read_mem in read_mems]
                    h2 = tf.reshape(tf.concat(1, h2_s, name="retrieved"), [-1, self._num_read_mems*self._num_units])
                else:
                    h2 = uncomplexify(self._read(conj_w_key, complexify(read_mems[0])), "retrieved")

            with vs.variable_scope("Gates"):
                gs = linear(r_hs + [inputs, h], 2 * self._num_units, True, 1.0)
            with vs.variable_scope("DualGates"):
                vs.get_variable_scope()._reuse = \
                    any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())
                gs = sigmoid(gs + linear([h2], 2 * self._num_units, False))
            r, u = tf.split(1, 2, gs)

            with vs.variable_scope("Candidate"):
                c = linear(r_hs + [inputs, r * h], self._num_units, True)

            with vs.variable_scope("DualCandidate"):
                vs.get_variable_scope()._reuse = \
                    any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())  # HACK
                c = tanh(c + linear([h2], self._num_units, False))

            to_add = u * (c - h)
            to_add_r, to_add_i = tf.split(1, 2, to_add)
            c_to_add = (tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
            new_ss = old_ss + uncomplexify(_comp_mul(w_key, c_to_add))
            new_h = tf.add(h, to_add, "out")

        return new_h, tf.concat(1, [new_ss, dual_mem])


#use with controller
class AttentionCell(RNNCell):

    def __init__(self, attention_states, attention_length, input_size=None, num_heads=1):
        self._attention_states = attention_states
        self._attention_length = attention_length
        self._num_heads = num_heads
        self._hidden_features = None
        self._num_units = self._attention_states.get_shape()[2].value
        self._input_size = input_size if input_size is not None else self._num_units

    @property
    def output_size(self):
        return self._attention_states.get_shape()[2].value * self._num_heads

    def __call__(self, inputs, state, scope=None):
        if self._hidden_features is None:
            self._attn_length = math_ops.reduce_max(self._attention_length)
            attention_states = tf.slice(self._attention_states, [0,0,0], tf.pack([-1, self._attn_length, -1]))
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            self._hidden = array_ops.reshape(attention_states, tf.pack([-1, self._attn_length, 1, self._num_units]))
            hidden_features = []

            for a in range(self._num_heads):
                k = tf.get_variable("AttnW_%d" % a, [1, 1, self._num_units, self._num_units])
                hidden_features.append(nn_ops.conv2d(self._hidden, k, [1, 1, 1, 1], "SAME"))
            self._hidden_features = hidden_features


        ds = []  # Results of attention reads will be stored here.

        batch_size = tf.shape(inputs)[0]
        mask = tf.tile(tf.reshape(tf.lin_space(1.0, tf.cast(self._attn_length, tf.float32), self._attn_length), [1, -1]),
                       tf.pack([batch_size, 1]))
        batch_size_scale = batch_size // tf.shape(self._attention_length)[0] # used in decoding
        lengths = tf.tile(tf.expand_dims(tf.cast(self._attention_length, tf.float32), 1),
                          tf.pack([batch_size_scale, self._attn_length]))

        mask = tf.cast(tf.greater(mask, lengths), tf.float32) * -1000.0
        for a in range(self._num_heads):
            with tf.variable_scope("Attention_%d" % a):
                y = linear(inputs, self._num_units, True)
                y = array_ops.reshape(y, [-1, 1, 1, self._num_units])
                # Attention mask is a softmax of v^T * tanh(...).
                v = tf.get_variable("AttnV_%d" % a, [self._num_units])
                hf = tf.cond(tf.equal(batch_size_scale,1),
                             lambda: self._hidden_features[a],
                             lambda: tf.tile(self._hidden_features[a], tf.pack([batch_size_scale,1,1,1])))
                s = math_ops.reduce_sum(v * math_ops.tanh(hf + y), [2, 3])

                a = nn_ops.softmax(s + mask)
                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                    array_ops.reshape(a, tf.pack([-1, self._attn_length, 1, 1])) * self._hidden,
                    [1, 2])
                ds.append(array_ops.reshape(d, [-1, self._num_units]))

        return tf.concat(1, ds), None

    def zero_state(self, batch_size, dtype):
        return super().zero_state(batch_size, dtype)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return 0


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