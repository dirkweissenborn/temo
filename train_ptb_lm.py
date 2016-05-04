from rnn_cell_plus import *
import numpy as np
import random
import time
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.ptb import reader
import os
import copy
#  Adapted from tensorflow ptb_word_lm.py

class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, FLAGS):
        self.batch_size = batch_size = FLAGS.batch_size
        self.num_steps = num_steps = FLAGS.num_steps
        vocab_size = FLAGS.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        mem_size = FLAGS.mem_size
        batch_size = FLAGS.batch_size

        cell = None
        if FLAGS.cell == 'LSTM':
            cell = BasicLSTMCell(mem_size)
        elif FLAGS.cell == 'GRU':
            cell = GRUCell(mem_size)
        elif FLAGS.cell == 'MORU':
            biases = FLAGS.moru_op_biases
            if biases is not None:
                biases = map(lambda s: float(s), biases.split(","))
            ops = FLAGS.moru_ops.split(",")
            cell = MORUCell.from_op_names(ops, biases, mem_size, op_controller_size=FLAGS.moru_op_ctr)

        if FLAGS.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layers)

        if is_training and FLAGS.keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=FLAGS.keep_prob, output_keep_prob=FLAGS.keep_prob)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, mem_size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and FLAGS.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, mem_size])
        softmax_w = tf.get_variable("softmax_w", [mem_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(FLAGS.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          FLAGS.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.0)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


if __name__ == "__main__":
    # data loading specifics
    tf.app.flags.DEFINE_string('data', 'data/ptb', 'data dir of ptb.')

    # model
    tf.app.flags.DEFINE_integer("mem_size", 200, "hidden size of model")
    tf.app.flags.DEFINE_float("init_scale", 0.05, "uniform init weight")

    # training
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
    tf.app.flags.DEFINE_float("learning_rate_decay", 1.0, "Learning rate decay when loss on validation set does not improve.")
    tf.app.flags.DEFINE_float("max_grad_norm", 5.0, "Maximum norm of grad, after that use clipping.")
    tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples per batch.")
    tf.app.flags.DEFINE_integer("num_steps", 30, "Number of steps per training batch.")
    tf.app.flags.DEFINE_integer("vocab_size", 10000, "Size of vocabulary.")
    tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
    tf.app.flags.DEFINE_string("cell", 'MORU', "'LSTM', 'GRU', 'MORU'")
    tf.app.flags.DEFINE_integer("seed", 12345, "Random seed.")
    tf.app.flags.DEFINE_integer("runs", 10, "How many runs.")
    tf.app.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")
    tf.app.flags.DEFINE_string("result_file", None, "Where to write results.")
    tf.app.flags.DEFINE_string("moru_ops", 'keep,replace', "operations of moru cell.")
    tf.app.flags.DEFINE_string("moru_op_biases", None, "biases of moru operations at beginning of training. "
                                                       "Defaults to 0 for each.")
    tf.app.flags.DEFINE_integer("moru_op_ctr", None, "Size of op ctr. By default ops are controlled by current input"
                                                     "and previous state. Given a positive integer, an additional"
                                                     "recurrent op ctr is introduced in MORUCell.")

    FLAGS = tf.app.flags.FLAGS
    eval_FLAGS = copy.deepcopy(FLAGS)
    raw_data = reader.ptb_raw_data(FLAGS.data)
    train_data, valid_data, test_data, _ = raw_data
    perplexities = []

    rng = random.Random(FLAGS.seed)
    for run_id in range(FLAGS.runs):
        tf.reset_default_graph()
        last_valid_perplexities = [float("inf")] * 3
        with tf.Session() as sess:
            tf.set_random_seed(rng.randint(0, 10000))
            initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                        FLAGS.init_scale)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, FLAGS=FLAGS)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, FLAGS=FLAGS)
                eval_FLAGS.batch_size = 1
                eval_FLAGS.num_steps = 1
                mtest = PTBModel(is_training=False, FLAGS=eval_FLAGS)

            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.trainable_variables())
            i = 0
            while True:
                print("Epoch: %d Learning rate: %.4f" % (i + 1, sess.run(m.lr)))
                train_perplexity = run_epoch(sess, m, train_data, m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(sess, mvalid, valid_data, tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                if last_valid_perplexities[-1]-1 < valid_perplexity:
                    if min(last_valid_perplexities[:-1])-1 < valid_perplexity:
                        break
                    if FLAGS.learning_rate_decay < 1.0:
                        print("Decaying learning rate.")
                        sess.run(m.lr.assign(m.lr * FLAGS.learning_rate_decay))
                else:
                    saver.save(sess, '/tmp/my-model')

                last_valid_perplexities.append(valid_perplexity)
                last_valid_perplexities = last_valid_perplexities[1:]
                i += 1

            saver.restore(sess, '/tmp/my-model')
            test_perplexity = run_epoch(sess, mtest, test_data, tf.no_op())
            perplexities.append(test_perplexity)
            print("######## Run %d #########" % run_id)
            print("Test Perplexity: %.3f" % test_perplexity)
            print('########################')
            os.remove('/tmp/my-model')

    mean_perplexities = sum(perplexities) / len(perplexities)


    def s_dev(mean, pop):
        d = 0.0
        for el in pop:
            d += (mean - el) * (mean - el)
        return math.sqrt(d / len(pop))

    import json
    if FLAGS.result_file:
        with open(FLAGS.result_file, 'w') as f:
            f.write('Test Perplexity: %.4f (%.4f)\n\n' % (mean_perplexities, s_dev(mean_perplexities, perplexities)))
            f.write("Configuration: \n")
            f.write(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))

    print('######## Overall #########')
    print('Test Perplexity: %.4f (%.4f)' % (mean_perplexities, s_dev(mean_perplexities, perplexities)))
    print('########################')
