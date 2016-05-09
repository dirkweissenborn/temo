# Adaption from original TF code

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from wmt14 import data_utils
from wmt14 import my_seq2seq
from rnn_cell_plus import *


class TranslationModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, source_vocab_size, target_vocab_size, max_length, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, cell_type="GRU", attention=False,
                 num_read_keys=0, forward_only=False):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          max_length: max_length of source or target sentence
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self._rng = random.Random(123)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope("translation", initializer=initializer):
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            self.symbol_bias = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, self.symbol_bias)

            def seq2seq_f(encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, do_decode):
                enc_cell, ctr_cell = None, None
                if cell_type == "AssociativeGRU":
                    print("Use AssociativeGRU!")
                    enc_cell = AssociativeGRUCell(size, num_copies=8, input_size=2*size,
                                                  rng=random.Random(123), num_read_keys=num_read_keys)
                    dec_cell = DualAssociativeGRUCell(size, num_read_mems=2, num_copies=8, input_size=2*size,
                                                      rng=random.Random(123), num_read_keys=num_read_keys)

                    enc_cell = SelfControllerWrapper(enc_cell, size)

                    def outproj(out, out_size):
                        output = linear([out], 2 * out_size, True)
                        output = tf.reduce_max(tf.reshape(output, [-1, 2, out_size]), [1], keep_dims=False)
                        return output
                    dec_cell = SelfControllerWrapper(dec_cell, size, outproj, size)

                    inputs = rev_encoder_inputs
                    with tf.variable_scope("encoder"):
                        with tf.variable_scope("embedding"):
                            with ops.device("/cpu:0"):
                                embedding = vs.get_variable("embedding", [source_vocab_size, size])
                                embedded = [tf.nn.embedding_lookup(embedding, array_ops.reshape(inp, [-1])) for inp in encoder_inputs]
                                rev_embedded = [tf.nn.embedding_lookup(embedding, array_ops.reshape(inp, [-1])) for inp in rev_encoder_inputs]

                        # Encoder.
                        with tf.variable_scope("forward"):
                            _, encoder_state = \
                                my_seq2seq.my_rnn(enc_cell, embedded, sequence_length=encoder_length, dtype=tf.float32)

                        with tf.variable_scope("backward"):
                            _, rev_encoder_state = \
                                my_seq2seq.my_rnn(enc_cell, rev_embedded, sequence_length=encoder_length, dtype=tf.float32)

                    c = None
                    encoder_mem = tf.slice(encoder_state, [0, 0], [-1, enc_cell.state_size-size])
                    if dec_cell.state_size > enc_cell.state_size*2-size:
                        rest_state = tf.zeros([batch_size, dec_cell.state_size - enc_cell.state_size*2 + size], tf.float32)
                        # zero memory for decoder assoc mem + assoc_mem of forward and backward encoder + rev output of self controller (within rev_encoder_state)
                        c = tf.concat(1, [rest_state, encoder_mem, rev_encoder_state])
                    else:
                        c = tf.concat(1, [encoder_mem, rev_encoder_state])
                    c = tf.reshape(c, [-1, dec_cell.state_size])
                    return my_seq2seq.embedding_rnn_decoder(decoder_inputs, decoder_length, c, dec_cell,
                                                            target_vocab_size, size,
                                                            output_projection=output_projection, beam_size=5,
                                                            feed_previous=do_decode)
                else:
                    enc_cell = None
                    if cell_type == "LSTM":
                        enc_cell = BasicLSTMCell(size,size)
                    else:
                        enc_cell = GRUCell(size, size)
                    if num_layers > 1:
                        enc_cell = tf.nn.rnn_cell.MultiRNNCell([enc_cell] * num_layers)
                    if attention:
                        print("Use attention!")
                        ctr_cell = None
                        if cell_type == "LSTM":
                            ctr_cell = BasicLSTMCell(size,input_size=2*size)
                        else:
                            ctr_cell = GRUCell(size, 2*size)

                        with tf.variable_scope("encoder"):
                            with tf.variable_scope("embedding"):
                                with ops.device("/cpu:0"):
                                    embedding = vs.get_variable("embedding", [source_vocab_size, size])
                                    embedded = [tf.nn.embedding_lookup(embedding, array_ops.reshape(inp, [-1])) for inp in encoder_inputs]
                                    rev_embedded = [tf.nn.embedding_lookup(embedding, array_ops.reshape(inp, [-1])) for inp in rev_encoder_inputs]

                            # Encoder.
                            with tf.variable_scope("forward"):
                                encoder_outputs, encoder_states = \
                                    my_seq2seq.my_rnn(enc_cell, embedded, sequence_length=encoder_length, dtype=tf.float32)

                                top_states = [array_ops.reshape(e, [-1, 1, enc_cell.output_size])
                                              for e in encoder_outputs]
                                encoder_states = array_ops.concat(1, top_states)
                            with tf.variable_scope("backward"):
                                rev_encoder_outputs, rev_encoder_state = \
                                    my_seq2seq.my_rnn(enc_cell, rev_embedded, sequence_length=encoder_length, dtype=tf.float32)

                                rev_top_states = [array_ops.reshape(e, [-1, 1, enc_cell.output_size])
                                                  for e in rev_encoder_outputs]
                                rev_attention_states = tf.reverse_sequence(
                                    tf.concat(1, rev_top_states), tf.cast(encoder_length, tf.int64), 1, 0)

                            encoder_states = tf.reshape(tf.concat(2, [encoder_states, rev_attention_states]),
                                                          [batch_size, -1, enc_cell.output_size])

                        c = tf.slice(rev_encoder_state, [0, 0], [-1, ctr_cell.state_size])
                        def outproj(out, out_size):
                            output = linear([out], 2 * out_size, True)
                            output = tf.reduce_max(tf.reshape(output, [-1, 2, out_size]), [1], keep_dims=False)
                            return output
                        dec_cell = ControllerWrapper(ctr_cell, AttentionCell(encoder_states, encoder_length, size), outproj, size)

                        c = tf.concat(1, [c, tf.zeros([batch_size, dec_cell.state_size-enc_cell.state_size])])

                        # Decoder.
                        return my_seq2seq.embedding_rnn_decoder(
                            decoder_inputs, decoder_length, c, dec_cell,
                            target_vocab_size, size,
                            output_projection=output_projection if do_decode else None,
                            feed_previous=do_decode, beam_size=5)
                    else:
                        with tf.variable_scope("encoder"):
                            _, rev_encoder_state, _ = \
                                    my_seq2seq.embedding_rnn_decoder(rev_encoder_inputs, encoder_length,
                                                                     enc_cell.zero_state(batch_size, tf.float32),
                                                                     enc_cell, source_vocab_size, size,
                                                                     feed_previous=False)

                        dec_cell = TranslationOutputWrapper(enc_cell, size)
                        return my_seq2seq.embedding_rnn_decoder(decoder_inputs, decoder_length, rev_encoder_state,
                                                                dec_cell, target_vocab_size, size,
                                                                output_projection=output_projection if do_decode else None,
                                                                feed_previous=do_decode, beam_size=5)


            # Feeds for inputs.
            self.encoder_inputs = []
            self.rev_encoder_inputs = []
            self.decoder_inputs = []
            self.encoder_length = tf.placeholder(tf.int32, shape=[None], name="encoder_length")
            self.decoder_length = tf.placeholder(tf.int32, shape=[None], name="decoder_length")
            for i in range(max_length):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="encoder{0}".format(i)))
            for i in range(max_length):  # Last bucket is the biggest one.
                self.rev_encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                              name="rev_encoder{0}".format(i)))
            for i in range(max_length):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="decoder{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1]
                       for i in range(len(self.decoder_inputs) - 1)]

            # Training outputs and losses.
            rnn_outputs, _, self.decoded = seq2seq_f(self.encoder_inputs, self.rev_encoder_inputs, self.decoder_inputs,
                                                     self.encoder_length, self.decoder_length, forward_only)
            self.outputs = [nn_ops.xw_plus_b(o, w, self.symbol_bias) for o in rnn_outputs]

            # Gradients and SGD update operation for training the model.
            if not forward_only:
                params = tf.trainable_variables()
                self.loss = my_seq2seq.sequence_loss(self.outputs[:-1], targets, self.decoder_length,
                                                     softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
                opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)
                gradients = tf.gradients(self.loss, params)
                for g, p in zip(gradients, params):
                    if g is None:
                        print("Gradient for %s is None." % p.name)
               # gradients = [tf.clip_by_value(g, -5.0, 5.0) if g is not None else g for g in gradients]
                clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            self.saver = tf.train.Saver(tf.all_variables())

    def set_no_unk(self, sess):
        sess.run(tf.scatter_update(self.symbol_bias, [data_utils.UNK_ID], [-10000]))

    def step(self, session, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = max(encoder_length), max(decoder_length)

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.rev_encoder_inputs[l].name] = rev_encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        for l in range(encoder_size, self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[-1]
            input_feed[self.rev_encoder_inputs[l].name] = rev_encoder_inputs[-1]
        for l in range(decoder_size, self.max_length):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[-1]

        input_feed[self.encoder_length.name] = encoder_length
        input_feed[self.decoder_length.name] = decoder_length

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.loss]  # Loss for this batch.
        else:
            output_feed = [self.loss]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # None, loss, outputs


    def decode(self, session, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, forward_only):
        encoder_size, decoder_size = max(encoder_length), max(decoder_length)
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(encoder_size):
            input_feed[self.rev_encoder_inputs[l].name] = rev_encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        for l in range(encoder_size, self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[-1]
            input_feed[self.rev_encoder_inputs[l].name] = rev_encoder_inputs[-1]
        for l in range(decoder_size, self.max_length):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[-1]

        input_feed[self.encoder_length.name] = encoder_length
        input_feed[self.decoder_length.name] = decoder_length

        outputs = session.run(self.decoded, input_feed)
        return outputs # loss, outputs, decoded symbols

    def reset_rng(self, sess, data):
        num_steps = self.global_step.eval(sess)
        for i in range(num_steps):
            for _ in range(self.batch_size):
                _, _ = self._rng.choice(data)


    def get_batch(self, data):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = [], [], [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = self._rng.choice(data)
            encoder_size = len(encoder_input)
            decoder_size = len(decoder_input)
            encoder_lengths.append(encoder_size)
            decoder_lengths.append(decoder_size)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - encoder_size)
            encoder_inputs.append(encoder_input + encoder_pad)
            rev_encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = self.max_length - decoder_size - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_rev_encoder_inputs, batch_decoder_inputs = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(self.max_length):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))
            batch_rev_encoder_inputs.append(
                np.array([rev_encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        return batch_encoder_inputs, batch_rev_encoder_inputs, batch_decoder_inputs, \
               np.array(encoder_lengths, dtype=np.int32), np.array(decoder_lengths, dtype=np.int32)


class TranslationOutputWrapper(RNNCell):

    def __init__(self, cell, output_size):
        self._cell = cell
        self._output_size = output_size

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        # Default scope: "OutputProjectionWrapper"
        with vs.variable_scope("Output_Projection"):
            output = linear([output], 2 * self._output_size, True)
            output = tf.reduce_max(tf.reshape(output, [-1, 2, self._output_size]), [1], keep_dims=False)
        return output, res_state
