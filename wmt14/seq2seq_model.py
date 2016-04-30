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
from moru_cell import *


class Seq2SeqModel(object):
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
                 learning_rate_decay_factor, cell_type="GRU",
                 num_samples=-1, forward_only=False):
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
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                      self.target_vocab_size)
            softmax_loss_function = sampled_loss

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, encoder_length, decoder_length, do_decode):
            if cell_type == "AssociativeGRU":
                source_cell = AssociativeGRUCell(size, num_copies=8, input_size=size, rng=random.Random(123))
                target_cell = DualAssociativeGRUCell(size, num_copies=8, input_size=size, share=False, rng=random.Random(123))

                source_cell = EmbeddingWrapper(source_cell, source_vocab_size, size)

                print("Use AssociativeGRU")
                with tf.variable_scope("source"):
                    with tf.variable_scope("assoc"):
                        source_out, c = my_seq2seq.my_rnn(source_cell, encoder_inputs, sequence_length=encoder_length, dtype=tf.float32)
                        source_out = [tf.reshape(o, [-1, size]) for o in source_out]
                    with tf.variable_scope("rnn"):
                        _, final_source_state = my_seq2seq.my_rnn(GRUCell(size, source_cell.output_size),
                                                                  source_out, sequence_length=encoder_length, dtype=tf.float32)

                with tf.variable_scope("target"):
                    rest_state = tf.zeros([batch_size, target_cell.state_size - source_cell.state_size + source_cell.output_size], tf.float32)
                    c = tf.concat(1, [rest_state, tf.slice(c, [0, source_cell.output_size], [-1, -1]), final_source_state])

                    target_cell = tf.nn.rnn_cell.MultiRNNCell([target_cell, GRUCell(size, target_cell.output_size)])
                    return my_seq2seq.embedding_rnn_decoder(decoder_inputs, decoder_length, c, target_cell,
                                                            target_vocab_size, size, output_projection=output_projection,
                                                            feed_previous=do_decode)
            else:
                # Create the internal multi-layer cell for our RNN.
                single_cell = None
                if cell_type == "GRU":
                    single_cell = tf.nn.rnn_cell.GRUCell(size)
                else:
                    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)

                cell = single_cell
                if num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
                return my_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.encoder_length = tf.placeholder(tf.int32, shape=[None], name="encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, shape=[None], name="decoder_length")
        for i in range(max_length):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(max_length):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, _ = seq2seq_f(self.encoder_inputs, self.decoder_inputs,
                                        self.encoder_length, self.decoder_length, True)
        else:
            self.outputs, _ = seq2seq_f(self.encoder_inputs, self.decoder_inputs,
                                        self.encoder_length, self.decoder_length, False)

        self.loss = my_seq2seq.sequence_loss(self.outputs[:-1], targets, self.decoder_length,
                                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                                            max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, encoder_length, decoder_length, forward_only):
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
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        for l in range(encoder_size, self.max_length):
            input_feed[self.encoder_inputs[l].name] = decoder_inputs[-1]
        for l in range(decoder_size, self.max_length):
            input_feed[self.decoder_inputs[l].name] = encoder_inputs[-1]

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
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

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
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = [], [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            encoder_size = len(encoder_input)
            decoder_size = len(decoder_input)
            encoder_lengths.append(encoder_size)
            decoder_lengths.append(decoder_size)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - encoder_size)
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = self.max_length - decoder_size - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(self.max_length):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        return batch_encoder_inputs, batch_decoder_inputs, \
               np.array(encoder_lengths, dtype=np.int32), np.array(decoder_lengths, dtype=np.int32)
