# adaptions to make use of length as input in rnns

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

"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
import tensorflow as tf


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function



def my_rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNCell "cell".

    The simplest form of RNN network generated is:
      state = cell.zero_state(...)
      outputs = []
      for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
      return (outputs, state)

    However, a few other options are available:

    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.

    The dynamic calculation performed is, at time t for batch row b,
      (output, state)(b, t) =
        (t >= sequence_length(b))
          ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
          : cell(input(b, t), state(b, t - 1))

    Args:
      cell: An instance of RNNCell.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, cell.input_size].
      initial_state: (optional) An initial state for the RNN.  This must be
        a tensor of appropriate type and shape [batch_size x cell.state_size].
      dtype: (optional) The data type for the initial state.  Required if
        initial_state is not provided.
      sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size [batch_size].  Values in [0, T).
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A pair (outputs, state) where:
        outputs is a length T list of outputs (one for each input)
        state is the final state

    Raises:
      TypeError: If "cell" is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list, or if the input depth
        cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if inputs[0].get_shape().ndims != 1:
            (fixed_batch_size, input_size) = inputs[0].get_shape().with_rank(2)
            if input_size.value is None:
                raise ValueError(
                    "Input size (second dimension of inputs[0]) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(inputs[0])[0]

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)

        if sequence_length is not None:  # Prepare variables
            zero_output = array_ops.zeros(tf.pack([batch_size, cell.output_size]),
                                          dtype if dtype is not None else initial_state.dtype)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = rnn._rnn_step(
                    time, sequence_length, min_sequence_length, max_sequence_length,
                    zero_output, state, call_cell)
            else:
                (output, state) = call_cell()

            outputs.append(output)

        return (outputs, state)


def rnn_decoder(decoder_inputs, sequence_length, initial_state, cell, loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    """
    with variable_scope.variable_scope(scope or "rnn_decoder") as varscope:
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output

        outputs = []
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

            # Temporarily avoid EmbeddingWrapper and seq2seq badness
            # TODO(lukaszkaiser): remove EmbeddingWrapper
        if decoder_inputs[0].get_shape().ndims != 1:
            (fixed_batch_size, input_size) = decoder_inputs[0].get_shape().with_rank(2)
            if input_size.value is None:
                raise ValueError(
                    "Input size (second dimension of inputs[0]) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = decoder_inputs[0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(decoder_inputs[0])[0]

        state = initial_state

        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)

        if sequence_length is not None:  # Prepare variables
            zero_output = tf.zeros(tf.pack([batch_size, cell.output_size]), initial_state.dtype)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        prev = None
        for time, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, inp)
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(inp, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = rnn._rnn_step(
                    time, sequence_length, min_sequence_length, max_sequence_length,
                    zero_output, state, call_cell)
            else:
                (output, state) = call_cell()
            outputs.append(output)
            if loop_function is not None:
                prev = output

        return outputs, state

def basic_rnn_seq2seq(encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell, dtype=dtypes.float32, scope=None):
    """Basic RNN sequence-to-sequence model.

    This model first runs an RNN to encode encoder_inputs into a state vector,
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell type, but don't share parameters.

    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size].
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
        _, enc_state = my_rnn(cell, encoder_inputs, dtype=dtype, sequence_length=encoder_length)
        return rnn_decoder(decoder_inputs, decoder_length, enc_state, cell)


def tied_rnn_seq2seq(encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell,
                     loop_function=None, dtype=dtypes.float32, scope=None):
    """RNN sequence-to-sequence model with tied encoder and decoder parameters.

    This model first runs an RNN to encode encoder_inputs into a state vector, and
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell and share parameters.

    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size].
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol), see rnn_decoder for details.
      dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
        scope = scope or "tied_rnn_seq2seq"
        _, enc_state = my_rnn(
            cell, encoder_inputs, dtype=dtype, scope=scope, sequence_length=encoder_length)
        variable_scope.get_variable_scope().reuse_variables()
        return rnn_decoder(decoder_inputs, decoder_length, enc_state, cell,
                           loop_function=loop_function, scope=scope)


def embedding_rnn_decoder(decoder_inputs, decoder_length, initial_state, cell, num_symbols,
                          embedding_size, output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True, scope=None):
    """RNN decoder with embedding and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has
        shape [num_symbols]; if provided and feed_previous=True, each fed
        previous output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_projection is not None:
        proj_weights = ops.convert_to_tensor(output_projection[0],
                                             dtype=dtypes.float32)
        proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
        proj_biases = ops.convert_to_tensor(
            output_projection[1], dtype=dtypes.float32)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
        with ops.device("/cpu:0"):
            embedding = variable_scope.get_variable("embedding",
                                                    [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return rnn_decoder(emb_inp, decoder_length, initial_state, cell,
                           loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell,
                          num_encoder_symbols, num_decoder_symbols,
                          embedding_size, output_projection=None,
                          feed_previous=False, dtype=dtypes.float32,
                          scope=None):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial state for both the encoder and encoder
        rnn cells (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_seq2seq"

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq"):
        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_state = my_rnn(encoder_cell, encoder_inputs, dtype=dtype, sequence_length=encoder_length)

        # Decoder.
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        if isinstance(feed_previous, bool):
            return embedding_rnn_decoder(
                decoder_inputs, decoder_length, encoder_state, cell, num_decoder_symbols,
                embedding_size, output_projection=output_projection,
                feed_previous=feed_previous)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=reuse):
                outputs, state = embedding_rnn_decoder(
                    decoder_inputs, decoder_length, encoder_state, cell, num_decoder_symbols,
                    embedding_size, output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False)
                return outputs + [state]

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        return outputs_and_state[:-1], outputs_and_state[-1]


def embedding_tied_rnn_seq2seq(encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell,
                               num_symbols, embedding_size,
                               output_projection=None, feed_previous=False,
                               dtype=dtypes.float32, scope=None):
    """Embedding RNN sequence-to-sequence model with tied (shared) parameters.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_symbols x input_size]). Then it runs an RNN to encode embedded
    encoder_inputs into a state vector. Next, it embeds decoder_inputs using
    the same embedding. Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_symbols: Integer; number of symbols for both encoder and decoder.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has
        shape [num_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype to use for the initial RNN states (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_tied_rnn_seq2seq".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_projection is not None:
        proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
        proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_tied_rnn_seq2seq"):
        with ops.device("/cpu:0"):
            embedding = variable_scope.get_variable("embedding",
                                                    [num_symbols, embedding_size])

        emb_encoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_decoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                              for x in decoder_inputs]

        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_symbols)

        if isinstance(feed_previous, bool):
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection, True) if feed_previous else None
            return tied_rnn_seq2seq(emb_encoder_inputs, emb_decoder_inputs, encoder_length, decoder_length, cell,
                                    loop_function=loop_function, dtype=dtype)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection, False) if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=reuse):
                outputs, state = tied_rnn_seq2seq(
                    emb_encoder_inputs, emb_decoder_inputs, encoder_length, decoder_length, cell,
                    loop_function=loop_function, dtype=dtype)
                return outputs + [state]

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        return outputs_and_state[:-1], outputs_and_state[-1]


def attention_decoder(decoder_inputs, decoder_length,  initial_state, attention_states, attention_length, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: when num_heads is not positive, there are no inputs, or shapes
        of attention_states are not set.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
   # if not attention_states.get_shape()[1:2].is_fully_defined():
   #    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                        # % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = math_ops.reduce_max(attention_length)
        attn_size = attention_states.get_shape()[2].value
        attention_states = tf.slice(attention_states, [0,0,0], tf.pack([-1, attn_length, cell.output_size]))
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, tf.pack([-1, attn_length, 1, attn_size]))
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            mask = tf.tile(tf.reshape(tf.lin_space(1.0, tf.cast(attn_length, tf.float32), attn_length), [1, -1]),
                           tf.pack([batch_size, 1]))
            lengths = tf.tile(tf.reshape(tf.cast(attention_length, tf.float32), [-1, 1]), tf.pack([1, attn_length]))
            mask = tf.cast(tf.greater(mask, lengths), tf.float32) * -1000.0
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = rnn_cell.linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s + mask)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, tf.pack([-1, attn_length, 1, 1])) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)

        if decoder_length is not None:
            decoder_length = math_ops.to_int32(decoder_length)

        if decoder_length is not None:  # Prepare variables
            zero_output = array_ops.zeros(tf.pack([batch_size, cell.output_size]),
                                          dtype if dtype is not None else initial_state.dtype)
            min_sequence_length = math_ops.reduce_min(decoder_length)
            max_sequence_length = math_ops.reduce_max(decoder_length)

        prev = None
        HACK = [attns]
        for time, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, inp)
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pylint: disable=cell-var-from-loop
            def call_cell():
                # Merge input and previous attentions into one vector of the right size.
                x = rnn_cell.linear([inp] + HACK[0], cell.input_size, True)
                cell_output, cell_state = cell(x, state)
                # Run the attention mechanism.
                if time == 0 and initial_state_attention:
                    with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                        HACK[0] = attention(state)
                else:
                    HACK[0] = attention(state)

                with variable_scope.variable_scope("AttnOutputProjection"):
                    output = rnn_cell.linear([cell_output] + HACK[0], output_size, True)
                return output, cell_state

            # pylint: enable=cell-var-from-loop
            if decoder_length is not None:
                (output, state) = rnn._rnn_step(
                    time, decoder_length, min_sequence_length, max_sequence_length,
                    zero_output, state, call_cell)
            else:
                (output, state) = call_cell()
            outputs.append(output)
            if loop_function is not None:
                prev = output

    return outputs, state


def embedding_attention_decoder(decoder_inputs, decoder_length, initial_state, attention_states, attention_length,
                                cell, num_symbols, embedding_size, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_size: Size of the output vectors; if None, use output_size.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has shape
        [num_symbols]; if provided and feed_previous=True, each fed previous
        output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      dtype: The dtype to use for the RNN initial states (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
        with ops.device("/cpu:0"):
            embedding = variable_scope.get_variable("embedding",
                                                    [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp, decoder_length, initial_state, attention_states, attention_length, cell, output_size=output_size,
            num_heads=num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, encoder_length, decoder_length, cell,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size,
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None, initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = my_rnn(
            encoder_cell, encoder_inputs, dtype=dtype, sequence_length=encoder_length)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs, decoder_length, encoder_state, attention_states, encoder_length, cell,
                num_decoder_symbols, embedding_size, num_heads=num_heads,
                output_size=output_size, output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=reuse):
                outputs, state = embedding_attention_decoder(
                    decoder_inputs, decoder_length, encoder_state, attention_states, encoder_length, cell,
                    num_decoder_symbols, embedding_size, num_heads=num_heads,
                    output_size=output_size, output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention)
                return outputs + [state]

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        return outputs_and_state[:-1], outputs_and_state[-1]


def sequence_loss_by_example(logits, targets, length,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy log_perps for a sequence of logits (per example).

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with ops.op_scope(logits + targets + [length], name,
                      "sequence_loss_by_example"):
        zero_loss = tf.zeros_like(targets[0], tf.float32)
        log_perps = zero_loss

        if length is not None:
            length = math_ops.to_int32(length)

        if length is not None:
            min_sequence_length = math_ops.reduce_min(length)
            max_sequence_length = math_ops.reduce_max(length)

        for time, logit in enumerate(logits):
            if time > 0: tf.get_variable_scope().reuse_variables()
            # pylint: disable=cell-var-from-loop
            def call_cell():
                crossent = None
                target = array_ops.reshape(targets[time], [-1])
                if softmax_loss_function is None:
                    crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logit, target)
                else:
                    crossent = softmax_loss_function(logit, target)
                return crossent, crossent + log_perps
            # pylint: enable=cell-var-from-loop
            if length is not None:
                (output, log_perps) = rnn._rnn_step(
                    time, length, min_sequence_length, max_sequence_length,
                    zero_loss, log_perps, call_cell)
            else:
                (cross_ent, log_perps) = call_cell()
        if average_across_timesteps:
            log_perps = log_perps / tf.cast(length, tf.float32)

    return log_perps


def sequence_loss(logits, targets, length,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with ops.op_scope(logits + targets + [length], name, "sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, length,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, dtypes.float32)
        else:
            return cost
