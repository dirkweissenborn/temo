import tensorflow as tf


def beam_search(beam_size, num_symbols, output_projection = None):
    # log_beam_probs: list of [beam_size, 1] Tensors
    #  Ordered log probabilities of the `beam_size` best hypotheses
    #  found in each beam step (highest probability first).
    # beam_symbols: list of [beam_size] Tensors
    #  The ordered `beam_size` words / symbols extracted by the beam
    #  step, which will be appended to their corresponding hypotheses
    #  (corresponding hypotheses found in `beam_path`).
    # beam_path: list of [beam_size] Tensor
    #  The ordered `beam_size` parent indices. Their values range
    #  from [0, `beam_size`), and they denote which previous
    #  hypothesis each word should be appended to.
    log_beam_probs, beam_symbols, beam_path  = [], [], []
    def inner_beam_search(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])

        # Compute
        #  log P(next_word, hypothesis) =
        #  log P(next_word | hypothesis)*P(hypothesis) =
        #  log P(next_word | hypothesis) + log P(hypothesis)
        # for each hypothesis separately, then join them together
        # on the same tensor dimension to form the example's
        # beam probability distribution:
        # [P(word1, hypothesis1), P(word2, hypothesis1), ...,
        #  P(word1, hypothesis2), P(word2, hypothesis2), ...]

        # If TF had a log_sum_exp operator, then it would be
        # more numerically stable to use:
        #   probs = prev - tf.log_sum_exp(prev, reduction_dims=[1])
        probs = tf.log(tf.nn.softmax(prev))
        # i == 1 corresponds to the input being "<GO>", with
        # uniform prior probability and only the empty hypothesis
        # (each row is a separate example).
        if i > 1:
            probs = tf.reshape(probs + log_beam_probs[-1],
                               [-1, beam_size * num_symbols])

        # Get the top `beam_size` candidates and reshape them such
        # that the number of rows = batch_size * beam_size, which
        # allows us to process each hypothesis independently.
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        symbols = indices % num_symbols # Which word in vocabulary.
        beam_parent = indices // num_symbols # Which hypothesis it came from.

        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)
        return tf.nn.embedding_lookup(embedding, symbols)