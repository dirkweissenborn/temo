# THIS IS AN ADAPTION FROM THE ORIGINAL translate.py OF TF

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.
See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from wmt14 import data_utils
from wmt14 import translation_model


tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 30000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 30000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp/wmt14", "Training directory.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Type of cell to use.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_length", 50, "limit length of sentences.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("test", False, "Decode test set.")
tf.app.flags.DEFINE_boolean("no_unk", False, "Do not use <UNK> during decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("attention", False, "Use attention.")
tf.app.flags.DEFINE_string("device", "/cpu:0", "Run on device.")
tf.app.flags.DEFINE_string("decode_out", "/tmp/decoded", "Directory of decoding output.")


FLAGS = tf.app.flags.FLAGS


def read_data(source_path, target_path, max_length, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = []
    m_length = 0
    print("Reading from source: %s ; and target: %s" % (source_path, target_path))
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            counter = 0
            source, target = source_file.readline(), target_file.readline()
            while source and target and (not max_size or counter < max_size):
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                if source_ids and target_ids and len(source_ids) < max_length and len(target_ids) < max_length:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    m_length = max(len(target_ids), max(len(source_ids), m_length))
                    data_set.append([source_ids, target_ids])
                    if max_size is not None and max_size >  0 and counter >= max_size:
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set, m_length


def create_model(session, forward_only, max_length):
    """Create translation model and initialize or load parameters in session."""
    with tf.device(FLAGS.device):
        model = translation_model.TranslationModel(
            FLAGS.en_vocab_size, FLAGS.fr_vocab_size, max_length,
            FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, cell_type=FLAGS.cell_type,
            forward_only=forward_only, attention=FLAGS.attention)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    print("Preparing WMT data in %s" % FLAGS.data_dir)
    en_train, fr_train, en_dev, fr_dev, en_test, fr_test, _, _ = data_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))) as sess:
        # Read data.
        print("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set, max_length = read_data(en_dev, fr_dev, FLAGS.max_length)
        train_set, l = read_data(en_train, fr_train, FLAGS.max_length, FLAGS.max_train_data_size)
        max_length = max(l, max_length)
        train_total_size = len(train_set)

        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, max_length)

        num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
        print("Num params: %d" % num_params)
        print("Num params (without embeddings): %d" % (num_params -
                                                       (FLAGS.en_vocab_size + 2*FLAGS.fr_vocab_size) * FLAGS.size +
                                                        FLAGS.fr_vocab_size))


        # This is the training loop.
        step_time, loss, norm = 0.0, 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length = model.get_batch(train_set)
            step_norm, step_loss, _ = model.step(sess, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            norm += step_norm / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f norm %.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity, norm))
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length = model.get_batch(dev_set)
                _, eval_loss, _ = model.step(sess, encoder_inputs, rev_encoder_inputs, decoder_inputs,
                                             encoder_length, decoder_length, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and eval_loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(eval_loss)
                print("  eval perplexity %.2f" % eval_ppx)
                sys.stdout.flush()


def decode():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Create model and load parameters.
        FLAGS.batch_size = 1
        model = create_model(sess, True, FLAGS.max_length)
        if FLAGS.no_unk:
            model.set_no_unk(sess)
        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en" % FLAGS.en_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.fr" % FLAGS.fr_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length = \
                model.get_batch([(token_ids, [data_utils.PAD_ID] * 2 * len(token_ids))])
            # Get output symbols for the sentence.
            outputs = model.decode(sess, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, True)
            print(outputs)
            for i in range(len(outputs) // 2):
                for j in range(outputs[i*2].shape[0]):
                    output = outputs[i*2][j].tolist()
                    prob = outputs[i*2+1][j]
                    # If there is an EOS symbol in outputs, cut them at that point.
                    #if data_utils.EOS_ID in outputs:
                    #    output = output[:output.index(data_utils.EOS_ID)]
                    # Print out French sentence corresponding to outputs.
                    print("%s (%.3f)" % (" ".join([tf.compat.as_str(rev_fr_vocab[o]) for o in output]), prob))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def decode_testset():
    _, _, _, _, en_test, fr_test, en_vocab_path, fr_vocab_path = data_utils.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size, only_test=True)
    en_vocab, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    test_set, max_length = read_data(en_test, fr_test, FLAGS.max_length)
    if not os.path.exists(FLAGS.decode_out):
        os.mkdir(FLAGS.decode_out)
    src_file = os.path.join(FLAGS.decode_out, "src.txt")
    ref_file = os.path.join(FLAGS.decode_out, "ref.txt")
    trs_file = os.path.join(FLAGS.decode_out, "trs.txt")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Create model and load parameters.
        FLAGS.batch_size = 1
        model = create_model(sess, True, FLAGS.max_length)
        if FLAGS.no_unk:
            model.set_no_unk(sess)
        print("Start decoding test file to %s!" % FLAGS.decode_out)
        size = len(test_set)
        with open(src_file, 'w') as sf, open(ref_file, 'w') as rf, open(trs_file, 'w') as xf:
            i = 0
            for (en_tokens, fr_tokens) in test_set:
                if not FLAGS.no_unk or data_utils.UNK_ID not in en_tokens:
                    encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length = model.get_batch([(en_tokens, [data_utils.PAD_ID] *
                                                                                                            max(max_length, 2 * len(en_tokens)))])

                    outputs = model.decode(sess, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, True)
                    best = (None, -float("inf"))
                    for i in range(len(outputs) // 2):
                        for j in range(outputs[i*2].shape[0]):
                            prob = outputs[i*2+1][j]
                            if prob > best[1]:
                                output = outputs[i*2][j].tolist()
                                best = (output, prob)

                    output = best[0]
                    trs = " ".join([tf.compat.as_str(rev_fr_vocab[o]) for o in output])
                    src = " ".join([tf.compat.as_str(rev_en_vocab[o]) for o in en_tokens])
                    ref = " ".join([tf.compat.as_str(rev_fr_vocab[o]) for o in fr_tokens])

                    sf.write(src+"\n")
                    xf.write(trs+"\n")
                    rf.write(ref+"\n")
                    sf.flush()
                    xf.flush()
                    rf.flush()
                i += 1
                if i % 10 == 0:
                    sys.stdout.write("\r%.1f%%" % ((i*100.0) / size))
                    sys.stdout.flush()
        print("")
        print("DONE!")


def self_test():
    """Test the translation model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 layers of 32.
        with tf.device(FLAGS.device):
            model = translation_model.TranslationModel(10, 10, 6, 32, 2,
                                                       5.0, 32, 0.3, 0.99, num_samples=8,
                                                       cell_type=FLAGS.cell_type, attention=FLAGS.attention)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = [([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6]), ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])]
        for _ in range(5):  # Train the fake model for 5 steps.
            encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length = model.get_batch(data_set)
            model.step(sess, encoder_inputs, rev_encoder_inputs, decoder_inputs, encoder_length, decoder_length, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.test:
        decode_testset()
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()