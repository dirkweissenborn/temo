import util
import nltk
import numpy as np
from moru_cell import *
import random
from sklearn.utils import shuffle
import os
from tensorflow.python.ops.rnn import rnn
import time
import sys
import functools


def training(embeddings, FLAGS):
    # Load data
    print("Preparing data...")
    embedding_size = embeddings.vectors.shape[1]
    trainA, trainB, devA, devB, testA, testB, y_scores, vocab, oo_vocab = load_data(FLAGS.data, embeddings)

    # embeddings
    task_embeddings = np.random.normal(size=[len(vocab)+len(oo_vocab), embedding_size]).astype("float32")
    for w, i in vocab.items():
        task_embeddings[len(oo_vocab) + i] = embeddings[w]

    # accumulate counts for buckets
    def max_length(sentences, max_l=0):
        for s in sentences:
            l = len(s)
            max_l = max(l, max_l)
        return max_l

    max_l = max_length(trainA)
    max_l = max_length(trainB, max_l)
    max_l = max_length(devA, max_l)
    max_l = max_length(devB, max_l)
    max_l = max_length(testA, max_l)
    max_l = max_length(testB, max_l)

    print("Done.")

    l2_lambda = FLAGS.l2_lambda
    learning_rate = FLAGS.learning_rate
    h_size = FLAGS.h_size
    mem_size = FLAGS.mem_size
    rng = random.Random(FLAGS.seed)
    batch_size = FLAGS.batch_size

    accuracies = []

    idsA, idsB, lengthsA, lengthsB = None, None, None, None

    for run_id in range(FLAGS.runs):
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.set_random_seed(rng.randint(0, 10000))
            rng2 = random.Random(rng.randint(0, 10000))

            cellA = cellB = None
            if FLAGS.cell == 'LSTM':
                cellA = cellB = BasicLSTMCell(mem_size, embedding_size)
            elif FLAGS.cell == 'GRU':
                cellA = cellB = GRUCell(mem_size, embedding_size)
            elif FLAGS.cell == 'MORU':
                biases = FLAGS.moru_op_biases
                if biases is not None:
                    biases = map(lambda s: float(s), biases.split(","))
                ops = FLAGS.moru_ops.split(",")
                cellA = cellB = MORUCell.from_op_names(ops, biases, mem_size, embedding_size, FLAGS.moru_op_ctr)

            tunable_embeddings, fixed_embeddings = task_embeddings, None
            if FLAGS.embedding_mode == "fixed":
                tunable_embeddings, fixed_embeddings = task_embeddings[:len(oo_vocab)], task_embeddings[len(oo_vocab):]

            with tf.device(FLAGS.device):
                model = create_model(max_l, l2_lambda, learning_rate, h_size, cellA, cellB, tunable_embeddings,
                                     fixed_embeddings, FLAGS.keep_prob)

            tf.get_variable_scope().reuse_variables()

            op_weights = [w.outputs[0] for w in tf.get_default_graph().get_operations()
                          if not "grad" in w.name and w.name[:-2].endswith("op_weight") and FLAGS.cell == 'MORU']

            def evaluate(dsA, dsB, labels):
                idsA, idsB, lengthsA, lengthsB = None, None, None, None
                e_off = 0
                accuracy = 0.0
                y = encode_labels(labels)
                op_weights_monitor = {ops[int(w.name[-3:-2])]:[] for w in op_weights}

                while e_off < len(dsA):
                    idsA, idsB, lengthsA, lengthsB = batchify(dsA[e_off:e_off + batch_size],
                                                              dsB[e_off:e_off + batch_size],
                                                              idsA, idsB, lengthsA, lengthsB,
                                                              max_length=max_l,
                                                              max_batch_size=batch_size)
                    size = min(len(dsA) - e_off, batch_size)
                    allowed_conds = ["/cond_%d/" % (2*i) for i in range(min(np.min(lengthsA), np.min(lengthsB)))]
                    current_weights = [w for w in op_weights if any(c in w.name for c in allowed_conds)]
                    result = sess.run([model["probs"]] + current_weights[:10],
                                    feed_dict={model["idsA"]: idsA[:,:size],
                                               model["idsB"]: idsB[:,:size],
                                               model["lengthsA"]: lengthsA[:size],
                                               model["lengthsB"]: lengthsB[:size]})
                    accuracy += np.sum(np.equal(np.argmax(result[0], axis=1), y[e_off:e_off+size]))
                    e_off += size

                    for probs, w in zip(result[1:], current_weights):
                        op_weights_monitor[ops[int(w.name[-3:-2])]].extend(probs.tolist())

                for k,v in op_weights_monitor.items():
                    hist, _ = np.histogram(np.array(v), bins=5,range=(0.0,1.0))
                    hist = (hist * 1000) / np.sum(hist)
                    print(k, hist.tolist())

                accuracy /= e_off
                return accuracy

            saver = tf.train.Saver(tf.trainable_variables())
            sess.run(tf.initialize_all_variables())
            num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
            print("Num params: %d" % num_params)
            print("Num params (without embeddings): %d" % (num_params - (len(oo_vocab) + len(vocab)) * embedding_size))

            shuffledA, shuffledB, y = \
                shuffle(list(trainA), list(trainB), list(y_scores[0]), random_state=rng2.randint(0, 1000))

            offset = 0
            loss = 0.0
            epochs = 0
            i = 0
            accuracy = float("-inf")
            step_time = 0.0
            epoch_acc = 0.0
            while not FLAGS.eval:
                start_time = time.time()
                idsA, idsB, lengthsA, lengthsB = batchify(shuffledA[offset:offset+batch_size],
                                                          shuffledB[offset:offset+batch_size],
                                                          idsA, idsB, lengthsA, lengthsB,
                                                          max_length=max_l,
                                                          max_batch_size=batch_size)
                train_labels = encode_labels(y[offset:offset+batch_size])
                # update initialized embeddings only after first epoch
                update = model["update"] if epochs>= 1 else model["update_ex"]
                l, _ = sess.run([model["loss"], update],
                                feed_dict={model["idsA"]:idsA,
                                           model["idsB"]:idsB,
                                           model["lengthsA"]: lengthsA,
                                           model["lengthsB"]: lengthsB,
                                           model["y"]:train_labels})

                offset += batch_size
                loss += l
                i += 1
                step_time += (time.time() - start_time)

                sys.stdout.write("\r%.1f%% Loss: %.3f" % ((i*100.0) / FLAGS.checkpoint, loss / i))
                sys.stdout.flush()

                if offset + batch_size > len(shuffledA):
                    epochs += 1
                    shuffledA, shuffledB, y = shuffle(shuffledA, shuffledB, y, random_state=rng2.randint(0, 1000))
                    offset = 0
                    sess.run(model["keep_prob"].assign(1.0))
                    acc = evaluate(devA, devB, y_scores[1])
                    sess.run(model["keep_prob"].initializer)
                    print("\n%d epochs done! Accuracy on Dev: %.3f" % (epochs, acc))
                    if acc < epoch_acc + 1e-3:
                        print("Decaying learning-rate!")
                        lr = tf.get_variable("model/lr")
                        sess.run(lr.assign(lr * FLAGS.learning_rate_decay))
                    epoch_acc = acc

                if i == FLAGS.checkpoint:
                    loss /= i
                    sess.run(model["keep_prob"].assign(1.0))
                    acc = evaluate(devA, devB, y_scores[1])
                    sess.run(model["keep_prob"].initializer)

                    print("\nTrain loss: %.3f, Accuracy on Dev: %.3f, Step Time: %.3f" % (loss, acc, step_time/i))
                    i = 0
                    step_time = 0.0
                    loss = 0.0
                    if acc > accuracy + 1e-5:
                        accuracy = acc
                        saver.save(sess, FLAGS.model_path)
                    else:
                        if epochs >= FLAGS.min_epochs:
                            break

            saver.restore(sess, FLAGS.model_path)
            sess.run(model["keep_prob"].assign(1.0))
            acc = evaluate(testA, testB, y_scores[2])
            accuracies.append(acc)
            print('######## Run %d #########' % run_id)
            print('Test Accuracy: %.4f' % acc)
            print('########################')

    mean_accuracy = sum(accuracies) / len(accuracies)

    def s_dev(mean, pop):
        d = 0.0
        for el in pop:
            d += (mean-el) * (mean-el)
        return math.sqrt(d/len(pop))

    print('######## Overall #########')
    print('Test Accuracy: %.4f (%.4f)' % (mean_accuracy,  s_dev(mean_accuracy, accuracies)))
    print('########################')

    if FLAGS.result_file:
        with open(FLAGS.result_file, 'w') as f:
            f.write('Accuracy: %.4f (%.4f)\n' % (mean_accuracy,  s_dev(mean_accuracy, accuracies)))
            f.write("Configuration: \n")
            f.write(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))

    return mean_accuracy


def load_data(loc, embeddings):
    """
    Load the SNLI dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(os.path.join(loc, 'snli_1.0_train.txt'), 'r') as f:
        for line in f:
            text = line.strip().split('\t')
            if text[0] != '-':
                trainA.append(text[5])
                trainB.append(text[6])
                trainS.append(text[0])
    with open(os.path.join(loc, 'snli_1.0_dev.txt'), 'r') as f:
        for line in f:
            text = line.strip().split('\t')
            if text[0] != '-':
                devA.append(text[5])
                devB.append(text[6])
                devS.append(text[0])
    with open(os.path.join(loc, 'snli_1.0_test.txt'), 'r') as f:
        for line in f:
            text = line.strip().split('\t')
            if text[0] != '-':
                testA.append(text[5])
                testB.append(text[6])
                testS.append(text[0])

    vocab, oo_vocab = dict(), dict()
    def encode(sentence):
        if "<s>" not in oo_vocab:
            oo_vocab["<s>"] = len(oo_vocab)
        if "</s>" not in oo_vocab:
            oo_vocab["</s>"] = len(oo_vocab)
        word_ids = [-oo_vocab["<s>"]-1]
        for w in nltk.word_tokenize(sentence.lower()):
            wv = embeddings.get(w)
            if wv is None:
                if w not in oo_vocab:
                    oo_vocab[w] = len(oo_vocab)
                word_ids.append(-oo_vocab[w]-1)
            else:
                if w not in vocab:
                    vocab[w] = len(vocab)
                word_ids.append(vocab[w])
        word_ids.append(-oo_vocab["</s>"]-1)
        return word_ids

    trainA = [encode(s) for s in trainA[1:]]
    trainB = [encode(s) for s in trainB[1:]]
    devA = [encode(s) for s in devA[1:]]
    devB = [encode(s) for s in devB[1:]]
    testA = [encode(s) for s in testA[1:]]
    testB = [encode(s) for s in testB[1:]]

    def normalize_ids(ds):
        for word_ids in ds:
            for i in range(len(word_ids)):
                word_ids[i] += len(oo_vocab)

    normalize_ids(trainA)
    normalize_ids(trainB)
    normalize_ids(devA)
    normalize_ids(devB)
    normalize_ids(testA)
    normalize_ids(testB)

    for k in oo_vocab:
        oo_vocab[k] = len(oo_vocab) - oo_vocab[k] - 1

    return trainA, trainB, devA, devB, testA, testB, [trainS[1:], devS[1:], testS[1:]], vocab, oo_vocab


def encode_labels(labels):
    Y = np.zeros([len(labels)]).astype('int64')
    for j, y in enumerate(labels):
        if y == 'neutral':
            Y[j] = 0
        elif y == 'entailment':
            Y[j] = 1
        else:
            Y[j] = 2
    return Y


#create batch given example sentences
def batchify(batchA, batchB, idsA, idsB, lengthsA, lengthsB, max_length=None, max_batch_size=None):
    idsA = np.ones([max_length, max_batch_size]) if idsA is None else idsA
    idsB = np.ones([max_length, max_batch_size]) if idsB is None else idsB

    lengthsA = np.zeros([max_batch_size], np.int32) if lengthsA is None else lengthsA
    lengthsB = np.zeros([max_batch_size], np.int32) if lengthsB is None else lengthsB

    for i in range(len(batchA)):
        lengthsA[i] = len(batchA[i])
        for j in range(len(batchA[i])):
            idsA[j][i] = batchA[i][j]

    for i in range(len(batchB)):
        lengthsB[i] = len(batchB[i])
        for j in range(len(batchB[i])):
            idsB[j][i] = batchB[i][j]

    return idsA, idsB, lengthsA, lengthsB


# Create Model
def create_model(length, l2_lambda, learning_rate, h_size, cellA, cellB, tunable_embeddings, fixed_embeddings, keep_prob,
                 initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    with tf.variable_scope("model", initializer=initializer):
        idsA = tf.placeholder(tf.int32, [length, None])
        idsB = tf.placeholder(tf.int32, [length, None])
        lengthsA = tf.placeholder(tf.int32, [None])
        lengthsB = tf.placeholder(tf.int32, [None])
        learning_rate = tf.get_variable("lr", (), tf.float32, tf.constant_initializer(learning_rate), trainable=False)

        batch_size = tf.gather(tf.shape(idsA), [1])

        keep_prob_var = tf.get_variable("keep_prob", (), initializer=tf.constant_initializer(keep_prob, tf.float32),
                                        trainable=False)

        def create_embeddings():
            #with tf.device("/cpu:0"):
            E = None
            if fixed_embeddings is not None and fixed_embeddings.shape[0] > 0:
                E = tf.get_variable("E_fix", initializer=tf.identity(fixed_embeddings), trainable=True)
            if tunable_embeddings is not None and tunable_embeddings.shape[0] > 0:
                E_tune = tf.get_variable("E_tune", initializer=tf.identity(tunable_embeddings), trainable=True)
                if E is not None:
                    E = tf.concat(0, [E_tune, E])
                else:
                    E = E_tune
            return E

        def my_rnn(ids, cell, lengths, E=None, additional_inputs=None, rev=False, init_state=None):
            inp = None
            if ids is not None:
                #with tf.device("/cpu:0"):
                inp = tf.nn.embedding_lookup(E, ids)
                if additional_inputs is not None:
                    inp = tf.concat(2, [inp, additional_inputs])
            else:
                inp = additional_inputs

            if init_state is None:
                init_state = tf.zeros([cell.state_size], tf.float32)
                init_state = tf.tile(init_state, batch_size)
                init_state = tf.reshape(init_state, [-1, cell.state_size])

            inps = tf.split(0, length, inp)
            for i in range(length):
                inps[i] = tf.squeeze(inps[i], [0])
            outs, final_state = rnn(cell, inps, init_state, sequence_length=lengths)
            last_out = tf.slice(final_state, [0, 0], [-1, cell.output_size])
            # mean pooling
            #last_out = tf.reduce_sum(tf.pack(outs), [0]) / tf.cast(tf.reshape(tf.tile(lengths, [cell.output_size]), [-1, cell.output_size]) , tf.float32)
            return last_out, final_state, outs

        if keep_prob < 1.0:
            cellA = DropoutWrapper(cellA, keep_prob_var)
            cellB = DropoutWrapper(cellB, keep_prob_var)
        E = create_embeddings()
        with tf.variable_scope("rnn", initializer=initializer):
            p, s, outsP = my_rnn(idsA, cellA, lengthsA, E)
            tf.get_variable_scope().reuse_variables()
            if cellB.state_size > cellA.state_size:
                rest_state = tf.zeros([cellB.state_size - cellA.state_size], tf.float32)
                rest_state = tf.reshape(tf.tile(rest_state, batch_size), [-1, cellB.state_size - cellA.state_size])
                s = tf.concat(1, [rest_state, s])
            h, _, outsH = my_rnn(idsB, cellB, lengthsB, E, init_state=s)

        #with tf.variable_scope("accum", initializer=initializer):
         #   p, s, _ = my_rnn(None, GRUCell(h_size, cellA.output_size), lengthsA, E, additional_inputs=tf.pack(outsP))
         #   tf.get_variable_scope().reuse_variables()
         #   h, _, _ = my_rnn(None, GRUCell(h_size, cellB.output_size), lengthsB, E, init_state=s, additional_inputs=tf.pack(outsH))

        h = tf.concat(1, [p, h, tf.abs(p-h)])
        h = tf.contrib.layers.fully_connected(h, h_size, activation_fn=lambda x: tf.maximum(0.0, x), weight_init=None)
        h = tf.contrib.layers.fully_connected(h, h_size, activation_fn=lambda x: tf.maximum(0.0, x), weight_init=None)
        scores = tf.contrib.layers.fully_connected(h, 3, weight_init=None)
        probs = tf.nn.softmax(scores)
        y = tf.placeholder(tf.int64, [None])

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(scores, y)) / tf.cast(batch_size, tf.float32)
        train_params = tf.trainable_variables()
        if l2_lambda > 0.0:
            l2_loss = l2_lambda * tf.reduce_sum(array_ops.pack([tf.nn.l2_loss(t) for t in train_params]))
            loss = loss+l2_loss

    grads = tf.gradients(loss, train_params)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    grads_params = list(zip(grads, train_params))
    grads_params_ex_emb = [(g,p) for (g,p) in grads_params if not p.name.endswith("E_fix")]

    update = tf.train.AdamOptimizer(learning_rate, beta1=0.0).apply_gradients(grads_params)
    update_exclude_embeddings = tf.train.AdamOptimizer(learning_rate, beta1=0.0).apply_gradients(grads_params_ex_emb)
    return {"idsA":idsA, "idsB":idsB, "lengthsA":lengthsA, "lengthsB":lengthsB, "y":y,
            "probs":probs, "scores":scores,"keep_prob": keep_prob_var,
            "loss":loss, "update":update, "update_ex":update_exclude_embeddings}


if __name__ == "__main__":
    # data loading specifics
    tf.app.flags.DEFINE_string('data', 'data/snli_1.0/', 'data dir of SNLI.')
    tf.app.flags.DEFINE_string('embedding_format', 'prepared', 'glove|word2vec_bin|word2vec|dict|prepared')
    tf.app.flags.DEFINE_string('embedding_file', 'snli_embeddings.pkl', 'path to embeddings')

    # model
    tf.app.flags.DEFINE_integer("mem_size", 200, "hidden size of model")
    tf.app.flags.DEFINE_integer("h_size", 200, "size of interaction")

    # training
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
    tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization raten (only for batch training).")
    tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
    tf.app.flags.DEFINE_integer("batch_size", 50, "Number of examples per batch.")
    tf.app.flags.DEFINE_integer("min_epochs", 3, "Minimum num of epochs")
    tf.app.flags.DEFINE_string("cell", 'MORU', "'LSTM', 'GRU', 'MORU'")
    tf.app.flags.DEFINE_integer("seed", 12345, "Random seed.")
    tf.app.flags.DEFINE_integer("runs", 10, "How many runs.")
    tf.app.flags.DEFINE_string('embedding_mode', 'fixed', 'fixed|tuned|combined')
    tf.app.flags.DEFINE_integer('tunable_dim', 10,
                                'number of dims for tunable embeddings if embedding mode is combined')
    tf.app.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")
    tf.app.flags.DEFINE_integer('checkpoint', 1000, 'number of batches until checkpoint.')
    tf.app.flags.DEFINE_integer('num_copies', 1, 'number of copies for associative RNN.')
    tf.app.flags.DEFINE_integer('num_read_keys', 0, 'number of additional read keys for associative RNN.')
    tf.app.flags.DEFINE_string("result_file", None, "Where to write results.")
    tf.app.flags.DEFINE_string("moru_ops", 'max,mul,keep,replace,diff,min,forget', "operations of moru cell.")
    tf.app.flags.DEFINE_string("moru_op_biases", None, "biases of moru operations at beginning of training. "
                                                       "Defaults to 0 for each.")
    tf.app.flags.DEFINE_integer("moru_op_ctr", None, "Size of op ctr. By default ops are controlled by current input"
                                                     "and previous state. Given a positive integer, an additional"
                                                     "recurrent op ctr is introduced in MORUCell.")
    tf.app.flags.DEFINE_boolean('eval', False, 'only evaluation')
    tf.app.flags.DEFINE_string('model_path', '/tmp/snli-model', 'only evaluation')
    tf.app.flags.DEFINE_string('device', '/gpu:0', 'device to run on')

    FLAGS = tf.app.flags.FLAGS

    kwargs = None
    if FLAGS.embedding_format == "glove":
        kwargs = {"vocab_size": 2196017, "dim": 300}

    print("Loading embeddings...")
    e = util.load_embeddings(FLAGS.embedding_file, FLAGS.embedding_format)
    print("Done.")


    import json
    print("Configuration: ")
    print(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))
    training(e, FLAGS)
