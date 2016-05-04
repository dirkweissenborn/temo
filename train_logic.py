import util
from rnn_cell_plus import *
import random
from sklearn.utils import shuffle
from tensorflow.python.ops.rnn import rnn
import os
import numpy as np
import re


def training(FLAGS):
    embedding_size = FLAGS.input_size # todo: can be different from input_size, no?
    
    # Load data
    if FLAGS.debug:
        train = load_data(os.path.join(FLAGS.data, "debug"))
        dev = load_data(os.path.join(FLAGS.data, "debug"))
        test = load_data(os.path.join(FLAGS.data, "debug"))
        FLAGS.checkpoint = 10
        FLAGS.batch_size = len(train)
    else:
        train = load_data(os.path.join(FLAGS.data, "train"))
        dev = load_data(os.path.join(FLAGS.data, "dev"))
        test = load_data(os.path.join(FLAGS.data, "test"))

    #for datum in train[0:10]:
    #    print(datum)
    #os._exit(0)

    vocab = dict()
    for i in range(0, 12):
        vocab[i] = len(vocab)
    for sym in ["<unk>", "<padding>"]:
        vocab[sym] = len(vocab)     
    
    #if FLAGS.binary:
    #    train = filter(lambda x: x[1] != 0, train)
    #    test = filter(lambda x: x[1] != 0, test)
    #    dev = filter(lambda x: x[1] != 0, dev)

    print("#Training sequences: %d" % len(train))
    print("#Dev sequences: %d" % len(test))
    print("#Test sequences: %d" % len(test))

    # todo: have a flag to switch on trainable embeddings
    embeddings = None
    if FLAGS.one_hot:
        embeddings = np.eye(len(vocab))
        #FLAGS.embedding_size = len(vocab)
        FLAGS.input_size = len(vocab)
        embedding_size = len(vocab)
    else:
        #embeddings = tf.Variable(tf.random_normal([embedding_size, len(vocab)]),
        #                         trainable=FLAGS.trainable_embeddings)
        embeddings = np.random.randn(len(vocab), embedding_size)

    #if FLAGS.one_hot:
    train = [([embeddings[i] for i in seq], seq, y) for (seq, y) in train]
    dev = [([embeddings[i] for i in seq], seq, y) for (seq, y) in dev]
    test = [([embeddings[i] for i in seq], seq, y) for (seq, y) in test]
    #else:
    #    train = [([tf.nn.embedding_lookup(embeddings, i) for i in seq], seq, y)
    #             for (seq, y) in train]
    #    dev = [([tf.nn.embedding_lookup(embeddings, i) for i in seq], seq, y)
    #             for (seq, y) in dev]
    #    test = [([tf.nn.embedding_lookup(embeddings, i) for i in seq], seq, y)
    #             for (seq, y) in test]

    def max_length(sentences, max_l = 0):
        for s in sentences:
            l = len(s[0])
            max_l = max(l, max_l)
        return max_l

    max_l = max_length(train)
    max_l = max_length(dev, max_l)
    max_l = max_length(test, max_l)

    l2_lambda = FLAGS.l2_lambda
    learning_rate = FLAGS.learning_rate
    mem_size = FLAGS.mem_size
    rng = random.Random(FLAGS.seed)
    batch_size = FLAGS.batch_size

    accuracies = []
    inp, ids, lengths = None, None, None
    for run_id in range(FLAGS.runs):
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.set_random_seed(rng.randint(0, 10000))
            rng2 = random.Random(rng.randint(0, 10000))

            cell = None
            input_size = embedding_size
            #if FLAGS.embedding_mode == "combined":
            #    input_size = embedding_size + task_embeddings.shape[1]
            if FLAGS.cell == 'LSTM':
                cell = BasicLSTMCell(mem_size, input_size=input_size)
            elif FLAGS.cell == 'GRU':
                cell = GRUCell(mem_size, input_size)
            elif FLAGS.cell == 'MORU':
                biases = FLAGS.moru_op_biases
                if biases is not None:
                    biases = map(lambda s: float(s), biases.split(","))
                ops = FLAGS.moru_ops.split(",")
                print(ops)
                cell = MORUCell.from_op_names(ops, biases, mem_size, input_size, FLAGS.moru_op_ctr)


            nclasses = 2 if FLAGS.binary else 5
            model = create_model(max_l, l2_lambda, learning_rate, cell, embeddings, FLAGS.embedding_mode,
                                 FLAGS.keep_prob, nclasses)
            tf.get_variable_scope().reuse_variables()
            op_weights = [w.outputs[0] for w in tf.get_default_graph().get_operations()
                          if not "grad" in w.name and w.name[:-2].endswith("op_weight") and FLAGS.cell == 'MORU']
            def evaluate(ds):
                inp, ids, lengths = None, None, None
                e_off = 0
                accuracy = 0.0
                op_weights_monitor = {w.name[-11:]:[] for w in op_weights}
                while e_off < len(ds):
                    inp, ids, lengths = batchify(ds[e_off:e_off + batch_size],
                                                 vocab["<padding>"],
                                                 inp, ids, lengths,
                                                 max_length=max_l,
                                                 max_batch_size=batch_size)
                    size = min(len(ds) - e_off, batch_size)
                    allowed_conds = ["/cond_%d/" % i for i in range(np.min(lengths))]
                    current_weights = [w for w in op_weights if any(c in w.name for c in allowed_conds)]
                    random.shuffle(current_weights)
                    result = sess.run([model["probs"]] + current_weights[:10],
                                   feed_dict={model["inp"]: inp[:,:size],
                                              model["ids"]: ids[:,:size],
                                              model["lengths"]: lengths[:size]})
                    y = encode_labels(ds[e_off:e_off + batch_size], FLAGS.binary)
                    accuracy += np.sum(np.equal(np.argmax(result[0], axis=1), y))
                    for probs, w in zip(result[1:], current_weights):
                        op_weights_monitor[w.name[-11:]].extend(probs.tolist())

                    e_off += batch_size

                accuracy = accuracy / len(ds)

                for k,v in op_weights_monitor.items():
                    hist, _ = np.histogram(np.array(v), bins=5,range=(0.0,1.0))
                    hist = (hist * 1000) / np.sum(hist)
                    print(k, hist.tolist())

                return accuracy

            saver = tf.train.Saver(tf.trainable_variables())
            sess.run(tf.initialize_all_variables())

            shuffled = shuffle(list(train), random_state=rng2.randint(0, 1000))

            loss = 0.0
            epochs = 0
            i = 0
            offset = 0

            converged = False
            accuracy = 0.0
            train_accuracy = 0.0
            while not converged:
                inp, ids, lengths = batchify(shuffled[offset:offset+batch_size],
                                              vocab["<padding>"],
                                              inp, ids, lengths,
                                              max_length=max_l,
                                              max_batch_size=batch_size)


                train_labels = encode_labels(shuffled[offset:offset+batch_size], FLAGS.binary)
                l, ps, _ = sess.run([model["loss"], model["probs"], model["update"]],
                                    feed_dict={model["inp"]: inp,
                                               model["ids"]: ids,
                                               model["lengths"]: lengths,
                                               model["y"]: train_labels})

                train_accuracy += np.sum(np.equal(np.argmax(ps, axis=1), train_labels))

                loss += l
                i += 1
                offset += batch_size

                if offset+batch_size > len(shuffled):
                    offset = 0
                    epochs += 1
                    print("%d epochs done!" % epochs)
                    shuffled = shuffle(shuffled, random_state=rng2.randint(0, 1000))

                if FLAGS.checkpoint < 0 and offset == 0 or i == FLAGS.checkpoint:
                    loss /= i
                    sess.run(model["keep_prob"].assign(1.0))
                    acc = evaluate(dev)
                    train_accuracy /= (i * batch_size)
                    sess.run(model["keep_prob"].initializer)
                    print("Train loss: %.3f, Accuracy: %.3f, Accuracy on Dev: %.3f" % (loss, train_accuracy, acc))
                    i = 0
                    loss = 0.0
                    train_accuracy = 0.0
                    if acc > accuracy:
                        accuracy = acc
                        saver.save(sess, '/tmp/my-model')
                    else:
                        if FLAGS.learning_rate_decay < 1.0:
                            print("Decaying learning rate.")
                            lr = tf.get_variable("model/lr")
                            sess.run(lr.assign(lr * FLAGS.learning_rate_decay))
                        if epochs >= FLAGS.min_epochs:
                            break

            saver.restore(sess, '/tmp/my-model')
            sess.run(model["keep_prob"].assign(1.0))
            acc = evaluate(test)
            accuracies.append(acc)

            print('######## Run %d #########' % run_id)
            print('Test Accuracy: %.4f' % acc)
            print('########################')

    mean_accuracy = sum(accuracies) / len(accuracies)

    def s_dev(mean, pop):
        d = 0.0
        for el in pop:
            d += (mean - el) * (mean - el)
        return math.sqrt(d / len(pop))

    if FLAGS.result_file:
        with open(FLAGS.result_file, 'w') as f:
            f.write('Test Accuracy: %.4f (%.4f)\n\n' % (mean_accuracy, s_dev(mean_accuracy, accuracies)))
            f.write("Configuration: \n")
            f.write(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))

    print('######## Overall #########')
    print('Test Accuracy: %.4f (%.4f)' % (mean_accuracy, s_dev(mean_accuracy, accuracies)))
    print('########################')


def load_data(path):
    data = []
    f = open(path+".txt", "r")
    for line in f:        
        datum = line[:-1].split(" ")
        seq = [int(x) for x in datum[1:]]
        label = int(datum[0])
        # fixme: sanity check
        #if np.random.random_integers(0, 1) > 0:
        #    label = 1
        #else:
        #    label = 0
        data.append((seq, label))
    return data


def encode_labels(batch, binary):
    Y = np.zeros((len(batch))).astype('int64')
    #for j, ((_, _), y) in enumerate(batch):
    for j, (_, _, y) in enumerate(batch):
        #if binary:
        #    Y[j] = min(y + 2, 3) / 2
        #else:
        #    Y[j] = y + 2
        Y[j] = y
    return Y


# create batch given example sentences
def batchify(batch, padding, inp, ids, lengths, max_length=None, max_batch_size=None):
    embedding_size = FLAGS.input_size

    inp = np.zeros([max_length, max_batch_size, embedding_size]) if inp is None else inp
    ids = np.ones([max_length, max_batch_size], np.int32) if ids is None else ids
    lengths = np.zeros([max_batch_size], np.int32) if lengths is None else lengths

    for i in range(len(batch)):
        seq, seq_ids, label = batch[i]
        lengths[i] = len(seq)
        #print(seq)
        #print(seq_ids)
        #print(label)
        for j in range(lengths[i]):
            for k in range(embedding_size):
                inp[j][i] = seq[j]
            ids[j][i] = seq_ids[j]
    return inp, ids, lengths


# Create Model
def create_model(length, l2_lambda, learning_rate, cell, embeddings, embedding_mode, keep_prob, nclasses,
                 initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    with tf.variable_scope("model", initializer=initializer):
        embedding_size = cell.input_size - embeddings.shape[1] if embedding_mode == "combined" else cell.input_size
        inp = tf.placeholder(tf.float32, [length, None, embedding_size])
        learning_rate = tf.get_variable("lr", (), tf.float32, tf.constant_initializer(learning_rate), trainable=False)
        batch_size = tf.cast(tf.gather(tf.shape(inp), [1]), tf.float32)
        ids = tf.placeholder(tf.int32, [length, None])
        lengths = tf.placeholder(tf.int32, [None])

        keep_prob_var = tf.get_variable("keep_prob", (), initializer=tf.constant_initializer(keep_prob, tf.float32),
                                        trainable=False)
        if keep_prob < 1.0:
            cell = DropoutWrapper(cell, keep_prob_var, keep_prob_var)

        def my_rnn(inp, ids, cell, length, embeddings, rev=False, init_state=None):
            if ids:
                E = tf.get_variable("E_w", initializer=tf.identity(embeddings), trainable=True)
                if inp:
                    inp = tf.concat(2, [tf.nn.embedding_lookup(E, ids), inp])
                else:
                    inp = tf.nn.embedding_lookup(E, ids)

            if init_state is None:
                init_state = tf.get_variable("init_state", [cell.state_size], tf.float32)
                batch_size = tf.gather(tf.shape(inp), [1])
                init_state = tf.tile(init_state, batch_size)
                init_state = tf.reshape(init_state, [-1, cell.state_size])

            inps = tf.split(0, length, inp)
            for i in range(length):
                inps[i] = tf.squeeze(inps[i], [0])
                
            print(cell, inps, init_state)    
            _, final_state = rnn(cell, inps, init_state, sequence_length=lengths)
            out = tf.slice(final_state, [0, 0], [-1, cell.output_size])
            return out

        with tf.variable_scope("encoder_fw", initializer=initializer):
            h = my_rnn(None if embedding_mode == "tuned" else inp, None if embedding_mode == "fixed" else ids, cell,
                         length, embeddings)

        scores = tf.contrib.layers.fully_connected(h, nclasses, weight_init=None)
        probs = tf.nn.softmax(scores)
        y = tf.placeholder(tf.int64, [None])

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(scores, y)) / batch_size
        train_params = tf.trainable_variables()
        if l2_lambda > 0.0:
            l2_loss = l2_lambda * tf.reduce_sum(array_ops.pack([tf.nn.l2_loss(t)
                                                                for t in train_params if "E_w" not in t.name]))
            loss = loss + l2_loss

        update = tf.train.AdamOptimizer(learning_rate, beta1=0.0).minimize(loss, var_list=train_params,
                                                                           gate_gradients=2)
    return {"inp": inp, "ids": ids, "lengths": lengths, "y": y,
            "probs": probs, "scores": scores, "keep_prob": keep_prob_var,
            "loss": loss, "update": update}


if __name__ == "__main__":
    # data loading specifics
    tf.app.flags.DEFINE_string('data', 'data/logic', 'data dir of propositional logic unit test.')

    # model
    tf.app.flags.DEFINE_integer("input_size", 16, "input size of model")
    tf.app.flags.DEFINE_integer("mem_size", 16, "hidden size of model")

    # training
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
    tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization raten (only for batch training).")
    tf.app.flags.DEFINE_float("learning_rate_decay", 1.0,
                              "Learning rate decay when loss on validation set does not improve.")
    tf.app.flags.DEFINE_integer("batch_size", 50, "Number of examples per batch.")
    tf.app.flags.DEFINE_integer("min_epochs", 100, "Minimum num of epochs")
    tf.app.flags.DEFINE_string("cell", 'GRU', "'LSTM', 'GRU', 'RNN', 'MaxLSTM', 'MaxGRU', 'MaxRNN'")
    tf.app.flags.DEFINE_integer("seed", 12345, "Random seed.")
    tf.app.flags.DEFINE_integer("runs", 10, "How many runs.")
    tf.app.flags.DEFINE_integer("checkpoint", 10, "checkpoint at.")
    tf.app.flags.DEFINE_boolean('binary', True, 'binary evaluation')
    tf.app.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")
    tf.app.flags.DEFINE_string("result_file", None, "Where to write results.")
    tf.app.flags.DEFINE_string("moru_ops", 'max,mul,keep,replace', "operations of moru cell.")
    tf.app.flags.DEFINE_string("moru_op_biases", None, "biases of moru operations at beginning of training. "
                                                       "Defaults to 0 for each.")
    tf.app.flags.DEFINE_integer("moru_op_ctr", None, "Size of op ctr. By default ops are controlled by current input"
                                                 "and previous state. Given a positive integer, an additional"
                                                 "recurrent op ctr is introduced in MORUCell.")

    tf.app.flags.DEFINE_boolean("debug", False, "Train and test model on a tiny debug corpus.")

    tf.app.flags.DEFINE_string('embedding_mode', 'fixed', 'fixed|tuned|combined')
    tf.app.flags.DEFINE_boolean("one_hot", True, "Whether to use one hot encodings of symbols.")
    tf.app.flags.DEFINE_boolean("trainable_embeddings", False, "Whether non-one-hot embeddings can be trained.")

    FLAGS = tf.app.flags.FLAGS
    kwargs = None
    #if FLAGS.embedding_format == "glove":
    #    kwargs = {"vocab_size": 2196017, "dim": 300}

    #print("Loading embeddings...")
    #e = util.load_embeddings(FLAGS.embedding_file, FLAGS.embedding_format)
    #print("Done.")

    import json
    print("Configuration: ")
    print(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))
    training(FLAGS)

