import tensorflow as tf
import web.embeddings
import web.embedding
import nltk
import numpy as np
from moru_cell import *
import random
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr
import os
from tensorflow.python.ops.rnn import rnn

def training(embeddings, FLAGS):
    # Load data
    train, dev, test, y_scores = load_data(FLAGS.data)
    embedding_size = embeddings.vectors.shape[1]

    # Encode data
    zero_wv = np.zeros(embedding_size)
    def encode(sentence, vocab, fill_vocab=False):
        words = []
        word_ids = []
        if "<unk>" not in vocab:
            vocab["<unk>"] = len(vocab)
        if "<padding>" not in vocab:
            vocab["<padding>"] = len(vocab)
        for w in nltk.word_tokenize(sentence.lower()):
            if fill_vocab and w not in vocab:
                vocab[w] = len(vocab)
            wv = embeddings.get(w, zero_wv)
            words.append(wv)
            word_ids.append(vocab.get(w, vocab["<unk>"]))
        return words, word_ids

    vocab = dict()
    trainA, trainB = map(lambda s: encode(s, vocab, True), train[0]), map(lambda s: encode(s, vocab, True), train[1])
    devA, devB = map(lambda s: encode(s,vocab), dev[0]), map(lambda s: encode(s,vocab), dev[1])
    testA, testB = map(lambda s: encode(s,vocab), test[0]), map(lambda s: encode(s,vocab), test[1])
    
    # embeddings
    task_embeddings = None
    if FLAGS.embedding_mode != "combined":
        task_embeddings = np.zeros((len(vocab), embedding_size), np.float32)
        for w, i in vocab.iteritems():
            e = embeddings.get(w, embeddings.get(w.lower()))
            if e is None:
                print("Not in embeddings: " + w)
                if FLAGS.embedding_mode == "tuned":
                    e = np.random.uniform(-0.05, 0.05, embedding_size).astype("float32")
                else:
                    e = np.zeros((embedding_size,), np.float32)
            task_embeddings[i] = e
    else:
        task_embeddings = np.random.uniform(-0.05, 0.05, len(vocab) * FLAGS.tunable_dim)
        task_embeddings = task_embeddings.reshape((len(vocab), FLAGS.tunable_dim)).astype("float32")

    # accumulate counts for buckets
    def max_length(sentences, max_l = 0):
        for s in sentences:
            l = len(s[0])
            max_l = max(l, max_l)
        return max_l

    max_l = max_length(trainA)
    max_l = max_length(trainB, max_l)
    max_l = max_length(devA, max_l)
    max_l = max_length(devB, max_l)
    max_l = max_length(testA, max_l)
    max_l = max_length(testB, max_l)


    l2_lambda = FLAGS.l2_lambda
    learning_rate = FLAGS.learning_rate
    h_size = FLAGS.h_size
    mem_size = FLAGS.mem_size
    rng = random.Random(FLAGS.seed)
    batch_size = FLAGS.batch_size

    pearsons = []
    spearmans = []
    mses = []

    tA, tB, idsA, idsB, lengthsA, lengthsB = None, None, None, None, None, None

    for run_id in xrange(FLAGS.runs):
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.set_random_seed(rng.randint(0, 10000))
            rng2 = random.Random(rng.randint(0, 10000))

            cell = None
            if FLAGS.cell == 'LSTM':
                cell = BasicLSTMCell(mem_size, embedding_size)
            elif FLAGS.cell == 'GRU':
                cell = GRUCell(mem_size, embedding_size)
            elif FLAGS.cell == 'MORU':
                cell = MORUCell(mem_size, embedding_size)

            model = create_model(max_l, l2_lambda, learning_rate, h_size, cell, task_embeddings,
                                 FLAGS.embedding_mode, FLAGS.keep_prob)
            tf.get_variable_scope().reuse_variables()

            def evaluate(batchA, batchB, _scores):
                tA, tB, idsA, idsB, lengthsA, lengthsB = None, None, None, None, None, None
                e_off = 0
                ps = np.zeros((len(batchA), 5))
                while e_off < len(batchA):
                    tA, tB, idsA, idsB, lengthsA, lengthsB = batchify(batchA[e_off:e_off+batch_size],
                                                                      batchB[e_off:e_off+batch_size],
                                                                      vocab["<padding>"],
                                                                      tA, tB, idsA, idsB, lengthsA, lengthsB,
                                                                      max_length=max_l, max_batch_size=batch_size)
                    size = min(len(batchA)-e_off, batch_size)
                    ps1 = sess.run(model["probs"],
                                    feed_dict={model["inpA"]: tA[:,:size],
                                               model["inpB"]: tB[:,:size],
                                               model["idsA"]: idsA[:,:size],
                                               model["idsB"]: idsB[:,:size],
                                               model["lengthsA"]: lengthsA[:size],
                                               model["lengthsB"]: lengthsB[:size]})
                    ps[e_off:e_off+batch_size] = ps1
                    e_off += batch_size

                r = np.arange(1,6)
                yhat = np.dot(ps, r)
                pr = pearsonr(yhat, _scores)[0]
                sr = spearmanr(yhat, _scores)[0]
                se = mse(yhat, _scores)
                return pr, sr, se

            saver = tf.train.Saver(tf.trainable_variables())
            sess.run(tf.initialize_all_variables())

            shuffledA, shuffledB, y = \
                shuffle(list(trainA), list(trainB), list(y_scores[0]), random_state=rng2.randint(0, 1000))

            offset = 0
            loss = 0.0
            epochs = 0
            i = 0
            pearson = float("-inf")
            converged = False
            while not converged:
                tA, tB, idsA, idsB, lengthsA, lengthsB = batchify(shuffledA[offset:offset+batch_size],
                                                                  shuffledB[offset:offset+batch_size],
                                                                  vocab["<padding>"],
                                                                  tA, tB, idsA, idsB, lengthsA, lengthsB,
                                                                  max_length=max_l,
                                                                  max_batch_size=batch_size)
                train_labels = encode_labels(y[offset:offset+batch_size])
                l, _ = sess.run([model["loss"], model["update"]],
                                feed_dict={model["inpA"]:tA,
                                           model["inpB"]:tB,
                                           model["idsA"]:idsA,
                                           model["idsB"]:idsB,
                                           model["lengthsA"]: lengthsA,
                                           model["lengthsB"]: lengthsB,
                                           model["y"]:train_labels})

                offset += batch_size
                loss += l
                i += 1
                if offset + batch_size > len(shuffledA):
                    epochs += 1
                    shuffledA, shuffledB, y = shuffle(shuffledA, shuffledB, y, random_state=rng2.randint(0, 1000))
                    offset = 0
                    loss /= i
                    sess.run(model["keep_prob"].assign(1.0))
                    tp, _, tse = evaluate(shuffledA, shuffledB, y)
                    p, _, se = evaluate(devA, devB, y_scores[1])
                    sess.run(model["keep_prob"].initializer)

                    print("Train loss: %.3f, Pearson: %.3f, Pearson on Dev: %.3f" % (loss, tp, p))
                    i = 0
                    loss = 0.0
                    if p > pearson + 1e-5:
                        pearson = p
                        saver.save(sess, '/tmp/my-model')
                    else:
                        lr = tf.get_variable("model/lr")
                        sess.run(lr.assign(lr * FLAGS.learning_rate_decay))
                        if epochs >= FLAGS.min_epochs:
                            break

            saver.restore(sess, '/tmp/my-model')
            sess.run(model["keep_prob"].assign(1.0))
            pr, sr, se = evaluate(testA, testB, y_scores[2])
            pearsons.append(pr)
            spearmans.append(sr)
            mses.append(se)
            print '######## Run %d #########' % run_id
            print 'Test Pearson: %.4f' % pr
            print 'Test Spearman: %.4f' % sr
            print 'Test MSE: %.4f' % se
            print '########################'
            os.remove('/tmp/my-model')

    mean_pearson = sum(pearsons) / len(pearsons)
    mean_spearman = sum(spearmans) / len(spearmans)
    mean_mse = sum(mses) / len(mses)

    def s_dev(mean, pop):
        d = 0.0
        for el in pop:
            d += (mean-el) * (mean-el)
        return math.sqrt(d/len(pop))

    print '######## Overall #########'
    print 'Test Pearson: %.4f (%.4f)' % (mean_pearson,  s_dev(mean_pearson, pearsons))
    print 'Test Spearman: %.4f (%.4f)' % (mean_spearman,  s_dev(mean_spearman, spearmans))
    print 'Test MSE: %.4f (%.4f)' % (mean_mse,  s_dev(mean_mse, mses))
    print '########################'


def load_data(loc):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(os.path.join(loc, 'SICK_train.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(os.path.join(loc, 'SICK_trial.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(os.path.join(loc, 'SICK_test_annotated.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [devA[1:], devB[1:]], [testA[1:], testB[1:]], [trainS, devS, testS]



def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y


#create batch given example sentences
def batchify(batchA, batchB, padding, tA, tB, idsA, idsB, lengthsA, lengthsB, max_length=None, max_batch_size=None):
    embedding_size = batchA[0][0][0].shape[0]

    tA = np.zeros([max_length, max_batch_size, embedding_size]) if tA is None else tA
    tB = np.zeros([max_length, max_batch_size, embedding_size]) if tB is None else tB
    idsA = np.ones([max_length, max_batch_size]) if idsA is None else idsA
    idsB = np.ones([max_length, max_batch_size]) if idsB is None else idsB
    idsA *= padding
    idsB *= padding

    lengthsA = np.zeros([max_batch_size], np.int32) if lengthsA is None else lengthsA
    lengthsB = np.zeros([max_batch_size], np.int32) if lengthsB is None else lengthsB

    for i in xrange(len(batchA)):
        lengthsA[i] = len(batchA[i][0])
        for j in xrange(len(batchA[i][0])):
            tA[j][i] = batchA[i][0][j]
            idsA[j][i] = batchA[i][1][j]

    for i in xrange(len(batchB)):
        lengthsB[i] = len(batchB[i][0])
        for j in xrange(len(batchB[i][0])):
            tB[j][i] = batchB[i][0][j]
            idsB[j][i] = batchB[i][1][j]

    return tA, tB, idsA, idsB, lengthsA, lengthsB


def fully_con_sim(inpA, inpB, h_size):
    inp = tf.concat(1, [inpA * inpB, tf.abs(inpA - inpB)])
    h = tf.contrib.layers.fully_connected(inp, h_size, activation_fn=tf.sigmoid,
                                          weight_init=None)
    scores = tf.contrib.layers.fully_connected(h, 5, weight_init=None)
    return scores


# Create Model
def create_model(length, l2_lambda, learning_rate, h_size, cell, embeddings, embedding_mode, keep_prob,
                 initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    with tf.variable_scope("model", initializer=initializer):
        inpA = tf.placeholder(tf.float32, [length, None, cell.input_size])
        inpB = tf.placeholder(tf.float32, [length, None, cell.input_size])
        idsA = tf.placeholder(tf.int32, [length, None])
        idsB = tf.placeholder(tf.int32, [length, None])
        lengthsA = tf.placeholder(tf.int32, [None])
        lengthsB = tf.placeholder(tf.int32, [None])
        inp = tf.concat(1, [inpA, inpB])
        lengths = tf.concat(0, [lengthsA, lengthsB])
        learning_rate = tf.get_variable("lr", (), tf.float32, tf.constant_initializer(learning_rate), trainable=False)

        batch_size = tf.cast(tf.gather(tf.shape(inpA), [1]), tf.float32)
        ids = tf.concat(1, [idsA, idsB])

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
            for i in xrange(length):
                inps[i] = tf.squeeze(inps[i], [0])
            _, final_state = rnn(cell, inps, init_state, sequence_length=lengths)
            out = tf.slice(final_state, [0, 0], [-1, cell.output_size])
            return out

        with tf.variable_scope("encoder_fw", initializer=initializer):
            h = my_rnn(None if embedding_mode == "tuned" else inp, None if embedding_mode == "fixed" else ids, cell,
                         length, embeddings)
           # h = tf.reshape(out, (length, -1, mem_size)), [length-1,0,0],[1,-1,-1])
            #h = tf.squeeze(h, [0])
        hA, hB = tf.split(0, 2, h)
        with tf.variable_scope("similarity", initializer=initializer):
            scores = fully_con_sim(hA, hB, h_size)

        probs = tf.nn.softmax(scores)
        y = tf.placeholder(tf.float32, [None, 5])

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(scores, y)) / batch_size
        train_params = tf.trainable_variables()
        if l2_lambda > 0.0:
            l2_loss = l2_lambda * tf.reduce_sum(array_ops.pack([tf.nn.l2_loss(t) for t in train_params]))
            loss = loss+l2_loss

    update = tf.train.AdamOptimizer(learning_rate, beta1=0.0).minimize(loss, var_list=train_params)
    return {"inpA":inpA, "inpB":inpB, "idsA":idsA, "idsB":idsB, "lengthsA":lengthsA, "lengthsB":lengthsB, "y":y,
            "probs":probs, "scores":scores,"keep_prob": keep_prob_var,
            "loss":loss, "update":update}


if __name__ == "__main__":
    # data loading specifics
    tf.app.flags.DEFINE_string('data', None, 'data dir of SICK.')
    tf.app.flags.DEFINE_string('embedding_format', 'prepared', 'glove|word2vec_bin|word2vec|dict|prepared')
    tf.app.flags.DEFINE_string('embedding_file', None, 'path to embeddings')

    # model
    tf.app.flags.DEFINE_string("mem_size", "150", "hidden size of model")
    tf.app.flags.DEFINE_string("h_size", "50", "size of interaction")

    # training
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
    tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization raten (only for batch training).")
    tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
    tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples per batch.")
    tf.app.flags.DEFINE_integer("min_epochs", 10, "Minimum num of epochs")
    tf.app.flags.DEFINE_string("cell", None, "'LSTM', 'GRU', 'MORU'")
    tf.app.flags.DEFINE_integer("seed", 12345, "Random seed.")
    tf.app.flags.DEFINE_integer("runs", 10, "How many runs.")
    tf.app.flags.DEFINE_string('embedding_mode', 'fixed', 'fixed|tuned|combined')
    tf.app.flags.DEFINE_integer('tunable_dim', 10,
                                'number of dims for tunable embeddings if embedding mode is combined')
    tf.app.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")

    FLAGS = tf.app.flags.FLAGS

    kwargs = None
    if FLAGS.embedding_format == "glove":
        kwargs = {"vocab_size": 2196017, "dim": 300}

    print "Loading embeddings..."
    e = None
    if FLAGS.embedding_format == "prepared":
        import io
        import pickle
        content = io.open(FLAGS.embedding_file, 'rb')
        state = pickle.load(content)
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(zip(words, counts))
            vocab = web.embedding.CountedVocabulary(word_count=word_count)
        else:
            vocab = web.embedding.OrderedVocabulary(voc)
        e = web.embedding.Embedding(vocabulary=vocab, vectors=vec)
    else:
        e = web.embeddings.load_embedding(FLAGS.embedding_file, format=FLAGS.embedding_format, normalize=False, clean_words=False,
                                          load_kwargs=kwargs)
    print "Done."

    h_sizes = map(int, FLAGS.h_size.split(","))
    mem_sizes = map(int, FLAGS.mem_size.split(","))
    cells = FLAGS.cell.split(",")

    import json
    for h_size in h_sizes:
        FLAGS.h_size = h_size
        for mem_size in mem_sizes:
            FLAGS.mem_size = mem_size
            for cell in cells:
                FLAGS.cell = cell
                print("Configuration: ")
                print(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))
                training(e, FLAGS)
