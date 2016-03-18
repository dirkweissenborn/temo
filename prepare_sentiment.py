import train_sentiment
import tensorflow as tf
import web.embeddings
import web.embedding
import os
import numpy as np

# data loading specifics
tf.app.flags.DEFINE_string('data', None, 'data dir of SST.')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'glove|word2vec_bin')
tf.app.flags.DEFINE_string('embedding_file', None, 'path to embeddings')
tf.app.flags.DEFINE_string('out_format', 'glove', 'glove|word2vec_bin')
tf.app.flags.DEFINE_string('out_file', None, 'path to output embeddings')

FLAGS = tf.app.flags.FLAGS

# Load data
train = train_sentiment.load_data(os.path.join(FLAGS.data, "train"))
dev = train_sentiment.load_data(os.path.join(FLAGS.data, "dev"))
test = train_sentiment.load_data(os.path.join(FLAGS.data, "test"))

#Load embeddings
kwargs = None
if FLAGS.embedding_format == "glove":
    kwargs = {"vocab_size": 2196017, "dim": 300}

print "Loading embeddings..."
embeddings = web.embeddings.load_embedding(FLAGS.embedding_file, format=FLAGS.embedding_format, normalize=False, clean_words=False, load_kwargs=kwargs)
print "Done."

embedding_size = embeddings.vectors.shape[1]

print "Collecting necessary embeddings for sentiment task..."
sst_embeddings = dict()


def fill_vocab(dataset):
    for tree in dataset:
        for w in tree.sentence:
            if w not in sst_embeddings:
                wv = embeddings.get(w, embeddings.get(w.lower(), np.random.uniform(-0.05, 0.05, embedding_size)))
                sst_embeddings[w] = wv

fill_vocab(train)
fill_vocab(dev)
fill_vocab(test)

sst_embeddings = web.embedding.Embedding.from_dict(sst_embeddings)

sst_embeddings.save(FLAGS.out_file)
