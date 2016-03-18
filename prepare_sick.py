import train_sick
import tensorflow as tf
import web.embeddings
import web.embedding
import os
import numpy as np
import nltk

# data loading specifics
tf.app.flags.DEFINE_string('data', None, 'data dir of SICK.')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'glove|word2vec_bin')
tf.app.flags.DEFINE_string('embedding_file', None, 'path to embeddings')
tf.app.flags.DEFINE_string('out_format', 'glove', 'glove|word2vec_bin')
tf.app.flags.DEFINE_string('out_file', None, 'path to output embeddings')

FLAGS = tf.app.flags.FLAGS

# Load data
[trainA, trainB], [devA, devB], [testA, testB], _ = train_sick.load_data(FLAGS.data)

#Load Embeddings
kwargs = None
if FLAGS.embedding_format == "glove":
    kwargs = {"vocab_size": 2196017, "dim": 300}

print "Loading embeddings..."
embeddings = web.embeddings.load_embedding(FLAGS.embedding_file, format=FLAGS.embedding_format, normalize=False, clean_words=False, load_kwargs=kwargs)
print "Done."

embedding_size = embeddings.vectors.shape[1]

print "Collecting necessary embeddings for sick task..."
sick_embeddings = dict()


def fill_vocab(dataset):
    for sentence in dataset:
        for w in nltk.word_tokenize(sentence):
            if w not in sick_embeddings:
                wv = embeddings.get(w, embeddings.get(w.lower(), np.random.uniform(-0.05, 0.05, embedding_size)))
                sick_embeddings[w] = wv
            w = w.lower()
            if w not in sick_embeddings:
                wv = embeddings.get(w, np.random.uniform(-0.05, 0.05, embedding_size))
                sick_embeddings[w] = wv

fill_vocab(trainA)
fill_vocab(trainB)
fill_vocab(devA)
fill_vocab(devB)
fill_vocab(testA)
fill_vocab(testB)

sick_embeddings = web.embedding.Embedding.from_dict(sick_embeddings)

sick_embeddings.save(FLAGS.out_file)
