import web.embeddings
import web.embedding


def load_embeddings(fn, format="prepared"):
    if format == "prepared":
        import io
        import pickle

        content = io.open(fn, 'rb')
        u = pickle._Unpickler(content)
        u.encoding = 'latin1'
        state = u.load()
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(zip(words, counts))
            vocab = web.embedding.CountedVocabulary(word_count=word_count)
        else:
            vocab = web.embedding.OrderedVocabulary(voc)
        return web.embedding.Embedding(vocabulary=vocab, vectors=vec)
    else:
        return web.embeddings.load_embedding(fn, format=format, normalize=False,
                                             clean_words=False)
