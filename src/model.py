from keras.layers import Bidirectional, LSTM, Dense, Embedding, Concatenate, Input, TimeDistributed
from keras.models import Model
from keras_contrib.layers import CRF
from utils import emb_mat_init


def build_model(embed, word_inv, npos, nne, nout, embed_len):
    width = len(word_inv) + 2

    pos = Input(shape=(None,), dtype='int32')
    pos_emb = Embedding(npos, npos // 2)(pos)

    ne = Input(shape=(None,), dtype='int32')
    ne_emb = Embedding(nne, nne // 2)(ne)

    form = Input(shape=(None,), dtype='int32')

    emb = Embedding(width,
                    embed_len,
                    embeddings_initializer=emb_mat_init(embed, word_inv),
                    mask_zero=True,
                    input_length=None)(form)

    emb.trainable = True

    concat = Concatenate()([emb, pos_emb, ne_emb])

    lstm = Bidirectional(LSTM(25, return_sequences=True), input_shape=(None, width))(concat)
    dense = TimeDistributed(Dense(nout, activation='softmax'))(lstm)
    crf = CRF(nout)(dense)
    model = Model(inputs=[form, pos, ne], outputs=crf)
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=['acc', crf.accuracy])
    return model


def make_model(x, y, embed, word_inv, npos, nne, nout, embed_len, train_len, epochs=3, batch_size=64):
    model = build_model(embed, word_inv, npos, nne, nout, embed_len)
    # steps = (train_len / batch_size)# + 1 if train_len % batch_size > 0 else 0
    # model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps, shuffle=False)
    model.fit(x, y, epochs=epochs, batch_size=batch_size)
    model.summary()
    return model


def make_model_batches(gen, embed, word_inv, npos, nne, nout, embed_len, train_len, epochs=3, batch_size=64):
    model = build_model(embed, word_inv, npos, nne, nout, embed_len)
    steps = (train_len / batch_size)  # + 1 if train_len % batch_size > 0 else 0
    model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps, shuffle=True)
    model.summary()
    return model
