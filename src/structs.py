import datetime
from pathlib import Path
from pickle import dump, load
from tempfile import TemporaryDirectory

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (Add, Bidirectional, Concatenate, CuDNNLSTM, Dense,
                          Dropout, Embedding, Input, TimeDistributed)
from keras.models import Model, Sequential, load_model
from keras.utils.generic_utils import get_custom_objects
from keras_contrib.layers import CRF
from utils import emb_mat_init


class ModelJar:
    def __init__(self, embed, mappings, cats, embed_len):
        word_inv = mappings['form']
        npos = len(mappings['pos'])
        nne = len(mappings['ne'])
        nout = len(cats)
        width = len(word_inv) + 2

        pos = Input(shape=(None, ), dtype='int32')
        pos_emb = Embedding(npos, npos // 2)(pos)

        ne = Input(shape=(None, ), dtype='int32')
        ne_emb = Embedding(nne, nne // 2)(ne)

        capital = Input(shape=(None, 3), name='Capital_Input')

        special = Input(shape=(None, 12), name='Special_Input')

        form = Input(shape=(None, ), dtype='int32')

        emb = Embedding(
            40000,
            embed_len,
            embeddings_initializer=emb_mat_init(embed, word_inv),
            input_length=None)(form)

        emb.trainable = True

        concat = Concatenate()([emb, pos_emb, ne_emb, capital, special])
        drop = Dropout(0.25)(concat)

        lstm1 = Bidirectional(
            CuDNNLSTM(100, return_sequences=True), input_shape=(None,
                                                                width))(drop)
        skip = Concatenate()([concat, lstm1])
        lstm2 = Bidirectional(
            CuDNNLSTM(100, return_sequences=True), input_shape=(None,
                                                                width))(skip)
        dense = Dense(nout, activation='softmax')(lstm2)
        # crf = CRF(nout, learn_mode='join', activation='softmax')
        # out = crf(dense)
        out = dense
        model = Model(inputs=[form, pos, ne, capital, special], outputs=out)
        # model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
        model.compile(
            loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
        #model.summary()

        self.model = model
        self.mappings = mappings
        self.cats = cats
        self.embed_len = embed_len

    def save(self, filename: Path):
        with TemporaryDirectory() as tmpdir:
            fname = Path(tmpdir) / 'model'
            self.model.save_weights(str(fname))
            with open(fname, 'r+b') as f:
                weights = f.read()

        with filename.open('w+b') as f:
            dump((weights, self.mappings, self.cats, self.embed_len), f)

    @staticmethod
    def load(filename: Path):
        with Path(filename).open('r+b') as f:
            weights, mappings, cats, embed_len = load(f)

        with TemporaryDirectory() as tmpdir:
            fname = Path(tmpdir) / 'model'
            with open(fname, 'w+b') as f:
                f.write(weights)

            jar = ModelJar({}, mappings, cats, embed_len)
            jar.model.load_weights(str(fname))

        return jar

    def train(self, x, y, epochs=3, batch_size=64):
        # steps = (train_len / batch_size)# + 1 if train_len % batch_size > 0 else 0
        # model.fit_generator(gen, epochs=epochs, steps_per_epoch=steps, shuffle=False)
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='acc', min_delta=0.0005),
                TensorBoard(
                    log_dir=
                    f'./logs/{datetime.datetime.now().strftime("%y%m%d_%H%M")}'
                )
            ])
        self.model.summary()

    def train_batches(self, gen, train_len, epochs=3, batch_size=64):
        steps = (
            train_len / batch_size)  # + 1 if train_len % batch_size > 0 else 0
        self.model.fit_generator(
            gen,
            epochs=epochs,
            steps_per_epoch=steps,
            shuffle=True,
            callbacks=[
                EarlyStopping(monitor='acc', min_delta=0.0005),
                TensorBoard(
                    log_dir=
                    f'./logs/{datetime.datetime.now().strftime("%y%m%d_%H%M")}'
                )
            ])
        self.model.summary()
