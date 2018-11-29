#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count, product
from pickle import load, dump
from docria.storage import DocumentIO
from utils import pickled, langforia
from gold_std import gold_std_idx, one_hot
import requests
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('glove', type=Path)
    return parser.parse_args()


@pickled
def load_glove(path):
    with path.open('r') as f:
        rows = map(lambda x: x.split(), f)
        embed = {}
        for row in rows:
            embed[row[0]] = np.asarray(row[1:], dtype='float32')
        return embed


def docria_extract(docs, lang='en'):
    train, gold = [], []
    lbl_sets = defaultdict(set)

    gold_std, cats = gold_std_idx(docs)
    one_hot(gold_std, cats)
    def get_entity(span):
        none = np.zeros(len(cats))
        return gold_std.get(span, none)
    for i, doc in enumerate(docs):
        if i % 10 == 0:
            print(i)
        spans = core_nlp_features(doc, train, lbl_sets, lang=lang)
        spans = [[get_entity(span) for span in sentence] for sentence in spans]
        gold.extend(spans)

    return train, lbl_sets, gold

def core_nlp_features(doc, train, lbl_sets, lang='en'):
    spans = []

    def add(features, name):
        lbl_sets[name].add(features[name])

    text = str(doc.texts['main'])
    corenlp = iter(langforia(text, lang).split('\n'))
    head = next(corenlp).split('\t')[1:]
    sentences = [[]]
    spans = [[]]
    for row in corenlp:
        if row:
            cols = row.split('\t')
            features = dict(zip(head, cols[1:]))
            add(features, 'pos')
            add(features, 'ne')
            sentences[-1].append(features)
            spans[-1].append((features['start'], features['end']))
        else:
            sentences.append([])
            spans.append([])
    if not sentences[-1]:
        sentences.pop(-1)
    train.extend(sentences)
    return spans


def extract_features(embed, train, lbl_sets):
    print(len(lbl_sets['pos']), lbl_sets['pos'])
    print(len(lbl_sets['ne']), lbl_sets['ne'])

    labels = {}
    mappings = {key: dict(zip(lbls, count(0))) for key, lbls in lbl_sets.items()}
    mappings['form'] = embed
    for key, mapping in mappings.items():
        labels[key] = []
        for sentence in train:
            labels[key].append([])
            for features in sentence:
                try:
                    label = mapping[features[key]]
                except KeyError:
                    label = np.zeros(len(next(iter(mapping.values()))))
                labels[key][-1].append(label)

    for key, lbls in lbl_sets.items():
        labels[key] = [to_categorical(vals, num_classes=len(lbls)) for vals in labels[key]]

    return [[np.concatenate(word) for word in zip(*sentence)] for sentence in zip(*labels.values())]


def model():
    model = Sequential()
    # input vectors of dim 108 -> output vectors of dim 25
    #model.add(Embedding(108, 25, input_length=10))
    # each sample is 10 vectors of 25 dimensions
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(10, 107)))
    # arbitrarily (?) pick 25 hidden units
    model.add(Bidirectional(LSTM(25)))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass

    embed = load_glove(args.glove)

    @pickled
    def read_and_extract(path, fun):
        with DocumentIO.read(path) as doc:
            return fun(list(doc))
    train, lbl_sets, gold = read_and_extract(args.file, lambda doc: docria_extract(doc, lang='en'))

    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding
    from keras.models import Sequential

    features = extract_features(embed, train, lbl_sets)
    xy = sorted(zip(features, gold), key=lambda x: len(x[0]), reverse=True)
    features, gold = [x for x,y in xy], [y for x,y in xy]
    print(features)

    for f in features[:1]:
        print(len(f))
        for v in f[:1]:
            print(len(v))

    def batch(feats, gold, batch_len=10):
        batch = feats[:batch_len], gold[:batch_len]
        feats, gold = feats[batch_len:], gold[batch_len:]
        longest = max(batch[0], key=len)
        for i,b in enumerate(batch):
            diff = longest - len(batch[0])
            batch = batch[0] + padding, batch[1]
        return 

    # data sets
    cutoff = 9 * len(features) // 10
    x_train = features[:cutoff]
    y_train = gold[:cutoff]
    x_test = features[cutoff:]
    y_test = gold[cutoff:]

    model = model()
    model.fit(x_train, y_train, batch_size=8, epochs=1)
