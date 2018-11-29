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
        entities = [[get_entity(span) for span in sentence] for sentence in spans]
        gold.extend(entities)

    return train, lbl_sets, gold

def build_indices(train, embed):
    wordset = set([features['form'] for sentence in train for features in sentence])
    wordset.update(embed.keys())
    word_ind = dict(enumerate(wordset, 2))
    return word_ind

def inverted(a):
    return {v:k for k,v in a.items()}

def build_sequence(l, invind):
    return [invind[w] for w in l]

def map2(fun, x, y):
    return fun(x[0], y[0]), fun(x[1], y[1])

def build_sequences(train, embed):
    indices = list(map(inverted, build_indices(train, embed)))
    data = 0
    xy_sequences = tuple(zip(*(map2(build_sequence, tup, indices) for tup in data)))
    return map(pad_sequences, xy_sequences)

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
        spans.pop(-1)
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
    model.add(Embedding(107, 50, mask_zero=True, input_length=None))
    model.add(Bidirectional(LSTM(25, return_sequences=True), input_shape=(None, 107)))
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
    features = sorted(enumerate(features), key=lambda x: len(x[1]), reverse=True)
    gold = [gold[i] for i,_ in features]
    print(len(gold[0]), len(features[0][1]))
    #xy = sorted(zip(features, gold), key=lambda x: len(x[0]), reverse=True)
    #features, gold = [x for x,y in xy], [y for x,y in xy]
    quit()

    def batch(feats, gold, batch_len=100):
        batch = feats[:batch_len], gold[:batch_len]
        del feats[:batch_len]
        del gold[:batch_len:]
        longest = max(map(len, batch[0]))
        pad_vec = np.zeros(len(batch[0][0]))
        for i in range(batch_len):
            diff = longest - len(batch[0][i])
            padding = [pad_vec] * diff
            batch[0][i].extend(padding)
            batch[1][i].extend(padding)
        return batch

    f, g = [], []
    while len(features) > 0:
        print(len(features), len(gold))
        f, g = batch(features, gold)

    # data sets
    cutoff = 9 * len(f) // 10
    x_train = f[:cutoff]
    y_train = g[:cutoff]
    x_test = f[cutoff:]
    y_test = g[cutoff:]

    model = model()
    model.fit(x_train, y_train, batch_size=100, epochs=1)
