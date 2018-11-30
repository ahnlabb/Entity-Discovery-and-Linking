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

    def get_entity(doc, span):
        none = np.zeros(len(cats))
        docid = doc.props['docid']
        return gold_std[docid].get(span, none)

    for i, doc in enumerate(docs):
        if i % 10 == 0:
            print(i)
        spans = core_nlp_features(doc, train, lbl_sets, lang=lang)
        entities = [[get_entity(doc, span) for span in sentence] for sentence in spans]
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
            spans[-1].append((int(features['start']), int(features['end'])))
        else:
            sentences.append([])
            spans.append([])
    if not sentences[-1]:
        sentences.pop(-1)
        spans.pop(-1)
    train.extend(sentences)
    return spans


def extract_features(embed, train, lbl_sets):
    #print(len(lbl_sets['pos']), lbl_sets['pos'])
    #print(len(lbl_sets['ne']), lbl_sets['ne'])

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
    model.add(Embedding(108, 50, input_shape=(414,), mask_zero=True))
    model.add(Bidirectional(LSTM(25, return_sequences=True)))
    model.add(Dense(13, activation='softmax'))
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
    from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding, Flatten
    from keras.models import Sequential

    features = extract_features(embed, train, lbl_sets)
    features = sorted(enumerate(features), key=lambda x: len(x[1]), reverse=True)
    gold = [gold[i] for i,_ in features]
    def print_dims(data):
        try:
            print(type(data), '(' + str(len(data)) + ')', end=' ')
            print_dims(data[0])
        except:
            print()
            return
    
    def add_feature(vec):
        new_vec = np.zeros(len(vec) + 1)
        new_vec[1:] = vec
        return new_vec

    # TODO: add more list comprehensions
    gold = [[add_feature(e) for e in s] for s in gold]
    features = [[add_feature(w) for w in v] for _,v in features]

    def batch(feats, gold, batch_len=100):
        f, g = feats[:batch_len], gold[:batch_len]
        del feats[:batch_len]
        del gold[:batch_len]
        # longest sentence in batch
        longest = max(map(len, f))
        pad_f = np.array([1] + [0] * (len(f[0]) - 1))
        pad_g = np.array([1] + [0] * (len(g[0]) - 1))
        for i in range(len(f)):
            diff = longest - len(f[i])
            f[i].extend([pad_f] * diff)
            g[i].extend([pad_g] * diff)
        return f, g

    x,y = batch(features, gold, batch_len=len(features))

    # data sets
    cutoff = 9 * len(x) // 10
    x_train = np.array(x)
    print_dims(x_train)
    y_train = np.array(y)
    x_test = np.array(x[cutoff:])
    y_test = np.array(y[cutoff:])

    model = model()
    model.summary()
    model.fit(x_train, y_train, batch_size=100, epochs=1)
