#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count
from pickle import load, dump

from docria.storage import DocumentIO
from keras.layers import Bidirectional, LSTM, Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
import requests
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('glove', type=Path)
    return parser.parse_args()


def load_glove(path):
    with path.open('r') as f:
        rows = map(lambda x: x.split(), f)
        embed = {}
        for row in rows:
            embed[row[0]] = np.asarray(row[1:], dtype='float32')

        return embed


def take_twos(iterable):
    itr = iter(iterable)
    while True:
        yield next(itr), next(itr)


def langforia_url(lang, config, format='json'):
    return f'http://vilde.cs.lth.se:9000/{lang}/{config}/api/{format}'


def langforia(text, lang, config='corenlp_3.8.0'):
    url = langforia_url(lang, config, format='tsv')
    request = requests.post(url, data=text)
    return request.text


def model():
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def core_nlp_features(doc, lang):
    train = []
    lbl_sets = defaultdict(set)

    def add(features, name):
        lbl_sets[name].add(features[name])

    for i, a in enumerate(doc):
        if i % 10 == 0:
            print(i)
        text = str(a.texts['main']).encode('utf-8')
        corenlp = iter(langforia(text, lang).split('\n'))
        head = next(corenlp).split('\t')[1:]
        sentences = [[]]
        for row in corenlp:
            if row:
                cols = row.split('\t')
                features = dict(zip(head, cols[1:]))
                add(features, 'pos')
                add(features, 'ne')
                sentences[-1].append(features)
            else:
                sentences.append([])
        if not sentences[-1]:
            sentences.pop(-1)
        train.extend(sentences)
    return train, lbl_sets


def extract_features(embed, core_nlp):
    train, lbl_sets = core_nlp

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


def pickled(path, fun):
    if path.suffix == '.pickle':
        with path.open('r+b') as f:
            return load(f)
    else:
        result = fun(path)
        with path.with_suffix('.pickle').open('w+b') as f:
            dump(result, f)
        return result


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass
    embed = pickled(args.glove, load_glove)

    def read_and_extract(path):
        with DocumentIO.read(path) as doc:
            doc = list(doc)
            print(len(doc))
            return core_nlp_features(doc, 'en')

    core_nlp = read_and_extract(args.file)
    features = extract_features(embed, core_nlp)

    print(features[0][0])
