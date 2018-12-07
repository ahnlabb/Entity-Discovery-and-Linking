#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count, product
from pickle import load, dump
from random import shuffle
import time
from docria.storage import DocumentIO
from utils import pickled, langforia, inverted, build_sequence, map2
from gold_std import gold_std_idx, one_hot, from_one_hot
import requests
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding, Flatten, Dropout
from keras.models import Sequential, load_model

import keras as ks


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('glove', type=Path)
    return parser.parse_args()


@pickled
def read_and_extract(path, fun):
    with DocumentIO.read(path) as doc:
        return fun(list(doc))


@pickled
def load_glove(path):
    with path.open('r') as f:
        rows = map(lambda x: x.split(), f)
        embed = {}
        for row in rows:
            embed[row[0]] = np.asarray(row[1:], dtype='float32')
        return embed


def get_core_nlp(docs, lang):
    def call_api(doc):
        text = str(doc.texts['main'])
        return langforia(text, lang).split('\n')
    api_data = []
    start = time.perf_counter()
    for i, doc in enumerate(docs, 1):
        print("Document %d/%d " % (i, len(docs)), end='', flush=True)
        api_data.append(call_api(doc))
        time_left = int((time.perf_counter() - start) / i * (len(docs) - i))
        print(f' ETA: {time_left // 60} min {time_left % 60} s    ', end='\r', flush=True)
    time_tot = int(time.perf_counter() - start)
    print(f'Done: {len(docs)} documents in {time_tot // 60} min {time_tot % 60} s')
    return api_data, docs


def docria_extract(core_nlp, docs):
    train, gold, span_index = [], [], []
    lbl_sets = defaultdict(set)

    gold_std, cats = gold_std_idx(docs)
    one_hot(gold_std, cats)

    def get_entity(doc, span):
        none = np.zeros(len(cats))
        docid = doc.props['docid']
        return gold_std[docid].get(span, none)

    for cnlp, doc in zip(core_nlp, docs):
        sentences, spans = core_nlp_features(cnlp, lbl_sets)
        entities = [[get_entity(doc, span) for span in sentence] for sentence in spans]
        span_index.extend(spans)
        gold.extend(entities)
        train.extend(sentences)

    return train, lbl_sets, gold, cats, span_index

def build_indices(train, embed):
    wordset = set([features['form'] for sentence in train for features in sentence])
    wordset.update(embed.keys())
    word_ind = dict(enumerate(wordset, 2))
    return word_ind

def build_sequences(train, embed):
    indices = list(map(inverted, build_indices(train, embed)))
    data = 0
    xy_sequences = tuple(zip(*(map2(build_sequence, tup, indices) for tup in data)))
    return map(pad_sequences, xy_sequences)

def core_nlp_features(corenlp, lbl_sets):
    corenlp = iter(corenlp)
    spans = []

    def add(features, name):
        feat = features[name]
        lbl_sets[name].add(feat)

    head = next(corenlp).split('\t')[1:]
    sentences = [[]]
    spans = [[]]
    for row in corenlp:
        if row:
            cols = row.split('\t')
            features = dict(zip(head, cols[1:]))
            if 'ne' not in features:
                features['ne'] = '_'
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
    return sentences, spans

def create_mappings(embed, lbl_sets):
    mappings = {key: dict(zip(lbls, count(0))) for key, lbls in lbl_sets.items()}
    mappings['form'] = embed
    return mappings


def extract_features(mappings, train):
    labels = {}
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

    for key, lbls in mappings.items():
        if key == 'form':
            continue
        labels[key] = [to_categorical(vals, num_classes=len(lbls.keys())) for vals in labels[key]]

    for sentence in zip(*labels.values()):
        yield [np.concatenate(word) for word in zip(*sentence)]


def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True, stateful=False), input_shape=(None, 158)))
    #model.add(Dropout(0.2))
    model.add(Dense(50, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

def corenlp_parse(text, lang='en'):
    result = langforia(text, lang).split('\n')
    itr = iter(result)
    head = next(itr).split('\t')[1:]
    sentences = [[]]
    spans = [[]]
    inside = ""
    for row in itr:
        if row:
            cols = row.split('\t')
            features = dict(zip(head, cols[1:]))
            if 'ne' in features:
                ne = features['ne']
                if inside:
                    if ne == ')':
                        ne = 'E-' + inside
                        inside = ''
                    else:
                        ne = 'I-' + inside
                elif ne[-1] == ')':
                    ne = 'S-' + ne[1:-1]
                    inside = ''
                else:
                    inside = ne[1:]
                    ne = 'B-' + inside
                features['ne'] = ne
            sentences[-1].append(features)
            spans[-1].append((features['start'], features['end']))
        else:
            sentences.append([])
            spans.append([])
    if not sentences[-1]:
        sentences.pop(-1)
        spans.pop(-1)
    return sentences, spans

    
def _batch(feats, gold, batch_len=32):
    f, g = feats[:batch_len], gold[:batch_len]
    del feats[:batch_len]
    del gold[:batch_len]
    # longest sentence in batch
    longest = max(map(len, f))
    pad_f = np.array([0] * (len(f[0][0]) - 1) + [1])
    # NOE-PAD
    pad_g = np.array([0] * (len(g[0][0]) - 1) + [1])
    for i in range(len(f)):
        diff = longest - len(f[i])
        f[i].extend([pad_f] * diff)
        g[i].extend([pad_g] * diff)
    return f, g


def batch(data, batch_len=32):
    f = data[:batch_len]
    del data[:batch_len]
    # longest sentence in batch
    longest = max(map(len, f))
    pad_f = np.array([0] * (len(f[0][0]) - 1) + [1])
    for i in range(len(f)):
        diff = longest - len(f[i])
        f[i].extend([pad_f] * diff)
    return f


def predict(model, mappings, cats, text):
    lbl_sets = defaultdict(set)
    sentences, spans = core_nlp_features(langforia(text, 'en').split('\n'), lbl_sets)
    features = [[add_feature(w) for w in f] for f in extract_features(mappings, sentences)]
    x = np.array((batch(features, batch_len=len(features))))
    Y = model.predict(x)
    return {'text': text, 'entities': [{'start': int(word['start']), 'stop': int(word['end']), 'class': str('-'.join(map(str, interpret_prediction(p, cats))))} for sentence, pred in zip(sentences, Y) for word, p in zip(sentence, pred)]}


# This is very stupid
def add_feature(vec):
    new_vec = np.zeros(len(vec) + 1)
    new_vec[:-1] = vec
    return new_vec


def interpret_prediction(y, cats):
    one_hot = np.array([int(x) for x in y == max(y)])
    return from_one_hot(one_hot, cats)


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass

    corenlp, docs = read_and_extract(args.file, lambda doc: get_core_nlp(doc, lang='en'))
    train, lbl_sets, gold, cats, span_index = docria_extract(corenlp, docs)
    embed = load_glove(args.glove)
    print(cats)

    with open('./cats.pickle', 'w+b') as f:
        dump(cats, f)

    mapfile = Path('./mappings.pickle')
    if mapfile.exists():
        mappings = load(open(mapfile, 'r+b'))
    else:
        mappings = create_mappings(embed, lbl_sets)
        dump(mappings, open(mapfile, 'w+b'))

    features = extract_features(mappings, train)
    features = sorted(enumerate(features), key=lambda x: len(x[1]), reverse=False)
    gold = [gold[i] for i,_ in features]
    span_index = [span_index[i] for i,_ in features]
    features = [v for _,v in features]


    def print_dims(data):
        try:
            try:
                print(type(data), str(data.shape), end=' ')
            except:
                print(type(data), '(' + str(len(data)) + ')', end=' ')
            print_dims(data[0])
        except:
            print()
            return
    

    # add NOE-OUT to zeroed gold standard vectors
    for i, sentence in enumerate(gold):
        for j, word in enumerate(sentence):
            if not word.any():
                gold[i][j] = np.array([0] * (len(word) - 2) + [1, 0])

    # TODO: add more list comprehensions
    features = [[add_feature(w) for w in v] for v in features] 

    def batch_generator(features, gold, batch_len=32):
        while len(features) > 0:
            yield batch(features, batch_len=batch_len), batch(gold, batch_len=batch_len)
    
    batch_len = 100
    batches = [(np.array(x), np.array(y))
               for x,y in batch_generator(features, gold, batch_len=batch_len)]
    batches = batches[:50] + batches[53:]
    test = batches[50:53]


    name = 'model.h5'
    if Path(name).exists():
        model = load_model(name)
    else:
        model = build_model()

        for i,b in enumerate(batches, 1):
            print('\nBatch', i,'\n-----------')
            x, y = b
            model.fit(x, y, epochs=10, validation_split=0.1, verbose=2, batch_size=100)

        model.save(name)
    
    text = "My friend, have you heard of the passing of George Bush Senior?"
    predictions = predict(model, mappings, cats, text)
    for p in predictions:
        for word in p:
            print(word)
    

    correct, total, correct_ent, total_ent = 0, 0, 0, 0
    for feat,gold in test:
        pred = model.predict(feat, verbose=0)
        for p,g in zip(pred,gold):
            for w1,w2 in zip(p,g):
                w1 = np.array([int(x) for x in w1 == max(w1)])
                c1 = from_one_hot(w1, cats)
                c2 = from_one_hot(w2, cats)
                total += 1
                if c1 == c2:
                    correct += 1
                if c2 != ('O', 'NOE', 'OUT') and c2 != ('O', 'NOE', 'PAD'):
                    total_ent += 1
                    if c1 == c2:
                        correct_ent += 1
    print(100 * correct / total, '% correct total')
    print(100 * correct_ent / total_ent, '% correct entities')
