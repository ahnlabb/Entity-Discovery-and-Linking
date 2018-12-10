#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count
from pickle import load, dump
import time
from docria.storage import DocumentIO
from utils import pickled, langforia, inverted, build_sequence, map2, flatten_once, print_dims
from gold_std import entity_to_dict, from_one_hot, one_hot, gold_std_idx, to_neleval, interpret_prediction
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding, Flatten, Dropout
from keras.models import Sequential, load_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('glove', type=Path)
    parser.add_argument('model', type=str)
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gold', action='store_true')
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
    train, gold, span_index, doc_index = [], [], [], []
    lbl_sets = defaultdict(set)

    gold_std, cats = gold_std_idx(docs)
    # mutate cats, return old cats
    numeric_cats = one_hot(gold_std, cats)

    def get_entity(doc, span):
        none = cats[('O', 'NOE', 'OUT')]
        docid = doc.props['docid']
        return gold_std[docid].get(span, none)

    for cnlp, doc in zip(core_nlp, docs):
        sentences, spans = core_nlp_features(cnlp, lbl_sets)
        entities = [[get_entity(doc, span) for span in sentence] for sentence in spans]
        current_doc = [[doc.props['docid'] for span in sentence] for sentence in spans]
        doc_index.extend(current_doc)
        span_index.extend(spans)
        gold.extend(entities)
        train.extend(sentences)
    
    return train, lbl_sets, gold, numeric_cats, span_index, doc_index


def corpus_extract(path):
    pass


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
    inside = ''
    for row in corenlp:
        if row:
            cols = row.split('\t')
            features = dict(zip(head, cols[1:]))
            if 'ne' in features:
                inside, features['ne'] = normalize_ne(inside, features['ne'])
            else:
                features['ne'] = '_'
            features['capital'] = features['form'][0].isupper()
            features['form'] = features['form'].lower()
            add(features, 'pos')
            add(features, 'ne')
            add(features, 'capital')
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


def update_mappings(mappings, lbl_sets):
    maxes = {feat: max(mapping.values()) for feat, mapping in mappings.items() if feat != 'form'}
    for key, lbls in lbl_sets.items():
        for lbl in lbls:
            if lbl not in mappings[key]:
                maxes[key] += 1
                mappings[key][lbl] = maxes[key]


def extract_features(mappings, sentences, padding=False):
    labels = {}
    for feat_name, mapping in mappings.items():
        labels[feat_name] = []
        for sentence in sentences:
            label_list = []
            for features in sentence:
                feature = features[feat_name]
                if feature in mapping:
                    label = mapping[feature]
                else:
                    mapping_len = len(next(iter(mapping.values())))
                    label = np.zeros(mapping_len)
                label_list.append(label)
            labels[feat_name].append(label_list)

    for feat, lbls in mappings.items():
        if feat == 'form':
            continue
        labels[feat] = [to_categorical(vals, num_classes=len(lbls)) for vals in labels[feat]]

    def concat(word):
        if padding:
            word = word + (np.zeros(1),)
        return np.concatenate(word)

    for sentence in zip(*labels.values()):
        yield [concat(word) for word in zip(*sentence)]


def build_model(feat_len=171, class_len=50):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(None, feat_len)))
    model.add(Dropout(0.2))
    model.add(Dense(class_len, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


def normalize_ne(inside, ne):
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

    return inside, ne


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
                inside, features['ne'] = normalize_ne(inside, features['ne'])

            sentences[-1].append(features)
            spans[-1].append((features['start'], features['end']))
        else:
            sentences.append([])
            spans.append([])
    if not sentences[-1]:
        sentences.pop(-1)
        spans.pop(-1)
    return sentences, spans


def batch(data, batch_len=32):
    f = data[:batch_len]
    del data[:batch_len]
    # longest sentence in batch
    longest = max(map(len, f))
    pad_f = np.array([0] * (len(f[0][0]) - 1) + [1])
    for i in range(len(f)):
        diff = longest - len(f[i])
        f[i].extend([pad_f] * diff)
    return np.array(f)


def batch_generator(features, batch_len=32):
    while len(features) > 0:
        yield batch(features, batch_len=batch_len)


def zipped_batch_generator(*lists, batch_len=32):
    for z in zip(*[batch_generator(L, batch_len=batch_len) for L in lists]):
        yield z


def predict(model, mappings, cats, text, padding=False):
    lbl_sets = defaultdict(set)
    sentences, spans = core_nlp_features(langforia(text, 'en').split('\n'), lbl_sets)
    features = list(extract_features(mappings, sentences, padding=padding))
    x = list(batch_generator(features))
    Y = model.predict(x)
    pred = [[interpret_prediction(p, cats) for p in y] for y in Y]
    return format_predictions(text, pred, sentences)


def format_predictions(input_text, predictions, sentences):
    pred_dict = {'text': input_text, 'entities': []}
    for sent, pred in zip(sentences, predictions):
        for word, class_tuple in zip(sent, pred):
            entity_dict = entity_to_dict(word['start'], word['end'], class_tuple)
            pred_dict['entities'].append(entity_dict)
    return pred_dict


def interpret_prediction(y, cats):
    one_hot = np.array([int(x) for x in y == max(y)])
    return from_one_hot(one_hot, cats)


def same_order(parent, *children, key=lambda x: len(x[1])):
    children = list(children)
    parent = sorted(enumerate(parent), key=key)
    for i, c in enumerate(children):
        children[i] = [c[j] for j, _ in parent]
    return tuple([[v for _, v in parent]] + children)


def test(xy):
    correct, total, correct_ent, total_ent = 0, 0, 0, 0
    for feat, gold in xy:
        pred = model.predict(feat, verbose=0)
        for p, g in zip(pred, gold):
            for w1, w2 in zip(p, g):
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


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        if arg == args.model:
            continue
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass

    corenlp, docs = read_and_extract(args.file, lambda doc: get_core_nlp(doc, lang='en'))
    train, lbl_sets, gold, cats, span_index, doc_index = docria_extract(corenlp, docs)
    embed = load_glove(args.glove)

    mapfile = Path('./%s.mappings.pickle' % args.model)
    if mapfile.exists():
        mappings = load(open(mapfile, 'r+b'))
        if args.refit:
            update_mappings(mappings, lbl_sets)
            dump(mappings, open(mapfile, 'w+b'))
    else:
        mappings = create_mappings(embed, lbl_sets)
        dump(mappings, open(mapfile, 'w+b'))

    features = extract_features(mappings, train, padding=True)
    features, gold, span_index, doc_index = same_order(features, gold, span_index, doc_index)

    if args.test:
        batch_len = 1
        lists = (features, span_index, doc_index)
    elif args.gold:
        batch_len = 1
        lists = (gold, span_index, doc_index)
    else:
        batch_len = 100
        lists = (features, gold)
    batches = list(zipped_batch_generator(*lists, batch_len=batch_len))

    fit = args.refit
    name = args.model
    if Path(name).exists():
        model = load_model(name)
    else:
        fit = True
        f_len = batches[0][0].shape[-1]
        model = build_model(feat_len=f_len, class_len=len(cats))

    if fit:
        for i, b in enumerate(batches, 1):
            print('\nBatch', i, '\n-----------')
            x, y = b
            model.fit(x, y, epochs=10, validation_split=0.1, verbose=2, batch_size=batch_len)

        model.save(name)
    
    if args.test or args.gold:
        neleval_out = ''
        i = 0
        for data, span_ind, doc_ind in batches:
            if args.test:
                data = model.predict(data, verbose=0)
            neleval_out += to_neleval(data, span_ind, doc_ind, cats, i)
            i += 1
        print(neleval_out, end='')
    else:
        test(batches)
