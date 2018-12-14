#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count, repeat
from pickle import load, dump
from collections import Counter
import base64
from docria.storage import DocumentIO
from utils import pickled, langforia, inverted, build_sequence, map2, flatten_once, print_dims, save_model, load_model, emb_mat_init, mapget, zip_from_end
from gold_std import entity_to_dict, from_one_hot, one_hot, gold_std_idx, to_neleval, interpret_prediction
from structs import ModelJar
import numpy as np
import time
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Bidirectional, LSTM, Dense, Activation, Embedding, Concatenate, Input
from keras.models import Model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('glove', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('lang', type=str)
    parser.add_argument('--predict', type=Path)
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
        print("Document %d/%d " % (i, len(docs)), end='', flush=True, file=sys.stderr)
        api_data.append(call_api(doc))
        time_left = int((time.perf_counter() - start) / i * (len(docs) - i))
        print(f' ETA: {time_left // 60} min {time_left % 60} s    ', end='\r', flush=True, file=sys.stderr)
    time_tot = int(time.perf_counter() - start)
    print(f'Done: {len(docs)} documents in {time_tot // 60} min {time_tot % 60} s', file=sys.stderr)
    return api_data, docs


def txt2xml(doc):
    index = {}
    for node in doc.layers['tac/segments']:
        text = node.fld.text
        if text:
            i, j = text.start, text.stop
            x, y = node.fld.xml.start, node.fld.xml.stop
            indices = enumerate([(x,y)]*(j-i), i)
            for txt, xml in indices:
                index[txt] = xml
    #for s, e in index.items():
    #    print(s, e)
    return index



def docria_extract(core_nlp, docs, saved_cats=None):
    train, gold, span_index, doc_index = [], [], [], []
    lbl_sets = defaultdict(set)

    gold_std, cats = gold_std_idx(docs)
    if saved_cats:
        cats = saved_cats
    # mutate cats, return old cats
    numeric_cats = one_hot(gold_std, cats)

    def get_entity(doc, span):
        none = cats[('O', 'NOE', 'OUT')]
        docid = doc.props['docid']
        return gold_std[docid].get(span, none)

    for cnlp, doc in zip(core_nlp, docs):
        sentences, spans = core_nlp_features(cnlp, lbl_sets)
        entities = [[get_entity(doc, s) for s in sentence] for sentence in spans]
        current_doc = [[doc.props['docid'] for _ in sentence] for sentence in spans]
        doc_index.extend(current_doc)
        span_index.extend([sentence for sentence in spans])
        gold.extend(entities)
        train.extend(sentences)

    return train, lbl_sets, gold, numeric_cats, span_index, doc_index


def build_indices(train, embed):
    wordset = set([features['form'] for sentence in train for features in sentence])
    wordset.update(embed.keys())
    word_inv = dict(zip(wordset, count(2)))
    return word_inv


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


def create_mappings(train, embed, lbl_sets):
    mappings = {key: dict(zip(lbls, count(1))) for key, lbls in lbl_sets.items()}
    for key in mappings:
        mappings[key]['OOV'] = 0
    mappings['form'] = build_indices(train, embed)
    return mappings


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
                    # Out of vocabulary
                    if feat_name == 'form':
                        mapping_len = len(next(iter(mapping.values())))
                        label = np.zeros(mapping_len)
                    else:
                        label = 0
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


def build_model(max_len, embed, word_inv, npos, nne, nout, embed_len):
    width = len(word_inv) + 2
    pos = Input(shape=(max_len, npos))
    ne = Input(shape=(max_len, nne))
    form = Input(shape=(max_len,))
    emb = Embedding(width,
            embed_len,
            embeddings_initializer=emb_mat_init(embed, word_inv),
            mask_zero=True,
            input_length=None)(form)

    emb.trainable = True

    concat = Concatenate()([emb, pos, ne])

    lstm = Bidirectional(LSTM(25, return_sequences=True), input_shape=(None, width))(concat)
    out = Dense(nout, activation='softmax')(lstm)
    model = Model(inputs= [form, pos, ne], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])
    return model

def make_model(max_len, x, y, embed, word_inv, npos, nne, nout, embed_len, epochs=3, batch_size=128):
    model = build_model(max_len, embed, word_inv, npos, nne, nout, embed_len)
    model.fit(x, y, epochs=epochs, batch_size=batch_size)
    model.summary()
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
    longest = max(map(len, f))
    feat_len = len(f[0][0])
    pad_f = np.array([0] * (feat_len - 1) + [1])
    for i in range(len(f)):
        diff = longest - len(f[i])
        f[i].extend([pad_f] * diff)
    return np.array(f)


def batch_generator(features, batch_len=32):
    while len(features) > 0:
        yield batch(features, batch_len=batch_len)


def zipped_batch_generator(*lists, batch_len=32):
    for z in zip(*[batch_generator(lst, batch_len=batch_len) for lst in lists]):
        yield z


def get_links(entity_lst, wiki_dir, wkd2fb):
    with wkd2fb.open('r+b') as f:
        fbmap = load(f)
    wikimap = {}
    for path in wiki_dir.glob('part-*'):
        with path.open('r') as f:
            for line in f:
                split = line.rfind(',')
                wkd = int(line[split+1:])
                wikimap[line[:split]] = fbmap.get(wkd, 'wkd'+str(base64.b64encode(str(wkd).encode('ascii'))))
    return [wikimap[ent] for ent in entity_lst]


def same_order(parent, *children, key=lambda x: len(x[1])):
    children = list(children)
    parent = sorted(enumerate(parent), key=key)
    for i, c in enumerate(children):
        children[i] = [c[j] for j, _ in parent]
    return tuple([[v for _, v in parent]] + children)


def test(model, xy):
    """rudimentary prediction test"""
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


def predict_to_layer(docs, corenlp, mappings):
    lbl_sets = defaultdict(set)
    for doc in docs:
        sentences, spans = core_nlp_features(corenlp, lbl_sets)
        features = extract_features(mappings, sentences, padding=True)

        entities = doc.add_layer('tac/entity', text=T.span('main'), xml=T.span('xml'))
        entities.add(text=main[0:100])
        span_translate(doc, 'tac/segments', ('xml', 'text'), 'tac/entity', ('text', 'xml')) 


def predict(model, mappings, cats, text, padding=False):
    lbl_sets = defaultdict(set)
    sentences, spans = core_nlp_features(langforia(text, 'en').split('\n'), lbl_sets)
    features = list(extract_features(mappings, sentences, padding=padding))
    x = list(batch_generator(features, batch_len=len(features)))
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


def to_categories(data, key, inv, default=None, categorical=True):
    fields = (mapget(key, sentence) for sentence in data)
    cat_seq = [build_sequence(f, inv, default=default) for f in fields]
    padded = pad_sequences(cat_seq)
    if categorical:
        return to_categorical(padded)
    return padded


def simple_eval(pred, gold, out_index):
    actual = Counter()
    correct = Counter()
    for p, g in zip(pred, gold):
        for a,b in zip_from_end(p, g):
            actual_tag = out_index[np.argmax(b)]
            actual[actual_tag] += 1
            if np.argmax(a) == np.argmax(b):
                correct[actual_tag] += 1

    for k in actual:
        print('-'.join(k), correct[k]/actual[k], actual[k])

    corr_sum = sum(correct[k] for k in correct if k != ('O', 'NOE', 'OUT'))
    act_sum = sum(actual[k] for k in actual if k != ('O', 'NOE', 'OUT'))
    print(corr_sum/act_sum)


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

    embed = load_glove(args.glove)
    embed_len = len(next(iter(embed.values())))
    corenlp, docs = read_and_extract(args.file, lambda docs: get_core_nlp(docs, lang=args.lang))

    jar = None
    if args.model.exists():
        jar = ModelJar.load(args.model, lambda jar: emb_mat_init(embed, jar.mappings['form']))
        train, lbl_sets, gold, cats, span_index, doc_index = docria_extract(corenlp, docs, saved_cats=jar.cats)
        mappings = jar.mappings
    else:
        train, lbl_sets, gold, cats, span_index, doc_index = docria_extract(corenlp, docs)
        mappings = create_mappings(train, embed, lbl_sets)

    x_word = to_categories(train, 'form', mappings['form'], default=1, categorical=False)
    x_pos = to_categories(train, 'pos', mappings['pos'])
    x_ne = to_categories(train, 'ne', mappings['ne'])
    y = pad_sequences(gold)

    if not args.model.exists():
        model = make_model(x_word.shape[1], [x_word, x_pos, x_ne], y, embed, mappings['form'], len(mappings['pos']), len(mappings['ne']), len(cats), embed_len, epochs=10)
        jar = ModelJar(model, mappings, cats, path=args.model)
    else:
        model = jar.model

    jar.save()

    if args.predict:
        core_nlp_test, docs_test = read_and_extract(args.predict, lambda docs: get_core_nlp(docs, lang=args.lang))
        test, _, gold_test, _, _, _ = docria_extract(core_nlp_test, docs_test)
        x_word_test = to_categories(test, 'form', mappings['form'], default=1, categorical=False)
        x_pos_test = to_categories(test, 'pos', mappings['pos'])
        x_ne_test = to_categories(test, 'ne', mappings['ne'])
        y_test = pad_sequences(gold)
        pred = model.predict([x_word_test, x_pos_test, x_ne_test])
        simple_eval(pred, gold_test, inverted(cats))
