#!/usr/bin/env python3
import sys
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
from itertools import count
from pathlib import Path
from pickle import load
from random import shuffle

import numpy as np
import regex as re

from docria import T
from docria.algorithm import span_translate
from docria.storage import DocumentIO
from gold_std import entity_to_dict, gold_std_idx, interpret_prediction
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from print_neleval import docria_to_neleval
from structs import ModelJar
from utils import (build_sequence, existing_path, inverted, langforia, mapget,
                   pickled, print_dims, zip_from_end)

keys = [('form', {
    'default': 1,
    'categorical': False
}), ('pos', {
    'categorical': False
}), ('ne', {
    'categorical': False
}), ('capital', {}), ('special', {})]


def get_args():
    """Parses the command line arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        'file',
        type=existing_path,
        nargs='+',
        help='one or more docria files to use for training')
    parser.add_argument(
        'glove',
        type=existing_path,
        help='file with embedding (tab or space separated, no header)')
    parser.add_argument(
        'model',
        type=Path,
        help='path to where model will be saved, '
        'if this file already exists training will be skipped and this model will be loaded'
    )
    parser.add_argument(
        'lang', type=str, choices=['en', 'zh', 'es'], help='model language')
    parser.add_argument(
        'elmapping',
        type=existing_path,
        help='mapping file for entity linking')
    parser.add_argument(
        '--predict',
        type=existing_path,
        help='docria file to use for prediction')
    return parser.parse_args()


@pickled
def read_and_extract(path, fun):
    with DocumentIO.read(path) as doc:
        return fun(list(doc))


@pickled
def load_glove(path):
    with path.open('r') as f:
        rows = map(lambda x: x.split(), f)
        embed = [(row[0], np.asarray(row[1:], dtype='float32'))
                 for row in rows]
        return embed


def get_core_nlp(docs, lang):
    def call_api(doc):
        text = str(doc.texts['main'])
        return langforia(text, lang).split('\n')

    api_data = []
    start = time.perf_counter()
    for i, doc in enumerate(docs, 1):
        print(
            "Document %d/%d " % (i, len(docs)),
            end='',
            flush=True,
            file=sys.stderr)
        api_data.append(call_api(doc))
        time_left = int((time.perf_counter() - start) / i * (len(docs) - i))
        print(
            f' ETA: {time_left // 60} min {time_left % 60} s    ',
            end='\r',
            flush=True,
            file=sys.stderr)
    time_tot = int(time.perf_counter() - start)
    print(
        f'Done: {len(docs)} documents in {time_tot // 60} min {time_tot % 60} s',
        file=sys.stderr)
    return api_data, docs


def txt2xml(doc):
    index = {}
    for node in doc.layers['tac/segments']:
        text = node.fld.text
        if text:
            i, j = text.start, text.stop
            x, y = node.fld.xml.start, node.fld.xml.stop
            indices = enumerate([(x, y)] * (j - i), i)
            for txt, xml in indices:
                index[txt] = xml
    #for s, e in index.items():
    #    print(s, e)
    return index


def docria_extract(core_nlp, docs, per_doc=False):
    train, gold, spandex = [], [], []
    lbl_sets = defaultdict(set)

    gold_std, cats = gold_std_idx(docs)

    def get_entity(docid, span):
        none = ('O', 'NOE', 'OUT')
        return gold_std[docid].get(span, none)

    for cnlp, doc in zip(core_nlp, docs):
        #layer = doc.add_layer('corenlp', text=T.span('main'), xml=T.span('xml'))
        docid = doc.props['docid']
        #main = doc.text['main']

        sentences, spans = core_nlp_features(cnlp, lbl_sets)
        #for sentence_spans in spans:
        #    for span in sentence_spans:
        #        layer.add(text=main[span])
        #span_translate(doc, 'tac/segments', ('xml', 'text'), 'corenlp', ('text', 'xml'))
        entities = [[get_entity(docid, s) for s in sentence]
                    for sentence in spans]

        if per_doc:
            gold.append(entities)
            train.append(sentences)
            spandex.append(spans)
        else:
            gold.extend(entities)
            train.extend(sentences)
            spandex.extend(spans)

    return train, lbl_sets, gold, cats, spandex, list(docs)


def build_indices(train, embed, embed_n=None):
    word_counts = Counter(
        features['form'] for sentence in train for features in sentence)
    if embed_n:
        tot = sum(word_counts.values())
        word_inv = {}
        i = 2
        for word, cnt in word_counts.items():
            if cnt * embed_n > tot:
                word_inv[word] = i
                i += 1
        print(i)
        for word, _ in embed:
            if word not in word_inv:
                word_inv[word] = i
                i += 1
    else:
        wordset = set(word_counts)
        wordset.update(word for word, _ in embed)
        word_inv = dict(zip(wordset, count(2)))
    return word_inv


def normalize_form(form):
    form = form.lower()
    if form.isdecimal():
        return '0'
    if '://' in form:
        return '{URL}'
    # zh_chars = re.findall(u'[\u4e00-\u9fff]+', form)
    # if zh_chars:
    # return zh_chars[0]
    return form


def core_nlp_features(corenlp, lbl_sets):
    def strip_tags(cnlp):
        return re.sub(r'<[^>]*>', '', cnlp)

    corenlp = iter(corenlp)
    spans = []

    def add(features, name):
        feat = features[name]
        lbl_sets[name].add(feat)

    head = next(corenlp).split('\t')[1:]
    sentences = [[]]
    spans = [[]]
    inside = ''
    special = [
        'the', 'in', 'of', 'to', 'from', 'by', 'his', 'president', "'s", 'at'
    ]
    lbl_sets['special'] = set(range(len(special) + 1))
    for row in corenlp:
        if row:
            cols = row.split('\t')
            features = dict(zip(head, cols[1:]))
            if 'ne' in features:
                inside, features['ne'] = normalize_ne(inside, features['ne'])
            else:
                features['ne'] = '_'
            features['capital'] = features['form'][0].isupper()
            features['form'] = normalize_form(features['form'])
            features['special'] = special.index(
                features['form']) + 1 if features['form'] in special else 0
            features['form'] = strip_tags(features['form'])
            if not features['form']:
                features['ne'] = '_'
                features['pos'] = 'TAG'
                features['capital'] = False
            add(features, 'pos')
            add(features, 'ne')
            add(features, 'capital')
            add(features, 'special')
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
    mappings = {
        key: dict(zip(lbls, count(1)))
        for key, lbls in lbl_sets.items()
    }
    for key in mappings:
        mappings[key]['OOV'] = 0
    mappings['form'] = build_indices(train, embed, embed_n=40000)
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
                    label = 0
                label_list.append(label)
            labels[feat_name].append(label_list)

    for feat, lbls in mappings.items():
        if feat == 'form':
            continue
        labels[feat] = [
            to_categorical(vals, num_classes=len(lbls))
            for vals in labels[feat]
        ]

    def concat(word):
        if padding:
            word = word + (np.zeros(1), )
        return np.concatenate(word)

    for sentence in zip(*labels.values()):
        yield [concat(word) for word in zip(*sentence)]


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
    elif len(ne) > 1:
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


def batch_generator(train, gold, mappings, keys=keys, batch_len=128):
    xy = sorted(zip(train, gold), key=lambda x: len(x[0]))
    n_batches = len(train) // batch_len + (len(train) % batch_len > 0)
    batches = list(range(n_batches))

    def process_batch(batch_xy):
        maxlen = len(batch_xy[-1][0])
        shuffle(batch_xy)
        x, y = zip(*batch_xy)
        outputs = list(categorical_gen(x, keys, mappings, maxlen))
        return outputs, pad_sequences(y, maxlen=maxlen)

    while True:
        shuffle(batches)
        for k in batches:
            batch_slice = slice(k * batch_len, (k + 1) * batch_len)
            yield process_batch(xy[batch_slice])


def categorical_gen(sentences, keys, mappings, maxlen):
    for key, args in keys:
        yield field_as_category(
            sentences, key, mappings[key], maxlen=maxlen, **args)


def predict_batch_generator(test, mappings, keys=keys):
    for sentences in test:
        maxlen = len(max(sentences, key=len))
        yield list(categorical_gen(sentences, keys, mappings, maxlen))


def reduce_tags(pred, spans, inv_cats):
    ents = []

    def get_next(itr):
        p, s = next(itr)
        start, stop = s
        i = np.argmax(p)
        confidence = p[i]
        tag, tp, lbl = inv_cats[i]
        cls = (tp, lbl)
        return start, stop, tag, cls, confidence

    itr = zip_from_end(pred, spans)
    try:
        while True:
            start, stop, tag, cls, confidence = get_next(itr)
            stack = []
            if tag == 'B':
                cont = True
                stack.append('B')
                while cont:
                    newstart, newstop, newtag, newcls, newconfidence = get_next(
                        itr)
                    stack.append(newtag)
                    if newtag == 'B':
                        start, stop = newstart, newstop
                        cls = newcls
                        continue
                    if newcls == cls:
                        if newtag == 'E':
                            ents.append((start, newstop, cls))
                            cont = False
                        elif newtag == 'I':
                            stop = newstop
                            tag = newtag
                        else:
                            pass
                            # print(tag, newtag, cls)
                    else:
                        # if newtag == 'O' and confidence > newconfidence:
                        # ents.append((start, stop, cls))
                        # else:
                        # print(stack, cls, newcls, main[(start, newstop)], start, newstop)
                        cont = False
            elif tag == 'S':
                ents.append((start, stop, cls))
                cont = False
            else:
                pass
    except StopIteration:
        return ents


def get_tgt(text, elmap, wiki=False):
    index = 1 if wiki else 0
    if str(text) in elmap:
        return elmap[str(text)][index]
    if str(text).lower() in elmap:
        return elmap[str(text).lower()][index]
    capitalized = ' '.join(w.capitalize() for w in str(text).split())
    return elmap.get(capitalized, (None, None))[index]


def predict_to_layer(model,
                     docs,
                     test,
                     spandex,
                     mappings,
                     inv_cats,
                     keys,
                     elmap={}):
    batches = predict_batch_generator(test, mappings, keys)
    i = 0
    nil = {}
    for doc, doc_spans, batch in zip(docs, spandex, batches):
        layer = doc.add_layer(
            'tac/entity', text=T.span('main'), xml=T.span('xml'))
        main = doc.text['main']
        predictions = model.predict_on_batch(batch)

        for pred, spans in zip(predictions, doc_spans):
            ents = reduce_tags(pred, spans, inv_cats)
            for start, stop, (tp, lbl) in ents:
                text = main[(start, stop)]
                tgt = get_tgt(text, elmap)
                if not tgt:
                    string = str(text)
                    if string not in nil:
                        i += 1
                        nil[string] = 'NIL%s' % format(i, '05d')
                    tgt = nil[string]
                layer.add(text=text, type=tp, label=lbl, target=tgt)

        span_translate(doc, 'tac/segments', ('xml', 'text'), 'tac/entity',
                       ('text', 'xml'))
        for match in re.finditer(r' author="([^"]*)"', str(doc.text['xml'])):
            #print(match[0], match[1], match.start(1), match.end(1) - 1)
            text = match[1]
            if text not in nil:
                i += 1
                nil[text] = 'NIL%s' % format(i, '05d')
            tgt = nil[text]
            layer.add(
                text=match[1],
                type='NAM',
                label='PER',
                target=tgt,
                xml=namedtuple('Span', ['start', 'stop'])(match.start(1),
                                                          match.end(1)))


def predict(jar, text, elmap={}, lang='en', padding=False):
    inv = inverted(jar.cats)
    langforia_res = langforia(text, lang).split('\n')
    sentences, _ = core_nlp_features(langforia_res, defaultdict(set))
    batch_gen = predict_batch_generator([sentences], jar.mappings)
    pred_mat = [jar.model.predict_on_batch(x) for x in batch_gen][0]
    pred = interpret_prediction(pred_mat, inv)
    spans = [[(int(word['start']), int(word['end'])) for word in s]
             for s in sentences]

    ents = [reduce_tags(p, s, inv) for p, s in zip(pred_mat, spans)]
    ents = [e for ent in ents for e in ent]
    targets = []
    for start, stop, (tp, lbl) in ents:
        name = text[start:stop]
        targets.append(get_tgt(name, elmap, wiki=True))
    return format_predictions(text, pred, sentences, ents, targets)


def format_predictions(input_text, predictions, sentences, ents, targets):
    ents = [{
        'class': '-'.join(cls),
        'start': start,
        'stop': stop
    } for start, stop, cls in ents]

    pred_dict = {
        'text': input_text,
        'entities': [],
        'reduced': ents,
    }

    for sent in zip(sentences, predictions):
        for word, class_tuple in zip_from_end(*sent):
            start, end = word['start'], word['end']
            entity_dict = entity_to_dict(start, end, class_tuple)
            pred_dict['entities'].append(entity_dict)
    return pred_dict, targets


def field_as_category(data,
                      key,
                      inv,
                      default=None,
                      categorical=True,
                      maxlen=None):
    fields = (mapget(key, sentence) for sentence in data)
    return to_categories(
        fields, inv, default=default, categorical=categorical, maxlen=maxlen)


def to_categories(data, inv, default=None, categorical=True, maxlen=None):
    cat_seq = [build_sequence(d, inv, default=default) for d in data]
    padded = pad_sequences(cat_seq, maxlen=maxlen)
    if categorical:
        categorical = to_categorical(padded, num_classes=len(inv))
        if categorical.ndim == 2:
            a, b = categorical.shape
            categorical = np.reshape(categorical, (a, 1, b))
        return categorical
    return padded


def simple_eval(pred, gold, out_index):
    actual = Counter()
    correct = Counter()
    for p, g in zip(pred, gold):
        for a, b in zip_from_end(p, g):
            actual_tag = out_index[np.argmax(b)]
            actual[actual_tag] += 1
            if np.argmax(a) == np.argmax(b):
                correct[actual_tag] += 1

    for k in actual:
        print('-'.join(k), correct[k] / actual[k], actual[k])

    corr_sum = sum(correct[k] for k in correct if k != ('O', 'NOE', 'OUT'))
    act_sum = sum(actual[k] for k in actual if k != ('O', 'NOE', 'OUT'))
    print(corr_sum / act_sum)


def init_model(embed, files, lang):
    train, lbl_sets, gold, cats = [], defaultdict(set), [], {}

    def _get_core_nlp(docs):
        return get_core_nlp(docs, lang=lang)

    for f in files:
        corenlp, docs = read_and_extract(f, _get_core_nlp)
        t, ls, g, cs, _, _ = docria_extract(corenlp, docs)
        train.extend(t)
        for k in ls:
            lbl_sets[k] |= ls[k]
        gold.extend(g)
        cats.update(cs)
    gold = to_categories(gold, cats)

    mappings = create_mappings(train, embed, lbl_sets)
    embed_len = len(embed[0][1])

    return ModelJar(embed, mappings, cats, embed_len), train, gold


def main(args):
    embed = load_glove(args.glove)

    def filter_short(seq):
        filter(lambda x: len(x) > 1, seq)

    if args.model.exists():
        jar = ModelJar.load(args.model)
    else:
        jar, train, gold = init_model(embed, args.file, args.lang)

        batch_len = 128
        batches = batch_generator(
            train, gold, jar.mappings, keys, batch_len=batch_len)
        jar.train_batches(batches, len(train), epochs=20, batch_size=batch_len)
        #jar.train([x_word, x_pos, x_ne], y, epochs=10, batch_size=batch_len)
        jar.save(args.model)

    with args.elmapping.open('r+b') as f:
        elmap = load(f)

    if args.predict:
        core_nlp_test, docs_test = read_and_extract(
            args.predict, lambda docs: get_core_nlp(docs, lang=args.lang))
        test, _, gold_test, _, spandex, docs = docria_extract(
            core_nlp_test, docs_test, per_doc=True)
        #gold_test = [to_categories(g, jar.cats) for g in gold_test]
        predict_to_layer(
            jar.model,
            docs,
            test,
            spandex,
            jar.mappings,
            inverted(jar.cats),
            keys,
            elmap=elmap)
        with open('predict.%s.tsv' % args.lang, 'w') as f:
            f.write(docria_to_neleval(docs, 'tac/entity'))

        #x_word_test = field_as_category(test, 'form', mappings['form'], default=1, categorical=False)
        #x_pos_test = field_as_category(test, 'pos', mappings['pos'], categorical=False)
        #x_ne_test = field_as_category(test, 'ne', mappings['ne'], categorical=False)
        #y_test = pad_sequences(gold_test)
        #pred = model.predict([x_word_test, x_pos_test, x_ne_test])
        #simple_eval(pred, gold_test, inverted(cats))

    #x_word = field_as_category(train, 'form', mappings['form'], default=1, categorical=False)
    #x_pos = field_as_category(train, 'pos', mappings['pos'], categorical=False)
    #x_ne = field_as_category(train, 'ne', mappings['ne'], categorical=False)
    #y = pad_sequences(gold)


if __name__ == '__main__':
    main(get_args())
