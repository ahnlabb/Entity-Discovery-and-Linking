from pathlib import Path
from keras.utils import to_categorical
from utils import trans_mut_map, inverted, flatten_once
from docria.storage import DocumentIO
from docria.algorithm import span_translate
from argparse import ArgumentParser
from itertools import product
import numpy as np


def one_hot(index, categories):
    n_cls = len(categories)
    def to_cat(x):
        return to_categorical(x, num_classes=n_cls)
    for k in index:
        trans_mut_map(index[k], categories, to_cat)
    old_cats = dict(categories)
    for key, val in categories.items():
        categories[key] = to_cat(val)
    return old_cats

def from_one_hot(vector, categories):
    assert len(vector) == len(categories)
    invind = inverted(categories)
    for i,v in enumerate(vector):
        if v == 1:
            return invind[i]
    raise ValueError("Zero vector")
    
def interpret_prediction(y, cats):
    one_hot = np.array([int(x) for x in y == max(y)])
    return from_one_hot(one_hot, cats)

def entity_to_dict(start, stop, entity):
    return {'start': int(start), 'stop': int(stop), 'class': '-'.join(entity)}

def json_compatible(index):
    return [entity_to_dict(s[0], s[1], e) for s,e in index.items()]

def gold_std_idx(docria):
        
    labels, types = set(), set()
    index = {}
    spandex = {}
    
    def longest_match(entity, longest, keys=None):
        if entity:
            span = (entity.start, entity.stop)
            if longest:
                if keys and span[0] == longest[0] and span[1] > longest[1]:
                    del keys[longest]
                if span[0] > longest[0]:
                    if span[1] <= longest[1] or span[0] <= longest[1] and span[1] > longest[1]:
                        return
            return span
            
    def tag_spans(keys, doc_index, text_spans):
        for key, nt in keys.items():
            def word_spans(key, nt):
                node, text_span = nt
                begin, end = key
                span_index = {}
                xml = str(node.fld.xml.text)[begin:end]
                words = xml.split()
                i, k = begin, text_span[0] if text_span else None
                for wc, word in enumerate(words, 1):
                    tag = 'I'
                    if wc == 1:
                        tag = 'B'
                    if len(words) == 1:
                        tag = 'S'
                    elif wc == len(words):
                        tag = 'E'
                    j, l = i + len(word), k + len(word) if k else None
                    # span_index :: xml span -> (output class, text span)
                    span_index[(i, j)] = ((tag, node.fld.type, node.fld.label),
                                          (k, l) if k else None)
                    i, k = j + 1, l + 1 if l else None
                return span_index

            for s, e in word_spans(key, nt).items():
                doc_index[s] = e[0]
                text_spans[e[1]] = s

    for doc in docria:
        doc_index, text_spans_index, keys_xml = {}, {}, {}
        longest_xml, longest_text = None, None
        for node in doc.layers['tac/entity/gold']:
            labels.add(node.fld.label)
            types.add(node.fld.type)
            
            xml_span = longest_match(node.fld.xml, longest_xml, keys_xml)
            text_span = longest_match(node.fld.text, longest_text)
            if xml_span:
                #print(xml_span, ' '.join(str(node.fld.xml).split()), node.fld.label, node.fld.type)
                longest_xml = xml_span
                if text_span:
                    longest_text = text_span
                    node_textspan = (node, text_span)
                else:
                    node_textspan = (node, None)
                keys_xml[xml_span] = node_textspan
                
        tag_spans(keys_xml, doc_index, text_spans_index)
            
        index[doc.props['docid']] = doc_index
        spandex[doc.props['docid']] = text_spans_index
        #print(sum(len(d) for d in index.values()))
        
    tags = {'B', 'E', 'I', 'S'}
    categories = {pair: index for index, pair
                  in enumerate(product(sorted(tags), sorted(types), sorted(labels)))}
    
    outside, padding = ('O','NOE','OUT'), ('O','NOE','PAD')
    categories[outside] = len(categories)
    categories[padding] = len(categories)
    
    return index, categories, spandex


def to_neleval(classes, span_index, doc_index, cats, iteration, include_outside=False):
    rows = []
    k = 0
    for cls,span,docid in zip(*map(flatten_once, (classes, span_index, doc_index))):
        cls = interpret_prediction(cls, cats)
        if cls[0] == 'O' and not include_outside:
            continue
        start, stop = str(span[0]), str(span[1] + 1)
        entity_id = 'NIL_' + str(iteration * len(classes) + k)
        row = docid + '\t' + start + '\t' + stop + '\t' + entity_id + '\t' + '1.0' + '\t' + cls[2]
        rows.append(row)
        k += 1
    if rows:
        return '\n'.join(rows) + '\n'
    return ''

def gold2vec(docria):
    def wrapper():
        index, categories = gold_std_idx(docria)
        index = one_hot(index, categories)
        return index

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--file', type=Path)
    return parser.parse_args()


def get_doc_index(docria, gold_std):
    
    index = {}
    for doc in docria:
        docid = doc.props['docid']
        index[docid] = {'text': str(doc.texts['main']),
                        'gold': json_compatible(gold_std[docid])}
    return index


if __name__ == '__main__':
    args = get_args()
    if args.file:
        with DocumentIO.read(args.file) as docria:
            gold_std, cats = gold_std_idx(docria)
            for i,doc in gold_std.items():
                for span, entity in doc.items():
                    print(span, entity)
