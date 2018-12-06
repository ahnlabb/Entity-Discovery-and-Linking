from pathlib import Path
from keras.utils import to_categorical
from utils import trans_mut_map, inverted
from docria.storage import DocumentIO
from argparse import ArgumentParser
from itertools import product


def one_hot(index, categories):
    n_cls = len(categories)
    to_cat = lambda x: to_categorical(x, num_classes=n_cls)
    for k in index:
        trans_mut_map(index[k], categories, to_cat)

def from_one_hot(vector, categories):
    assert len(vector) == len(categories)
    invind = inverted(categories)
    for i,v in enumerate(vector):
        if v == 1:
            return invind[i]
    raise ValueError("Zero vector")


def json_compatible(index):
    def make_dict(span, entity):
        return {'start': span[0], 'stop': span[1],
                'class': '-'.join(entity)}
    return [make_dict(s,e) for s,e in index.items()]

def gold_std_idx(docria):
    labels, types = set(), set()
    index = {}
    for doc in docria:
        doc_index = {}
        keys = {}
        longest = None
        for node in doc.layers['tac/entity/gold']:
            labels.add(node.fld.label)
            types.add(node.fld.type)
            entity = node.fld.text
            # ignore xml-only entities
            if entity:
                span = (entity.start, entity.stop)
                
                # TODO: simplify logic if possible
                if longest:
                    # there is a greater span with same start
                    if span[0] == longest[0] and span[1] > longest[1]:
                        del keys[longest]
                        longest = span
                    # we are either inside or after the current span
                    if span[0] > longest[0]:
                        # this should never happen (overlapping spans)
                        if span[0] <= longest[1] and span[1] > longest[1]:
                            continue
                            raise ValueError("Span %s and Span %s are overlapping"
                                            % (longest, span))
                        # we are inside the span
                        if span[1] <= longest[1]:
                            continue
                        # we have a new span
                        else:
                            longest = span
                else:
                    longest = span
                
                keys[span] = node
                
        for key, node in keys.items():
            def word_spans(key, node):
                begin, end = key
                span_index = {}
                words = str(node.fld.text.text)[begin:end].split()
                i = begin
                for k, word in enumerate(words, 1):
                    tag = 'I'
                    if k == 1:
                        tag = 'B'
                    if len(words) == 1:
                        tag = 'S'
                    elif k == len(words):
                        tag = 'E'
                    span_index[(i, i + len(word))] = (tag, node.fld.type, node.fld.label)
                    i += len(word) + 1
                return span_index
            
            for s,e in word_spans(key, node).items():
                doc_index[s] = e
                                 
        index[doc.props['docid']] = doc_index
    # TODO: ensure ordering is consistent between function calls
    tags = {'B', 'E', 'I', 'S'}
    categories = {pair: index for index, pair in enumerate(product(sorted(tags), sorted(types), sorted(labels)))}
    categories[('O','NOE','OUT')] = len(categories)
    categories[('O','NOE','PAD')] = len(categories)
    return index, categories

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
