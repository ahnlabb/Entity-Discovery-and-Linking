from pathlib import Path
#from keras.utils import to_categorical
from utils import trans_mut_map
from docria.storage import DocumentIO
from argparse import ArgumentParser
from itertools import product


def one_hot(index, categories):
    n_cls = len(categories)
    to_cat = lambda x: to_categorical(x, num_classes=n_cls)
    for i in range(len(index)):
        index[i] = trans_mut_map(index[i], categories, to_cat)

def to_categories(index, categories):
    for i in range(len(index)):
        index[i] = trans_mut_map(index[i], categories)

def json_compatible(index):
    def make_dict(span, entity):
        return {'start': span[0], 'stop': span[1],
                'class': '-'.join(entity)}
    return [make_dict(s,e) for s,e in index.items()]

def gold_std_idx(docria):
    labels, types = set(), set()
    index = {}
    with DocumentIO.read(docria) as doc_reader:
        for doc in doc_reader:
            doc_index = {}
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
                            del doc_index[longest]
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
                    doc_index[span] = (node.fld.type, node.fld.label)
            index[doc.props['docid']] = doc_index
    # TODO: ensure ordering is consistent between function calls
    categories = {pair: index for index, pair in enumerate(product(types, labels))}
    return index, categories

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--file', type=Path)
    return parser.parse_args()


def get_doc_index(docria, gold_std):
    index = {}
    with DocumentIO.read(docria) as reader:
        for doc in reader:
            docid = doc.props['docid']
            index[docid] = {'text': str(doc.texts['main']),
                            'gold': json_compatible(gold_std[docid])}
    return index


if __name__ == '__main__':
    args = get_args()
    if args.file:
        gold_std, _ = gold_std_idx(args.file)
        index = get_doc_index(args.file, gold_std)
        print(index)
