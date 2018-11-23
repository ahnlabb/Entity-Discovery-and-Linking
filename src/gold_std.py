from keras import to_categorical
from utils import trans_mut_map


def one_hot(index, categories):
    n_cls = len(categories.values())
    for doc_index in index:
        to_cat = lambda x: to_categorical(x, num_classes=n_cls)
        trans_mut_map(doc_index, categories, to_cat)

def to_categories(index, categories):
    for doc_index in index:
        trans_mut_map(doc_index, categories, lambda x: x)

def gold_std_idx(doc_reader):
    labels, types = set(), set()
    index = []
    for doc in doc_reader[:1]:
        doc_index = {}
        longest = None
        for node in doc.layers['tac/entity/gold']:
            labels.add(node.fld.label)
            types.add(node.fld.type)
            entity = node.fld.text
            # ignore xml-only entities (for now)
            if entity:
                span = (entity.start, entity.stop)
                if longest:
                    if span[0] == longest[0] and span[1] > longest[1]:
                        del doc_index[longest]
                        longest = span
                    if span[0] > longest[0]:
                        if span[1] <= longest[1]:
                            continue
                        else:
                            longest = span
                else:
                    longest = span
                doc_index[span] = (node.fld.type, node.fld.label)
        index.append(doc_index)
    categories = {pair: index for index, pair in enumerate(product(types, labels))}
    return index, categories
