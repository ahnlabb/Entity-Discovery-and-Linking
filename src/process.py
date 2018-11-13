#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from docria.storage import DocumentIO
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
#    parser.add_argument('glove', type=Path)
    return parser.parse_args()


def load_glove(path):
    with path.open('r') as f:
        rows = map(lambda x: x.split(), f)
        return {row[0]: np.array(float(v) for v in row[1:]) for row in rows}


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass
    doc = DocumentIO.read(args.file)
    for d in doc:
        if d.texts and 'main' in d.texts:
            entity_name = d.props['title']
            print(entity_name)
            print('-' * len(entity_name)) # pretty printing
            for block in str(d.texts['main']).split('\n'):
                for sentence in block.split('.'):
                    if entity_name in sentence:
                        print(sentence.lstrip() + '.')
            print()
