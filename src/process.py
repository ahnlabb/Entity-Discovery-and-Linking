#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from json import loads
from docria.storage import DocumentIO
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
        return {row[0]: np.array(float(v) for v in row[1:]) for row in rows}


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


def process(doc, lang):
    for a in doc:
        corenlp = langforia(str(a.texts['main']).encode('utf-8'), lang)
        print(corenlp)
        # for layer in corenlp['nodes']:
            # print(layer['layer'])
            # print(layer)
            # continue
            # for start, end in take_twos(layer['nodes'][0]['ranges']):
                # print(corenlp['text'][start:end])


if __name__ == '__main__':
    args = get_args()
    for arg in vars(args).values():
        try:
            assert(arg.exists())
        except AssertionError:
            raise FileNotFoundError(arg)
        except AttributeError:
            pass
    with DocumentIO.read(args.file) as doc:
        process(doc, 'en')
