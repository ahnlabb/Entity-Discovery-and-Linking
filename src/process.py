#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
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
        return {row[0]: np.asarray(row[1:], dtype='float32') for row in rows}


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


def process(doc, embed, lang):
    for a in doc:
        text = str(a.texts['main']).encode('utf-8')
        corenlp = iter(langforia(text, lang).split('\n'))
        head = next(corenlp).split('\t')[1:]
        sentences = [{}]
        for row in corenlp:
            if row:
                cols = row.split('\t')
                features = dict(zip(head, cols[1:]))
                sentences[-1][cols[0]] = features
            else:
                sentences.append({})
        if not sentences[-1]:
            sentences.pop(-1)


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
        embed = load_glove(args.glove)
        process(doc, embed, 'en')
