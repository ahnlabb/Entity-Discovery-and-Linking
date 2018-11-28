from pathlib import Path
from pickle import load, dump
import requests
import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
    """DOES NOT WORK !!!"""
    if not num_classes:
        num_classes = len(set(y))
    if type(dtype) is str:
        dtype = np.sctypeDict[dtype]
    array = np.zeros((len(y), num_classes), dtype=dtype)
    for i,k in enumerate(y):
        array[i][k-1] = 1
    return array

def trans_mut_map(ab, bc, f=lambda x: x):
    for k in ab:
        ab[k] = f(bc[ab[k]])

def trans_map(ab, bc, f=lambda x: x):
    return {k: f(bc[v]) for k,v in ab.items()}

def langforia_url(lang, config, format='json'):
    return f'http://vilde.cs.lth.se:9000/{lang}/{config}/api/{format}'

def langforia(text, lang, config='corenlp_3.8.0', format='tsv'):
    url = langforia_url(lang, config, format=format)
    headers = {"Content-Type": "application/text; charset=UTF-8"}
    request = requests.post(url, data=text.encode('utf-8'), headers=headers)
    if format == 'tsv':
        return request.text
    return request.json()

def pickled(fun):
    def wrapper(path):
        if path.suffix == '.pickle':
            with path.open('r+b') as f:
                return load(f)
        else:
            result = fun(path)
            with path.with_suffix('.pickle').open('w+b') as f:
                dump(result, f)
            return result
    return wrapper

def take_twos(iterable):
    itr = iter(iterable)
    while True:
        yield next(itr), next(itr)

