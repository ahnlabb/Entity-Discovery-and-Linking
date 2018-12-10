from pathlib import Path
from pickle import load, dump
from functools import reduce
import requests
import numpy as np


def flatten_once(iterable):
    return list(reduce(lambda a,b: a+b, iterable))

def inverted(a):
    return {v:k for k,v in a.items()}

def build_sequence(l, invind, default=None):
    if default:
        return [invind.get(w, default) for w in l]
    return [invind[w] for w in l]

def map2(fun, x, y):
    return fun(x[0], y[0]), fun(x[1], y[1])

def print_dims(data):
    """powerful debugging solution"""
    try:
        try:
            print(type(data), str(data.shape), end=' ')
        except:
            print(type(data), '(' + str(len(data)) + ')', end=' ')
        print_dims(data[0])
    except:
        print()
        return

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
    def wrapper(path, *args):
        if path.suffix == '.pickle':
            with path.open('r+b') as f:
                return load(f)
        else:
            result = fun(path, *args)
            with path.with_suffix('.pickle').open('w+b') as f:
                dump(result, f)
            return result
    return wrapper

def take_twos(iterable):
    itr = iter(iterable)
    while True:
        yield next(itr), next(itr)

