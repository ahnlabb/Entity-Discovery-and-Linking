from pathlib import Path
from pickle import load, dump
import requests

def langforia_url(lang, config, format='json'):
    return f'http://vilde.cs.lth.se:9000/{lang}/{config}/api/{format}'

def langforia(text, lang, config='corenlp_3.8.0', format='tsv'):
    url = langforia_url(lang, config, format=format)
    headers = {"Content-Type": "application/text; charset=UTF-8"}
    request = requests.post(url, data=text.encode('utf-8', headers=headers))
    if format == 'tsv':
        return request.text
    return request.json()

def pickled(path, fun):
    if path.suffix == '.pickle':
        with path.open('r+b') as f:
            return load(f)
    else:
        result = fun(path)
        with path.with_suffix('.pickle').open('w+b') as f:
            dump(result, f)
        return result

def take_twos(iterable):
    itr = iter(iterable)
    while True:
        yield next(itr), next(itr)

