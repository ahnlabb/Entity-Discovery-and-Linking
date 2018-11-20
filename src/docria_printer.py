from docria.storage import DocumentIO
from argparse import ArgumentParser
from pathlib import Path
from utils import take_twos

import xml.etree.ElementTree as et
import regex as re
import requests
import json


def langforia_url(lang, config, format='json'):
    return f'http://vilde.cs.lth.se:9000/{lang}/{config}/api/{format}'

def langforia(text, lang, config='corenlp_3.8.0'):
    url = langforia_url(lang, config, format='json')
    request = requests.post(url, data=text)
    return request.json()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--docria', type=Path)
    parser.add_argument('--json', type=Path)
    return parser.parse_args()

def remove_number(string):
    pattern = re.compile(r'\d+\. (.+)')
    match = re.match(pattern, string)
    if match:
        return match.group(1)
    else:
        return string

def print_entity(node):
    print('Entity:', node.fld.xml)
    print('Label:', node.fld.label)
    print('Type:', node.fld.type)
    print() 

def give_context(doc, gold_node):
    gold_span = (gold_node.fld.xml.start, gold_node.fld.xml.stop)
    for node in doc.layers['tac/segments']:
        seg_span = (node.fld.xml.start, node.fld.xml.stop)
        if gold_span[0] >= seg_span[0] and gold_span[1] <= gold_span[1]:
            text = node.fld.xml.text
            context = text[gold_span[0]-10:gold_span[1]+10]
            return context

def index_layer(nodes):
    layers = ['NamedEntity', 'Token']
    index = {}
    for d in nodes:
        layer = d['layer'].split('.')[-1]
        if layer in layers:
            index[layer] = {}
            properties = d['nodes'][0]['properties']
            ranges = take_twos(d['nodes'][0]['ranges'])
            for p in properties:
                begin, end = next(ranges)
                index[layer][begin] = p
                index[layer][end] = p
    return index

if __name__ == '__main__':
    args = get_args()
    if args.docria:
        with DocumentIO.read(args.docria) as doc_reader:
            for doc in list(doc_reader)[:1]:
                docforia = langforia(str(doc.text['main']).encode('utf-8'), 'en')
                print(json.dumps(docforia['DM10'], indent=2, sort_keys=True))
    if args.json:
        with open(args.json) as json_dump:
            docforia = json.load(json_dump)
            index = index_layer(docforia['nodes'])
            for i, v in index['Token'].items():
                print(i, v)
