from docria.storage import DocumentIO
from argparse import ArgumentParser
from pathlib import Path
from utils import take_twos, langforia
import regex as re
import os
import json



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--docria', type=Path)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--tsv', action='store_true')
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

def gold(doc):
    for node in doc.layers['tac/entity/gold']:
        entity = node.fld.text
        if entity:
            print(entity.start, entity.stop, entity, node.fld.label, node.fld.type)
    print(doc.texts['main']) 

def index_layer(nodes):
    layers = ['NamedEntity', 'Token', 'Sentence']
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
        format = 'tsv' if args.tsv else 'json'
        with DocumentIO.read(args.docria) as doc_reader:
            output = 'out'
            if not os.path.exists(output):
                os.makedirs(output)
            for i, doc in enumerate(doc_reader):
                docforia = langforia(str(doc.text['main']),'en',format=format)
                fname = str(output / Path(args.docria.stem)) + '.%d.%s' % (i,format)
                with open(fname, 'w') as out:
                    if format == 'tsv':
                        out.write(docforia)
                    else:
                        json.dump(docforia, out, indent=2, sort_keys=True)
    if args.json:
        with open(args.json) as json_dump:
            docforia = json.load(json_dump)
            index = index_layer(docforia['DM10']['nodes'])
            for name, layer in index.items():
                for i, v in enumerate(layer.items()):
                    print(name, i, v)
