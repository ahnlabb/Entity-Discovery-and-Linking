from docria.storage import DocumentIO
from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as et
import regex as re
import requests
import json


def langforia_url(lang, config, format='json'):
    return f'http://vilde.cs.lth.se:9000/{lang}/{config}/api/{format}'


def langforia(text, lang, config='corenlp_3.8.0'):
    url = langforia_url(lang, config, format='json')
    request = requests.post(url, data=text)
    return json.dumps(request.json()['DM10']['nodes'], indent=4, sort_keys=True)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
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

if __name__ == '__main__':
    args = get_args()
    with DocumentIO.read(args.file) as doc_reader:
        for doc in list(doc_reader)[:1]:
            print(langforia(str(doc.text['main']).encode('utf-8'), 'en'))
#           text = ''
#           for i, segment in enumerate(doc.layer['tac/segments']):
#               line = str(segment.fld.text)
#               if line != '\n':
#                   line.replace('\n', ' ')
#                   text += line
#           for sentence in text.split('.'):
#               if len(sentence) > 1:
#                   print(sentence.lstrip().rstrip() + '.')
