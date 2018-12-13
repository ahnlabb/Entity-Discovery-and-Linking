from docria.storage import DocumentIO
from docria.algorithm import span_translate
from argparse import ArgumentParser
from pathlib import Path 


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('src', type=str)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--text', type=str)
    return parser.parse_args()

def print_texts(field):
    text = field.get(args.src, field.get('xml', ''))
    if text:
        span = (str(text.start), str(text.stop))
        text_field = field.get('text')
        if text_field:
            text_span = (str(text_field.start), str(text_field.stop))
        else:
            text_span = ('', '')
        print('-'.join(span), '->', '-'.join(text_span) + ':', ' '.join(str(text).split()))
        
def txt2xml(doc):
    index = {}
    for node in doc.layers['tac/segments']:
        xstart, xstop = node.fld.xml.start, node.fld.xml.stop
        text = node.fld.text
        if text:
            tstart, tstop = text.start, text.stop
            i, x = tstart, xstart
            for word in str(text).split():
                j, y = i + len(word), x + len(word)
                index[i, j] = x, y
                i, x = j + 1, y + 1

if __name__ == '__main__':
    args = get_args()
    
    if args.layer and args.text:
        raise ValueError("You can't choose both layer and text!")
    if not (args.layer or args.text):
        raise ValueError("You have to choose either a layer or a text!")
        
    with DocumentIO.read(args.file) as docria:
        for doc in list(docria)[:1]:
            print(doc.props['docid'])
            if args.layer:
                layer = doc.layers[args.layer]
                for node in layer:
                    print_texts(node)
            elif args.text:
                field = doc.texts[args.text]
                for node in field._offsets:
                    print_texts(node)
