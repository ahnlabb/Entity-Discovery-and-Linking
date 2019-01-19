from argparse import ArgumentParser
from pathlib import Path

from docria.algorithm import span_translate
from docria.storage import DocumentIO


def docria_to_neleval(docs, layer, no_outside=True):
    rows = []
    i = 0
    for doc in docs:
        docid = doc.props['docid']
        main = doc.texts['xml']
        for node in doc.layer[layer]:
            if no_outside and node.fld.label == 'OUT':
                continue
            start, stop = node.fld.xml.start, node.fld.xml.stop
            form = ' '.join(str(main[start:stop]).split())
            row = [
                'XYZ',
                format(i, '05d'), form,
                docid + ':' + str(start) + '-' + str(stop - 1),
                node.fld.target, node.fld.label, node.fld.type, '1.0'
            ]
            rows.append('\t'.join(row))
            i += 1
    if rows:
        return '\n'.join(rows)
    return ''


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('layer', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.file:
        with DocumentIO.read(args.file) as docria:
            print(docria_to_neleval(docria, args.layer))
