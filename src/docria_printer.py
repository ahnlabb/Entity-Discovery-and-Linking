from docria.storage import DocumentIO
from argparse import ArgumentParser
from pathlib import Path
import re


def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    return parser.parse_args()

def remove_number(string):
    #pattern = re.compile(r'\d+\. ((\s+ )+) ?\.')
    pattern = re.compile(r'\d+\. (.+)')
    match = re.match(pattern, string)
    if match:
        return match.group(1)
    else:
        return string

if __name__ == '__main__':
    args = get_args()
    with DocumentIO.read(args.file) as doc:
        for i, d in enumerate(doc):
            # split by newline and remove empty lines
            text = [s for s in str(d.texts['main']).split('\n') if len(s) > 0]
            for j, t in enumerate(text):
                text = remove_number(t)
                print('%d.%d:' % (i,j), remove_number(t))
