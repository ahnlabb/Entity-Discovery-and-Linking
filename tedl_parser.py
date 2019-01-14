import fileinput
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('type')
parser.add_argument('out')
parser.add_argument('recall', type=float)
parser.add_argument('prec', type=float)
args = parser.parse_args()

sep = '*'*61

with fileinput.input(files=args.files) as f:
    runs = ''.join(l for l in f).split(sep)[1:]

    x, y = [], []
    for r in runs:
        for s in r.split('\n\n'):
            for line in s.split('\n'):
                if args.type in line:
                    prec, recall = line.split()[:2]
                    prec, recall = float(prec), float(recall)
                    if prec > 0 and recall > 0:
                        print('prec', prec, 'recall', recall)
                        x.append(recall)
                        y.append(prec)

    x, y = np.array(x), np.array(y)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(args.recall, args.prec, 'ro')
    plt.scatter(x, y)
    plt.savefig(args.out)
