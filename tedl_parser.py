import fileinput
from argparse import ArgumentParser

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

parser = ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('type')
parser.add_argument('out')
parser.add_argument('precrec', type=eval, nargs=2)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

sep = '*' * 61

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
                        if args.verbose:
                            print('prec', prec, 'recall', recall)
                        x.append(recall)
                        y.append(prec)

    plt.figure(num=None, figsize=(5, 5), dpi=500)
    x, y = np.array(x), np.array(y)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.scatter(x, y, marker='x')

    (y1, x1), (y2, x2) = args.precrec
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'o', c='#00EE00')

    xi = np.arange(0.01, 1, 0.01)
    yi = np.arange(0.01, 1, 0.01)
    xx, yy = np.meshgrid(xi, yi, sparse=True)
    z = 2 * yy * xx / (yy + xx)
    plt.contour(xi, yi, z, 10, linestyles='dotted')

    plt.savefig(args.out)
