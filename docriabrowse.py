import curses
from itertools import zip_longest, chain
from argparse import ArgumentParser
from pathlib import Path
from docria.storage import DocumentIO
from docria.model import Document, Node, Text, NodeLayerCollection


ESC = 27

def get_args():
    parser = ArgumentParser()
    parser.add_argument('docria', type=Path)
    parser.add_argument('-n', '--name_property', default='id')
    return parser.parse_args()

class IterViewer(object):
    def min(self):
        return 0

    def __len__(self):
        return 0

class ListDecorator(IterViewer):
    def __init__(self, itr):
        self.lst = list(itr)

    def __getitem__(self, index):
        return self.lst[index]

    def __setitem__(self, index):
        pass

    def __delitem__(self, index):
        pass

    def append(self, item):
        self.lst.append(item)

    def min(self):
        return 0

    def __len__(self):
        return len(self.lst)


class DocIter(ListDecorator):
    def __init__(self, docs, size, name_property):
        self.start = 0
        self.end = size
        self.size = size
        self.docs = docs
        self.name_property = name_property
        super().__init__(next(docs) for _ in range(size))

    def __iter__(self):
        for doc in self.lst:
            yield doc.props[self.name_property]


class DocViewer(ListDecorator):
    def __init__(self, doc):
        self.doc = doc
        super().__init__(chain(self.doc.props.values(), self.doc.layers.values(), self.doc.texts.values()))

    def __iter__(self):
        for key in self.doc.props:
            yield 'prop:  {}: {}'.format(key, self.doc.props[key])
        for key in self.doc.layers:
            yield 'layer: {}'.format(key)
        for key in self.doc.text:
            yield 'text: {}'.format(key)

    def min(self):
        return len(self.doc.props)


class NodeCollectionViewer(ListDecorator):
    def __init__(self, node_collection):
        super().__init__(iter(node_collection))

    def __iter__(self):
        for node in self.lst:
            yield f'{node._id}: {compact_node(node)}'

def compact_node(node):
    if 'target' in node:
        return node['target']
    if 'text' in node:
        return node['text']
    return repr(dict(node.items()))

class VarsViewer(IterViewer):
    def __init__(self, obj):
        self.varlist = vars(obj)

    def __iter__(self):
        for k, v in self.varlist.items():
            yield f'{k}: {v}'

class NodeViewer(ListDecorator):
    def __init__(self, obj):
        super().__init__(obj.items())

    def __iter__(self):
        for k, v in self.lst:
            yield f'{k}: {v}'


class TypeViewer(IterViewer):
    def __init__(self, obj):
        self.obj = obj
    def __iter__(self):
        yield str(type(self.obj))

def wrap(obj):
    if type(obj) is Document:
        return DocViewer(obj)
    if type(obj) is NodeLayerCollection:
        return NodeCollectionViewer(obj)
    if type(obj) is Node:
        return NodeViewer(obj)
    return TypeViewer(obj)


def create_main_loop(doc):
    return lambda stdscr: main(stdscr, doc)

def render(line, cols):
    return ('{:<%d}' % cols).format(line)

def main(stdscr, docs):
    stdscr.nodelay(0)
    curidx = 0

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    top = docs
    parents = []

    while True:
        rows = curses.LINES
        curidx = min(max(top.min(), curidx), rows)
        offset = max(0, curidx - rows + 3)
        cur = wrap(top[curidx])

        lines = zip_longest(top, cur, fillvalue='')
        for i in range(rows - 1):
            try:
                doc, meta = next(lines)
            except StopIteration:
                doc, meta = '', ''

            if i == curidx:
                stdscr.attrset(curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.attrset(curses.color_pair(0))

            right = curses.COLS // 2 - 1
            left = curses.COLS - right

            stdscr.addstr(i, 0, render(doc, left))

            stdscr.attrset(curses.color_pair(0))

            stdscr.addstr(i, left + 1, '|')
            stdscr.addstr(i, left + 2, render(meta, right))

        stdscr.refresh()
        ch = stdscr.getch()
        action = actions[ch]
        if action == 'up':
            curidx -= 1
        elif action == 'down':
            curidx += 1
            if curidx > rows:
                shift = 1
                offset += shift
                curidx -= shift
        elif action == 'prev_page':
            curidx -= rows
            if curidx < 0: curidx = 0
        elif action == 'next_page':
            curidx += rows
        elif action == 'exit':
            return
        elif action == 'forward':
            if cur.min() < len(cur):
                parents.append((top, curidx))
                top = cur
                curidx = 0
                stdscr.clear()
        elif action == 'back':
            top, curidx = parents.pop()
            stdscr.clear()

actions = {
        curses.KEY_UP: 'up',
        ord('k'): 'up',
        curses.KEY_DOWN: 'down',
        ord('j'): 'down',
        ord('q'): 'exit',
        ord('\n'): 'forward',
        ord('l'): 'forward',
        ord('h'): 'back'
        }

if __name__ == '__main__':
    args = get_args()
    with DocumentIO.read(args.docria) as doc_reader:
        loop = create_main_loop(DocIter(doc_reader, 100, args.name_property))

    curses.wrapper(loop)
