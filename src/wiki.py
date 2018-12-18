import base64
from pathlib import Path
from argparse import ArgumentParser
from pickle import dump, load

def get_args():
    parser = ArgumentParser()
    parser.add_argument('wiki_map', type=Path, help='')
    parser.add_argument('wkd2fb', type=Path, help='')
    parser.add_argument('out', type=Path, help='')
    return parser.parse_args()


def get_wikimap(wiki_dir, wkd2fb):
    with wkd2fb.open('r+b') as f:
        fbmap = load(f)
    wikimap = {}
    for path in wiki_dir.glob('part-*'):
        with path.open('r') as f:
            for line in f:
                split = line.rfind(',')
                try:
                    wkd = int(line[split+1:])
                    wikimap[line[:split]] = fbmap.get(wkd, 'wkd'+str(base64.b64encode(wkd.to_bytes((wkd.bit_length() + 7) // 8, byteorder='little'))))
                except ValueError:
                    pass
    return wikimap


if __name__ == '__main__':
    args = get_args()
    with args.out.open('w+b') as f:
        dump(get_wikimap(args.wiki_map, args.wkd2fb), f)
