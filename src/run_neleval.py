from pathlib import Path
from shutil import copy
from subprocess import run
from tempfile import TemporaryDirectory
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('gold', type=Path)
parser.add_argument('prediction', type=Path)
args = parser.parse_args()

with TemporaryDirectory() as tmpdir:
    path = Path(tmpdir)
    system = (path / 'system')
    out = (path / 'out')
    system.mkdir()
    out.mkdir()
    copy(args.prediction, system)

    def is_lang(lang):
        def predicate(line):
            row = line.split('\t')
            return len(row) > 3 and lang in row[3]
        return predicate

    gold = (path / 'gold')
    with gold.open('w') as f:
        f.write('\n'.join(filter(None, args.gold.read_text().split('\n'))))
    run(['./scripts/run_tac16_evaluation.sh', str(gold), str(system), str(out), '1'])
    for fname in out.iterdir():
        print(fname)
    print((out / '00report.tab').read_text())
    print((out / (args.prediction.name + '.evaluation')).read_text())
    input()
