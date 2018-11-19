from pathlib import Path

def pickled(path, fun):
    if path.suffix == '.pickle':
        with path.open('r+b') as f:
            return load(f)
    else:
        result = fun(path)
        with path.with_suffix('.pickle').open('w+b') as f:
            dump(result, f)
        return result
