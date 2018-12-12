from keras.models import Sequential, load_model
from pathlib import Path
from pickle import load, dump
from tempfile import mkstemp
import os


class ModelJar:
    def __init__(self, model, mappings, cats, path: Path = None):
        self.model = model
        self.mappings = mappings
        self.cats = cats
        self.path = path
        
    def save(self, filename: Path = None):
        if filename is None:
            if self.path is None:
                raise ValueError('Model needs to have a file name')
            filename = self.path
            
        if not self.mappings:
            print('Saving model without mappings...')
        if not self.cats:
            print('Saving model without classes...')
            
        fp, fname = mkstemp()
        self.model.save(fname)
        with open(fname, 'r+b') as f:
            self.model = f.read()
            
        os.close(fp)
        with filename.open('w+b') as f:
            dump(self, f)
            
    @staticmethod
    def load(filename: Path = None):
        if filename is None:
            if self.path is None:
                raise ValueError('Model needs to have a file name')
            filename = self.path
            
        with Path(filename).open('r+b') as f:
            jar = load(f)
            
        fp, fname = mkstemp()
        with open(fname, 'w+b') as f:
            f.write(jar.model)
        jar.model = load_model(fname)
        
        os.close(fp)
        return jar
        