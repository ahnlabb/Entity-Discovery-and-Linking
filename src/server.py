from pathlib import Path
from pickle import load
from time import sleep

import sparql
import tensorflow as tf
from docria.storage import DocumentIO
from flask import Flask, jsonify, render_template, request
from gold_std import get_doc_index, gold_std_idx
from process import get_tgt, predict
from structs import ModelJar
from utils import emb_mat_init, langforia

app = Flask(__name__)


@app.route('/el', methods=["POST"])
def entity_linking():
    payload = request.get_json()
    return jsonify(langforia(payload, 'en', format='json'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    return render_template('debug.html')


@app.route('/gold')
def doc_index():
    get_models()
    path = Path('./corpus/tac/lang/en/eng.2017.eval.docria')
    with DocumentIO.read(path) as docria:
        doc = list(docria)
        gold_std, _ = gold_std_idx(doc)
        index = get_doc_index(doc, gold_std)
        return jsonify(index)


@app.route('/link', methods=['POST'])
def get_links():
    fname = 'corpus/wikimap_{}.pickle'.format(
        request.args.get('lang', default='en'))
    ents = request.get_json()
    if fname not in wiki_map:
        with open(fname, 'r+b') as f:
            wiki_map[fname] = load(f)
    return jsonify(
        [get_tgt(text, wiki_map[fname], wiki=True) for text in ents])


@app.route('/models')
def get_models():
    fname = 'corpus/wikimap_{}.pickle'.format(
        request.args.get('lang', default='en'))
    if fname not in wiki_map:
        with open(fname, 'r+b') as f:
            wiki_map[fname] = load(f)
    with graph.as_default():
        for name in models:
            if not models[name]:
                jar = ModelJar.load(name)
                models[name] = jar
    return jsonify(list(models.keys()))


def get_docria(fname):
    with DocumentIO.read('corpus/en/' + fname) as doc_reader:
        return list(doc_reader)


@app.route('/browse/<docriafile>')
def show_docria(docriafile):
    return jsonify([doc.props for doc in get_docria(docriafile)])


@app.route('/predict', methods=['POST'])
def make_prediction():
    model_name = request.args.get('model', default='model.en.pickle')
    text = request.get_json()
    with graph.as_default():
        pred, targets = predict(
            models[model_name],
            text,
            elmap=wiki_map['corpus/wikimap_en.pickle'])
    wikidata = sparql.get_results(filter(None, targets), [41, 18, 154])
    print(wikidata)
    wikidata = [{k: v['value'] for k, v in r.items()} for r in wikidata]
    wikidata = {int(r['entity'].split('/')[-1][1:]): r for r in wikidata}
    pred['targets'] = [wikidata.get(tgt) for tgt in targets if tgt]
    return jsonify(pred)


models = {"model.en.pickle": None}
wiki_map = {}
graph = tf.get_default_graph()
