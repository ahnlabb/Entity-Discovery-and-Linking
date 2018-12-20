from flask import Flask, render_template, jsonify, request
from utils import langforia
from pathlib import Path
from gold_std import get_doc_index, gold_std_idx
from process import predict
from docria.storage import DocumentIO
from structs import ModelJar
from utils import emb_mat_init
import tensorflow as tf

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
    path = Path('./corpus/tac/lang/en/eng.2015.eval.docria')
    with DocumentIO.read(path) as docria:
        doc = list(docria)
        gold_std, _ = gold_std_idx(doc)
        index = get_doc_index(doc, gold_std)
        return jsonify(index)


@app.route('/models')
def get_models():
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
        pred = predict(models[model_name], text)
    return jsonify(pred)


models = {"model.en.pickle": None}
graph = tf.get_default_graph()
