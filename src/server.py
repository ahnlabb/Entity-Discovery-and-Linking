from pickle import load
from keras.models import load_model
from flask import Flask, render_template, jsonify, request
from utils import langforia
from pathlib import Path
from gold_std import get_doc_index, gold_std_idx
from process import predict
from docria.storage import DocumentIO
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
    path = Path('./corpus/tac/en/eng.2015.eval.docria')
    with DocumentIO.read(path) as docria:
        doc = list(docria)
        gold_std, _ = gold_std_idx(doc)
        index = get_doc_index(doc, gold_std)
        return jsonify(index)


@app.route('/predict', methods=['POST'])
def make_prediction():
    text = request.get_json()
    with graph.as_default():
        pred = predict(model, mappings, cats, text, padding=True)
    return jsonify(pred)


model = load_model('./model.h5')
with open('./cats.pickle', 'r+b') as f:
    cats = load(f)
with open('./mappings.pickle', 'r+b') as f:
    mappings = load(f)
graph = tf.get_default_graph()
