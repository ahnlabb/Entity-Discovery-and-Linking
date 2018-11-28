from flask import Flask, render_template, jsonify, request
from utils import langforia
from pathlib import Path
from gold_std import get_doc_index, gold_std_idx

app = Flask(__name__)


@app.route('/el', methods=["POST"])
def entity_linking():
    payload = request.get_json()
    return jsonify(langforia(payload, 'en', format='json'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gold')
def doc_index():
    path = Path('../corpus/tac/en/eng.2015.eval.docria')
    if path.exists():
        gold_std, _ = gold_std_idx(path)
        index = get_doc_index(path, gold_std)
        return index
