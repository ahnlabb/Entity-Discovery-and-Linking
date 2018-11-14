from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route('/el', methods=["POST"])
def entity_linking():
    payload = request.get_json()
    return jsonify(payload)


@app.route('/')
def index():
    return render_template('index.html')
