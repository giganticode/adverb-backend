import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf_device='/gpu:0'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "colbert", "colbert"))

import traceback
import argparse
from flask import Flask, jsonify, request
from flask_cors import CORS

from controllers.code_summary_controller import CodeSummaryController
from controllers.code_symbol_controller import CodeSymbolController
from controllers.code_search_codebert_controller import CodeSearchCodeBertController
from controllers.code_search_colbert_controller import CodeSearchColBertController

DEBUG = True
PORT = 8090

app = Flask(__name__, static_folder="")
# enable CORS for api endpoint
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# controllers
code_summary = CodeSummaryController()
code_symbol = CodeSymbolController()
code_search_code_bert = CodeSearchCodeBertController()
code_search_col_bert = CodeSearchColBertController()

# API routes
@app.route("/api/summary", methods = ["POST"])
def get_summary():
    try:
        summary = code_summary.get_summary(request)
        if summary:
            response = jsonify(summary)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
    except:
        pass
    return "Bad request", "400"

@app.route("/api/name", methods = ["POST"])
def get_symbol_name():
    try:
        name = code_symbol.get_symbol_name(request)
        if name:
            response = jsonify(name)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
    except:
        pass
    return "Bad request", "400"

@app.route("/api/search", methods = ["POST"])
def search():
    try:
        search_result = None
        if request.data:
            model = request.data.get("model", "colBERT")
            if model == "colBERT":
                code_search_code_bert.search_for_text(request)
            else:
                search_result = code_search_col_bert.search_for_text(request)
        if search_result:
            response = jsonify(search_result)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        pass
    return "Bad request", "400"

@app.route("/api/search_index", methods = ["POST"])
def search_index():
    try:
        if code_search_col_bert.indexing(request) != None:
            return "Success", "200"
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        pass
    return "Bad request", "400"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=DEBUG, action="store_true", help="Start the server in debug mode.")
    parser.add_argument("--port", default=PORT, type=int, action="store", help="Set the port for of the web server.")
    parser.add_argument("--host", default="0.0.0.0", type=str, action="store", help="Set the host of the web server.")
    args = parser.parse_args()

    port = args.port
    debug = args.debug
    host = args.host

    app.run(use_reloader=debug, port=port, debug=debug, host=host)