import argparse
from flask import Flask, jsonify, request
from flask_cors import CORS

from controllers.code_summary_controller import CodeSummaryController
from controllers.code_symbol_controller import CodeSymbolController

DEBUG = True
PORT = 8080

app = Flask(__name__, static_folder="")
# enable CORS for api endpoint
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# controllers
code_summary = CodeSummaryController()
code_symbol = CodeSymbolController()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=DEBUG, action="store_true", help="Start the server in debug mode.")
    parser.add_argument("--port", default=PORT, type=int, action="store", help="Set the port for of the web server.")
    parser.add_argument("--host", default="127.0.0.1", type=str, action="store", help="Set the host of the web server.")
    args = parser.parse_args()

    port = args.port
    debug = args.debug
    host = args.host

    app.run(use_reloader=debug, port=port, debug=debug, host=host)