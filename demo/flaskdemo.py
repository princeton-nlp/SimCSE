import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string

from tqdm import tqdm
from time import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from simcse import SimCSE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simcse_demo(port, args):
    app = Flask(__name__, static_folder='./static')
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    CORS(app)

    sentence_path = os.path.join(args.sentences_dir, args.example_sentences)
    query_path = os.path.join(args.sentences_dir, args.example_query)
    embedder = SimCSE(args.model_name_or_path)
    embedder.build_index(sentence_path)
    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/api', methods=['GET'])
    def api():
        query = request.args['query']
        top_k = int(request.args['topk'])
        threshold = float(request.args['threshold'])
        start = time()
        results = embedder.search(query, top_k=top_k, threshold=threshold)
        ret = []
        out = {}
        for sentence, score in results:
            ret.append({"sentence": sentence, "score": score})
        span = time() - start
        out['ret'] = ret
        out['time'] = "{:.4f}".format(span)
        return jsonify(out)

    @app.route('/files/<path:path>')
    def static_files(path):
        return app.send_static_file('files/' + path)
        
    @app.route('/get_examples', methods=['GET'])
    def get_examples():
        with open(query_path, 'r') as fp:
            examples = [line.strip() for line in fp.readlines()]
        return jsonify(examples)
    
    addr = args.ip + ":" + args.port
    logger.info(f'Starting Index server at {addr}')
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    IOLoop.instance().start()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--sentences_dir', default=None, type=str)
    parser.add_argument('--example_query', default=None, type=str)
    parser.add_argument('--example_sentences', default=None, type=str)
    parser.add_argument('--port', default='8888', type=str)
    parser.add_argument('--ip', default='http://127.0.0.1')
    parser.add_argument('--load_light', default=False, action='store_true')
    args = parser.parse_args()

    run_simcse_demo(args.port, args)