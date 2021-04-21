# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from .postprocess_utils import SICKEntailmentEval, STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval, SICKRelatednessEval, CREval, MREval, SUBJEval, MPQAEval, SSTEval, MRPCEval, TRECEval, dotdict, NLIEval, WikiEval
import logging

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def compute_whitening_matrix(vecs):
    """
    borrowed from https://github.com/bojone/BERT-whitening/blob/main/demo.py
    """
    print(vecs.shape)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu

def get_vecs(name, params, batcher, prepare):
    if (isinstance(name, list)):
        vec_list = [get_vecs(n, params, batcher, prepare) for n in name]
        return np.concatenate(vec_list, 0)

    list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STSBenchmark', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'NLI', 'wiki']

    params = dotdict(params)
    params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
    params.seed = 1111 if 'seed' not in params else params.seed

    params.batch_size = 64 if 'batch_size' not in params else params.batch_size
    params.nhid = 0 if 'nhid' not in params else params.nhid
    params.kfold = 5 if 'kfold' not in params else params.kfold
    if 'classifier' not in params or not params['classifier']:
        params.classifier = {'nhid': 0}

    tpath = params.task_path
    assert name in list_tasks, str(name) + ' not in ' + str(list_tasks)

    # Original SentEval tasks
    if name == 'CR':
        evaluation = CREval(tpath + '/downstream/CR', seed= params.seed)
    elif name == 'MR':
        evaluation = MREval(tpath + '/downstream/MR', seed= params.seed)
    elif name == 'MPQA':
        evaluation = MPQAEval(tpath + '/downstream/MPQA', seed= params.seed)
    elif name == 'SUBJ':
        evaluation = SUBJEval(tpath + '/downstream/SUBJ', seed= params.seed)
    elif name == 'SST2':
        evaluation = SSTEval(tpath + '/downstream/SST/binary', nclasses=2, seed= params.seed)
    elif name == 'SST5':
        evaluation = SSTEval(tpath + '/downstream/SST/fine', nclasses=5, seed= params.seed)
    elif name == 'TREC':
        evaluation = TRECEval(tpath + '/downstream/TREC', seed= params.seed)
    elif name == 'MRPC':
        evaluation = MRPCEval(tpath + '/downstream/MRPC', seed= params.seed)
    elif name == 'SICKRelatedness':
        evaluation = SICKRelatednessEval(tpath + '/downstream/SICK', seed= params.seed)
    elif name == 'STSBenchmark':
        evaluation = STSBenchmarkEval(tpath + '/downstream/STS/STSBenchmark', seed= params.seed)
    elif name == 'SICKEntailment':
        evaluation = SICKEntailmentEval(tpath + '/downstream/SICK', seed= params.seed)
    elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        fpath = name + '-en-test'
        evaluation = eval(name + 'Eval')(tpath + '/downstream/STS/' + fpath, seed= params.seed)
    elif name == "NLI":
        evaluation = NLIEval("", seed=params.seed)
    elif name == "wiki":
        evaluation = WikiEval("", seed=params.seed)

    params.current_task = name
    evaluation.do_prepare(params, prepare)

    results = evaluation.run(params, batcher)

    # regression tasks
    if name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        print("do postprocessing for " + name)
        enc1, enc2 = results
        vecs = np.concatenate([enc1, enc2], axis=0)

    # classification task with no split
    if name in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5']:
        print("do postprocessing for " + name)
        vecs = results
    
    # classification task with splits
    if name in ['TREC']:
        print("do postprocessing for " + name)
        train, test = results
        vecs = np.concatenate([train, test], axis=0)
    
    if name in ['MRPC']:
        print("do postprocessing for " + name)
        trainA, trainB, testA, testB = results
        vecs = np.concatenate([trainA, trainB, testA, testB], axis=0)
    
    if name in ['SICKEntailment', "NLI"]:
        print("do postprocessing for " + name)
        trainA, trainB, devA, devB, testA, testB = results
        vecs = np.concatenate([trainA, trainB, devA, devB, testA, testB], axis=0)

    if name in ['wiki']:
        print("do postprocessing for " + name)
        vecs = results
    
    return vecs

# return a batcher function with post processing
def batcher_fn(name, params, batcher, prepare):
    
    vecs = get_vecs(name, params, batcher, prepare)

    kernel, bias = compute_whitening_matrix(vecs)
    
    def post_batcher(params, batch):
        embeddings = np.array(batcher(params, batch))
        return (embeddings + bias).dot(kernel)
    
    return post_batcher


