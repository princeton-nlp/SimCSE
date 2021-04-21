# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import re
import os
import io
import json
import logging
import inspect
import torch
from torch import optim
from tqdm import tqdm

# Used to return all sentences embeddings from different datasets

# return protocols: 
# SICK-E: trainA, trainB, devA, devB, testA, testB
# STS*, SICK-R: embeddingA, embeddingB
# MR, CR, SUBJ, MPQA, SST, wiki: embeddings
# MRPC, TREC: training_embeddings, test_embeddings


def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class NLIEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** MNLI + SNLI*****\n\n')
        self.seed = seed
        train = self.loadFile("train")
        dev = self.loadFile("dev")
        test = self.loadFile("test")
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        if fpath == "test":
            mnli_fpath = "dev_mismatched"
        elif fpath == "dev":
            mnli_fpath = "dev_matched"
        else:
            mnli_fpath = fpath

        mnli_path = "./data/multinli_1.0/multinli_1.0_" + mnli_fpath + ".jsonl"
        snli_path = "./data/snli_1.0/snli_1.0_" + fpath + ".jsonl"
        sent1 = []
        sent2 = []
        with open(mnli_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = json.loads(line)
                if len(data["sentence1"]) >= 5 and len(data["sentence2"]) >= 5:
                    sent1.append(data["sentence1"].split())
                    sent2.append(data["sentence2"].split())

        with open(snli_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if len(data["sentence1"]) >= 5 and len(data["sentence2"]) >= 5:
                    sent1.append(data["sentence1"].split())
                    sent2.append(data["sentence2"].split())

        return {"X_A":sent1, "X_B":sent2}
    
    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + \
                  self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + \
                  self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B']),
                                   key=lambda z: (len(z[0]), len(z[1])))

            self.sick_data[key]['X_A'] = [x for (x, y) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in tqdm(range(0, len(self.sick_data[key]['X_A']), bsize)):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    if not isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.numpy()
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']

        return trainA, trainB, devA, devB, testA, testB

class SICKEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + \
                  self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + \
                  self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = self.encode_labels(self.sick_data['test']['y'])

        return trainA, trainB, devA, devB, testA, testB

    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j, i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j, i] = np.floor(y) - y + 1
        return Y


class SICKEntailmentEval(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Entailment*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[4])
        sick_data['y'] = [label2id[s] for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.sick_data['test']['y'])

        return trainA, trainB, devA, devB, testA, testB

class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_embeddingA = []
        all_embeddingB = []
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    all_embeddingA.append(enc1)
                    all_embeddingB.append(enc2)
        all_embeddingA = np.concatenate(all_embeddingA, 0)
        all_embeddingB = np.concatenate(all_embeddingB, 0)

        return all_embeddingA, all_embeddingB

class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class BinaryClassifierEval(object):
    def __init__(self, pos, neg, seed=1111):
        self.seed = seed
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        self.n_samples = len(self.samples)

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def run(self, params, batcher):
        enc_input = []
        # Sort to reduce padding
        sorted_corpus = sorted(zip(self.samples, self.labels),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples = [x for (x, y) in sorted_corpus]
        sorted_labels = [y for (x, y) in sorted_corpus]
        logging.info('Generating sentence embeddings')
        for ii in range(0, self.n_samples, params.batch_size):
            batch = sorted_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            enc_input.append(embeddings)
        enc_input = np.vstack(enc_input)
        logging.info('Generated sentence embeddings')

        return enc_input


class CREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)


class MREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)


class SUBJEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(self.__class__, self).__init__(obj, subj, seed)


class MPQAEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)

class SSTEval(object):
    def __init__(self, task_path, nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_path, 'sentiment-test'))
        self.sst_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                  self.sst_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

    def run(self, params, batcher):
        sst_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sst_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.sst_data[key]['X'],
                                     self.sst_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))

            sst_embed[key]['X'] = []
            for ii in range(0, len(self.sst_data[key]['y']), bsize):
                batch = self.sst_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                sst_embed[key]['X'].append(embeddings)
            sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        return  sst_embed['test']['X']

class MRPCEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'msr_paraphrase_train.txt'))
        test = self.loadFile(os.path.join(task_path,
                             'msr_paraphrase_test.txt'))
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.mrpc_data['train']['X_A'] + \
                  self.mrpc_data['train']['X_B'] + \
                  self.mrpc_data['test']['X_A'] + self.mrpc_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                mrpc_data['X_A'].append(text[3].split())
                mrpc_data['X_B'].append(text[4].split())
                mrpc_data['y'].append(text[0])

        mrpc_data['X_A'] = mrpc_data['X_A'][1:]
        mrpc_data['X_B'] = mrpc_data['X_B'][1:]
        mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]
        return mrpc_data

    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        for key in self.mrpc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
                                       self.mrpc_data[key]['X_B'],
                                       self.mrpc_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                mrpc_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    mrpc_embed[key][txt_type].append(embeddings)
                mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
            mrpc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = mrpc_embed['train']['A']
        trainB = mrpc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = mrpc_embed['train']['y']

        # Test
        testA = mrpc_embed['test']['A']
        testB = mrpc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = mrpc_embed['test']['y']

        return trainA, trainB, testA, testB

class TRECEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train_5500.label'))
        self.test = self.loadFile(os.path.join(task_path, 'TREC_10.label'))

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}
        tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
                   'HUM': 3, 'LOC': 4, 'NUM': 5}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1].split()
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]

        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y']),
                                    key=lambda z: (len(z[0]), z[1]))
        test_samples = [x for (x, y) in sorted_corpus_test]
        test_labels = [y for (x, y) in sorted_corpus_test]

        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')

        return train_embeddings, test_embeddings

class WikiEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Wiki *****\n\n')
        self.seed = seed
        self.data = self.loadFile()

    def loadFile(self):
        path = "./data/wiki_book/wiki_sent_1m_demo.txt"
        sent = []
        with open(path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip()
                if len(line) >= 5:
                    sent.append(line)

        return sent
    
    def do_prepare(self, params, prepare):
        samples = self.data
        return prepare(params, samples)

    def run(self, params, batcher):
        bsize = params.batch_size

        logging.info('Computing embedding for Wiki')
        # Sort to reduce padding
        self.data.sort(key=lambda z: len(z))

        emb = []
        for ii in tqdm(range(0, len(self.data), bsize)):
            batch = self.data[ii:ii + bsize]
            embeddings = batcher(params, batch, max_length=32)
            emb.append(embeddings)
        emb = np.vstack(emb)

        return emb
