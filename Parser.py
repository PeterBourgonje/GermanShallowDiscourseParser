#!/usr/bin/python3

import os
import sys
import time
import json
import numpy
import codecs
import argparse
import dill as pickle
from spacy.lang.de import German
from collections import defaultdict
from bert_serving.client import BertClient
from sklearn.metrics import f1_score, precision_score, recall_score

from flask import Flask
from flask import request
from flask_cors import CORS

# custom modules
import utils
import ConnectiveClassifier
import ExplicitArgumentExtractor
import ExplicitSenseClassifier
import ImplicitSenseClassifier
import PCCParser

nlp = German()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

app = Flask(__name__)
CORS(app)

cc = ConnectiveClassifier.ConnectiveClassifier()
eae = ExplicitArgumentExtractor.ExplicitArgumentExtractor()
esc = ExplicitSenseClassifier.ExplicitSenseClassifier()
isc = ImplicitSenseClassifier.ImplicitSenseClassifier()


class Token:

    def __init__(self, token, sentenceId, sentenceTokenId):
        self.token = token.text
        self.tokenId = token.i
        self.sentenceId = sentenceId
        self.sentenceTokenId = sentenceTokenId
        self.span = tuple((token.idx, token.idx+len(token.text)))

    def setConnective(self):
        self.isConnective = True

    def setMultiToken(self, y):
        if not hasattr(self, 'multiTokenIds'):
            self.multiTokenIds = []
        self.multiTokenIds.append(y)
    def setFullSentence(self, val):
        self.fullSentence = val


class Relation:

    def __init__(self, _id, _type, docId):
        self.relationId = _id
        self.relationType = _type
        self.docId = docId
        self.connective = []
        self.arg1 = []
        self.arg2 = []
        
    def addConnectiveToken(self, token):
        self.connective.append(token)
    def addIntArgToken(self, token):
        self.arg2.append(token)
    def addExtArgToken(self, token):
        self.arg1.append(token)
    def addSense(self, sense):
        self.sense = sense
        
def custom_tokenize(inp):

    # using spacy sentencizer/tokenizer since most(all?) nltk ones replace double quotes (and some other chars: https://www.nltk.org/_modules/nltk/tokenize/treebank.html)
    doc = nlp(inp)
    sents = {}
    tokens = {}
    for si, sent in enumerate(doc.sents):
        senttokens = []
        fullsent = ' '.join([t.text for t in sent])
        for ti, token in enumerate(sent):
            t = Token(token, si, ti)
            t.setFullSentence(fullsent)
            senttokens.append(t)
            tokens[t.tokenId] = t
        sents[si] = senttokens
        
    return sents, tokens

def relations2json(inp, relations):

    jso = []
    for rel in relations:
        js = {'ID': rel.relationId,
              'DocID': rel.docId,
              'Sense': rel.sense,
              'Type': rel.relationType}
        ### Arg1
        arg1 = {}
        arg1_charspanlist = []
        arg1_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.arg1])
        for chu in chunks: # can be discontinuous...
            spanlist = [[x.span[0] for x in rel.arg1 if x.tokenId == chu[0]][0], [x.span[1] for x in rel.arg1 if x.tokenId == chu[1]][0]]
            arg1_charspanlist.append(spanlist)
            arg1_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        arg1_rawtext = arg1_rawtext.strip()
        arg1_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId, x.sentenceTokenId] for x in rel.arg1]
        arg1['CharacterSpanList'] = arg1_charspanlist
        arg1['RawText'] = arg1_rawtext
        arg1['TokenList'] = arg1_tokenlist
        js['Arg1'] = arg1

        ### Arg2
        arg2 = {}
        arg2_charspanlist = []
        arg2_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.arg2])
        for chu in chunks: # can be discontinuous...
            spanlist = [[x.span[0] for x in rel.arg2 if x.tokenId == chu[0]][0], [x.span[1] for x in rel.arg2 if x.tokenId == chu[1]][0]]
            arg2_charspanlist.append(spanlist)
            arg2_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        arg2_rawtext = arg2_rawtext.strip()
        arg2_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId, x.sentenceTokenId] for x in rel.arg2]
        arg2['CharacterSpanList'] = arg2_charspanlist
        arg2['RawText'] = arg2_rawtext
        arg2['TokenList'] = arg2_tokenlist
        js['Arg2'] = arg2

        ### Conn
        conn = {}
        conn_charspanlist = []
        conn_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.connective])
        for chu in chunks:
            spanlist = [[x.span[0] for x in rel.connective if x.tokenId == chu[0]][0], [x.span[1] for x in rel.connective if x.tokenId == chu[1]][0]]
            conn_charspanlist.append(spanlist)
            conn_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        conn_rawtext = conn_rawtext.strip()
        conn_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId, x.sentenceTokenId] for x in rel.connective]
        conn['CharacterSpanList'] = conn_charspanlist
        conn['RawText'] = conn_rawtext
        conn['TokenList'] = conn_tokenlist
        js['Connective'] = conn

        jso.append(js)
        
    return json.dumps(jso, ensure_ascii=False)
    

def evaluate():

    pcc_folder = '/share/pcc2.2/'
    files = os.listdir(os.path.join(pcc_folder, 'connectives'))
    numIterations = 10
    splits = utils.getDataSplits(numIterations, len(files))

    connective_pscores = []
    connective_rscores = []
    connective_fscores = []
    for i in range(numIterations):
        testfiles = []
        trainfiles = []
        for index, _file in enumerate(files):
            if index >= splits[i] and index <= splits[i+1]:
                testfiles.append(_file)
            else:
                trainfiles.append(_file)
        cc.train(trainfiles)
        eae.train(trainfiles)
        pred, gold, f2tokens = cc.evaluate(testfiles)
        _cmf = f1_score(gold, pred, average='weighted')
        _cmp = precision_score(gold, pred, average='weighted')
        _cmr = recall_score(gold, pred, average='weighted')
        connective_fscores.append(_cmf)
        connective_pscores.append(_cmp)
        connective_rscores.append(_cmr)

        for f in f2tokens:
            gold_connective_relations = []
            pred_connective_relations = []
            already_processed_gold = []
            already_processed_pred = []
            _gid = 1
            _pid = 1
            ftokens = f2tokens[f]
            tokens = {}
            for _id in ftokens:
                t = ftokens[_id]
                t.tokenId = int(t.tokenId)
                t.setDocId(f)
                t.sentenceTokenId = t.sentencePosition
                tokens[t.tokenId] = t
                if t.isConnective and not (f, t.tokenId) in already_processed_gold:
                    goldrel = Relation(_gid, 'Explicit', f)
                    goldrel.addConnectiveToken(t)
                    if hasattr(t, 'multiTokenIds'):
                        for ot in t.multiTokenIds:
                            ott = ftokens[int(ot)]
                            ott.tokenId = int(ott.tokenId)
                            t.sentenceTokenId = t.sentencePosition
                            goldrel.addConnectiveToken(ott)
                            already_processed_gold.append((f, ott.tokenId))
                    gold_connective_relations.append(goldrel)
                    _gid += 1
                if hasattr(t, 'predictedConnective') and not (f, t.tokenId) in already_processed_pred:
                    predrel = Relation(_pid, 'Explicit', f)
                    predrel.addConnectiveToken(t)
                    if hasattr(t, 'multiTokenIds'):
                        for ot in t.multiTokenIds:
                            ott = ftokens[int(ot)]
                            ott.tokenId = int(ott.tokenId)
                            t.sentenceTokenId = t.sentencePosition
                            predrel.addConnectiveToken(ott)
                            already_processed_pred.append((f, ott.tokenId))
                    pred_connective_relations.append(predrel)
                    _pid += 1

            sents = PCCParser.wrapTokensInSentences(tokens)
            eae.predict(gold_connective_relations, sents, tokens)
            gold_arg_rels = eae.getGoldArgs(testfiles)
            sys.exit()
            # continue here! Should have args for gold conn-based prediction (inside gold_connective_relations) and actual args, easy to compare. Next step; comparing actual args to pred conn-based arg tokens...
            
            #pred_conn_args = eae.predict(pred_connective_relations, sents, tokens)

            
        



                    
        sys.exit()

    print('Classification report (10-fold cv):')
    print('\tConnective classification (weighted) average precision-score :', numpy.mean(connective_pscores))
    print('\tConnective classification (weighted) average recall-score    :', numpy.mean(connective_rscores))
    print('\tConnective classification (weighted) average f1-score        :', numpy.mean(connective_fscores))

    

def test():

    start = time.time()
    cc.train()
    eae.train()
    esc.train()
    isc.train()

    testfolder = '/share/pcc2.2/tokenized/'
    for testfile in os.listdir(testfolder):
        atf = os.path.join(testfolder, testfile)
        sys.stderr.write('INFO: Processing file: %s\n' % atf)
        inp = codecs.open(atf).read()

        sents, tokens = custom_tokenize(inp)
        cc.predict(sents)

        # populating list of relations, starting point are explicits/connectives
        relations = []
        _id = 1
        already_processed = [] # for phrasal connectives...
        for sid in sents:
            for i, token in enumerate(sents[sid]):
                if hasattr(token, 'isConnective') and not token.tokenId in already_processed:
                    rel = Relation(_id, 'Explicit', testfile)
                    rel.addConnectiveToken(token)
                    if hasattr(token, 'multiTokenIds'):
                        for ot in token.multiTokenIds:
                            rel.addConnectiveToken(tokens[ot])
                            already_processed.append(ot)
                    relations.append(rel)
                    _id += 1

        eae.predict(relations, sents, tokens)
        esc.predict(relations)

        newrels = isc.predict(relations, sents)
        maxrelid = max([x.relationId for x in relations]) if relations else 0
        for nr in newrels:
            r = Relation(maxrelid+1, 'Implicit', testfile)
            maxrelid += 1
            for t in nr[0]:
                r.addExtArgToken(t)
            for t in nr[1]:
                r.addIntArgToken(t)
            r.addSense(nr[2])
            relations.append(r)

        # to json output here
        jsonstr = relations2json(inp, relations)
        #print('relations:', jsonstr)
        assert json.loads(jsonstr)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    sys.stderr.write('INFO: Successfully parsed all files in test folder. Time taken: {:0>2}:{:0>2}:{:0>2}\n.'.format(int(hours), int(minutes), int(seconds)))


def main():
    
    inp = 'Wie schwierig es ist, in dieser Region einen Ausbildungsplatz zu finden, haben wir an dieser und anderer Stelle oft und ausführlich bewertet. Trotzdem bemühen sich Unternehmen sowie die Industrie- und Handelskammer Potsdam den Schulabgängern Wege in die Ausbildung aufzuzeigen. Und Beispielsweise gibt es ein mit entweder dies oder das, und dazu gibt es noch anstatt dass aapjes. Entweder bezahlen für die Schülung, oder später im Arsch gehen. Und das ist ein guter erster Schritt. Das weiß jedes Kind, aber nicht jeder hält sich daran. Das Schlimmste aber ist, dass noch heute versucht wird, zu mauscheln. Hier gibt es ein Satz. Hier gibt es noch ein Satz.' 

    """
    cc.train()
    eae.train()
    esc.train()
    isc.train()

    sents, tokens = custom_tokenize(inp)
    cc.predict(sents)

    # populating list of relations, starting point are explicits/connectives
    relations = []
    _id = 1
    already_processed = [] # for phrasal connectives...
    for sid in sents:
        for i, token in enumerate(sents[sid]):
            if hasattr(token, 'isConnective') and not token.tokenId in already_processed:
                rel = Relation(_id, 'Explicit', 'dummy')
                rel.addConnectiveToken(token)
                if hasattr(token, 'multiTokenIds'):
                    for ot in token.multiTokenIds:
                        rel.addConnectiveToken(tokens[ot])
                        already_processed.append(ot)
                relations.append(rel)
                _id += 1
    
    eae.predict(relations, sents, tokens)
    esc.predict(relations)

    newrels = isc.predict(relations, sents)
    maxrelid = max([x.relationId for x in relations])
    for nr in newrels:
        r = Relation(maxrelid+1, 'Implicit', 'dummy')
        maxrelid += 1
        for t in nr[0]:
            r.addExtArgToken(t)
        for t in nr[1]:
            r.addIntArgToken(t)
        r.addSense(nr[2])
        relations.append(r)
    
    # for dev/debugging, pickling result of the above
    pickle.dump(sents, codecs.open('sents_debug.pickle', 'wb'))
    pickle.dump(tokens, codecs.open('tokens_debug.pickle', 'wb'))
    pickle.dump(relations, codecs.open('relations_debug.pickle', 'wb'))
    """
    sents = pickle.load(codecs.open('sents_debug.pickle', 'rb'))
    tokens = pickle.load(codecs.open('tokens_debug.pickle', 'rb'))
    relations = pickle.load(codecs.open('relations_debug.pickle', 'rb'))

    jsonstr = relations2json(inp, relations)

    #for line in json.loads(jsonstr):
        #print(line)
    
    """
    for rel in relations:
        print('relid:', rel.relationId)
        print('type:', rel.relationType)
        print('conns:', [x.token for x in rel.connective])
        print('arg1:', [x.token for x in rel.arg1])
        print('arg2:', [x.token for x in rel.arg2])
        print('sense:', rel.sense)
        print()
    """

@app.route('/train', methods=['GET'])
def train():

    start = time.time()
    # check if there is a BertServer instance running:
    try:
        bertclient = BertClient(timeout=10000) # milliseconds...
        bertclient.encode(["I'm gone, and I best believe I'm leaving.", "Pack up my belongings then it's off into the evening.", "Now I haven't exactly been embraced by the populace.", "Set sail upon the seven deadly seas of the anonymous."])
    except TimeoutError:
        # example call: bert-serving-start -model_dir /share/bert-base-german-cased_tf_version/ -num_worker=4 -max_seq_len=52
        return 'ERROR: Time-out! Please verify that bert-serving server is running (see docs).\n'

    cc.train()
    eae.train()
    esc.train()
    isc.train()

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return 'INFO: Successfully trained models. Time taken: {:0>2}:{:0>2}:{:0>2}\n.'.format(int(hours), int(minutes), int(seconds))
    

@app.route('/parse', methods=['GET'])
def parse():

    if not hasattr(cc, 'clfs') or not hasattr(eae, 'sentposclf') or not hasattr(esc, 'clfs') or not hasattr(isc, 'mlp'):
        return 'ERROR: Could not find trained models. Are you sure you used the train endpoint?\n'

    if request.args.get('input') == None:
        return 'INFO: Please provide input text.\n'

    docid = None
    if request.args.get('docid') == None:
        docid = time.strftime('%m-%d-%Y_%H:%M:%S', time.gmtime())
    else:
        docid = request.args.get('docid')

        
    inp = request.args.get('input')
    sents, tokens = custom_tokenize(inp)
    cc.predict(sents)

    # populating list of relations, starting point are explicits/connectives
    relations = []
    _id = 1
    already_processed = [] # for phrasal connectives...
    for sid in sents:
        for i, token in enumerate(sents[sid]):
            if hasattr(token, 'isConnective') and not token.tokenId in already_processed:
                rel = Relation(_id, 'Explicit', docid)
                rel.addConnectiveToken(token)
                if hasattr(token, 'multiTokenIds'):
                    for ot in token.multiTokenIds:
                        rel.addConnectiveToken(tokens[ot])
                        already_processed.append(ot)
                relations.append(rel)
                _id += 1
    
    eae.predict(relations, sents, tokens)
    esc.predict(relations)

    newrels = isc.predict(relations, sents)
    maxrelid = max([x.relationId for x in relations]) if relations else 0
    for nr in newrels:
        r = Relation(maxrelid+1, 'Implicit', docid)
        maxrelid += 1
        for t in nr[0]:
            r.addExtArgToken(t)
        for t in nr[1]:
            r.addIntArgToken(t)
        r.addSense(nr[2])
        relations.append(r)

    # to json output here
    jsonstr = relations2json(inp, relations)

    return jsonstr


if __name__ == '__main__':

    #main() # for running without flask
    #test()
    evaluate()
    sys.exit()
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--port", help="port number to start flask app on", default=5000, type=int)
    args = argparser.parse_args()

    app.run(host='0.0.0.0', port=args.port,debug=True)

    
    # TODO:
    # then evaluation...
    # then done! (back to writing)

    
    #TODO: look into prod mode for flask (https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen)
