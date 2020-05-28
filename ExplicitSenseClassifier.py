import os
import re
import sys
import time
import numpy
import codecs
import dill as pickle
import configparser
from nltk.parse import stanford
from collections import defaultdict
from nltk import word_tokenize
from nltk.tree import ParentedTree
from bert_serving.client import BertClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm

# custom modules
import PCCParser
import DimLexParser
import utils



class ExplicitSenseClassifier():

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['parserdir']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['parserdir']
        os.environ['CLASSPATH'] = '%s/stanford-parser.jar' % self.config['lexparser']['parserdir']
        self.lexparser = stanford.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz')
        self.bertclient = None
        self.labelencodict = {}
        self.labeldecodict = {}
        self.maxencid = 42
        if os.path.exists(os.path.join(os.getcwd(), 'bert_client_encodings.pickle')):
            self.bertmap = pickle.load(codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'rb'))
        else:
            self.bertmap = {}
        if os.path.exists(os.path.join(os.getcwd(), 'pcc_memorymap.pickle')):
            self.parsermap = pickle.load(codecs.open(os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'rb'))
        else:
            self.parsermap = {}

    def encode(self, val):
        if val in self.labelencodict:
            return self.labelencodict[val]
        else:
            self.labelencodict[val] = self.maxencid
            self.labeldecodict[self.maxencid] = val
            self.maxencid += 1
            return self.labelencodict[val]
            
    def decode(self, val):
        return self.labeldecodict[val]


    def getFeatures(self, rel):

        ### syntactic features ###
        sentence = rel.connectiveTokens[0].fullSentence
        tokens = sentence.split()
        ptree = None
        if sentence in self.parsermap:
            ptree = self.parsermap[sentence]
        else:
            tree = self.lexparser.parse(re.sub('\)', ']', re.sub('\(', '[', sentence)).split())
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                self.parsermap[sentence] = ptree
                break # always taking the first, assuming that this is the best scoring tree.

        feat = ['_']*12
        match_positions = None
        if utils.iscontinuous([int(x.tokenId) for x in rel.connectiveTokens]): # continuous connective
            if utils.contains_sublist(tokens, [x.token for x in rel.connectiveTokens]):
                match_positions = utils.get_match_positions(tokens, [x.token for x in rel.connectiveTokens])
                if len(rel.connectiveTokens) == 1:
                    for position in match_positions:
                        feat = utils.getFeaturesFromTreeCont(ptree, position, rel.connectiveTokens[0].token)
                elif len(rel.connectiveTokens) > 1:
                    for startposition in match_positions:
                        positions = list(range(startposition, startposition+len(rel.connectiveTokens)))
                        feat = utils.getFeaturesFromTreeCont(ptree, list(positions), tuple([x.token for x in rel.connectiveTokens]))
        else: # discontinuous connective
            if utils.contains_discont_sublist(tokens, [x.token for x in rel.connectiveTokens]):
                match_positions = utils.get_discont_match_positions(tokens, [x.token for x in rel.connectiveTokens])
                feat = utils.getFeaturesFromTreeDiscont(ptree, match_positions, tuple([x.token for x in rel.connectiveTokens]))

        synfeats = [self.encode(v) for v in feat]

        ### bert features ###
        conn = ' '.join([x.token for x in rel.connectiveTokens])
        intarg = ' '.join([x.token for x in rel.intArgTokens])
        extarg = ' '.join([x.token for x in rel.extArgTokens])
        leftarg = extarg # unmarked order
        rightarg = intarg
        if rel.intArgTokens[-1].tokenId < rel.extArgTokens[0].tokenId: # marked order
            leftarg = intarg
            rightarg = extarg
        sentence = rel.connectiveTokens[0].fullSentence
        if tuple(tuple([leftarg, rightarg, conn])) in self.bertmap:
            enc = self.bertmap[tuple(tuple([leftarg, rightarg, conn]))]
        else:
            enc = self.bertclient.encode([leftarg.split(), rightarg.split(), conn.split()], is_tokenized=True)
            self.bertmap[tuple(tuple([leftarg, rightarg, conn]))] = enc

        return synfeats, numpy.concatenate(enc)

        
    def train(self, trainfiles=[]): # second arg is to use only train filtes in cross-evaluation setup (empty by default)

        start = time.time()
        sys.stderr.write('INFO: Starting training of explicit sense classifier...\n')
        # check if there is a BertServer instance running:
        try:
            self.bertclient = BertClient(timeout=10000) # milliseconds...
            self.bertclient.encode(["I'm gone, and I best believe I'm leaving.", "Pack up my belongings then it's off into the evening.", "Now I haven't exactly been embraced by the populace.", "Set sail upon the seven deadly seas of the anonymous."])
        except TimeoutError:
            sys.stderr.write('ERROR: Time-out! Please verify that bert-serving server is running (see docs).\n') # example call: bert-serving-start -model_dir /share/bert-base-german-cased_tf_version/ -num_worker=4 -max_seq_len=52
            return

        connectivefiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'connectives')) if re.search('/maz-\d+.xml', x)] # filtering out temp/hidden files that may be there
        syntaxfiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'syntax')) if re.search('/maz-\d+.xml', x)]

        fd = defaultdict(lambda : defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')
        self.conn2mostfrequent = defaultdict(lambda : defaultdict(int)) # getting this over the training data, to override with the most frequent sense in post-processing if predicted sense does not match with dimlex

        X_train_bert = []
        X_train_syn = []
        y_train = []

        # filtering out test files if a list of train fileids is specified
        if trainfiles:
            fd = {f:fd[f] for f in fd if f in trainfiles}

        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    bertfeats, synfeats = self.getFeatures(rel)
                    X_train_bert.append(bertfeats)
                    X_train_syn.append(synfeats)
                    y_train.append(rel.sense)
                    self.conn2mostfrequent[tuple([x.token for x in rel.connectiveTokens])][rel.sense] += 1

        # overwriting memory maps (commented out because the ones uploaded to github contain all training input)
        #pickle.dump(self.bertmap, codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'wb'))
        #pickle.dump(self.parsermap, codecs.open(os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'wb'))
        
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)
        mlp = MLPClassifier()

        clfs = [rf, mlp]
        X_train = [X_train_syn, X_train_bert]
        self.le = LabelEncoder()
        self.le.fit(y_train)
        self.clfs = [clf.fit(X, y_train) for clf, X in zip(clfs, X_train)]

        # get conn2senses from dimlex for manual overriding as post-processing step
        self.conn2senses = {}
        dimlex = DimLexParser.parseXML(os.path.join(self.config['DiMLex']['dimlexdir'], 'DimLex.xml'))
        for entry in dimlex:
            altdict = entry.alternativeSpellings
            senses = entry.sense2Probs.keys()
            for item in altdict: # canonical form is always in list of alt spellings
                tupl = tuple(word_tokenize(item))
                self.conn2senses[tupl] = senses


        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write('INFO: Done training explicit sense classifier...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes), int(seconds)))

    def getGoldSenses(self, testfiles):

        connectivefiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'connectives')) if re.search('/maz-\d+.xml', x)] # filtering out temp/hidden files that may be there
        syntaxfiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'syntax')) if re.search('/maz-\d+.xml', x)]

        fd = defaultdict(lambda : defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

        # taking test files only
        fd = {f:fd[f] for f in fd if f in testfiles}
        goldsenses = []
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    goldsenses.append(rel)
        return goldsenses


    def evaluate_gold(self, testfiles, f2gold):

        X_test_syn = []
        X_test_bert = []
        y_test = []
        candidates = []

        for f in f2gold:
            if f in testfiles:
                relations, sents, tokens = f2gold[f]
                for rel in relations:
                    if rel.relationType == 'explicit':
                        sentence = rel.connective[0].fullSentence
                        tokens = sentence.split()
                        ptree = None
                        if sentence in self.parsermap:
                            ptree = self.parsermap[sentence]
                        else:
                            tree = self.lexparser.parse(re.sub('\)', ']', re.sub('\(', '[', sentence)).split())
                            ptreeiter = ParentedTree.convert(tree)
                            for t in ptreeiter:
                                ptree = t
                                self.parsermap[sentence] = ptree
                                break # always taking the first, assuming that this is the best scoring tree.
                        feat = ['_']*12
                        match_positions = None
                        if utils.iscontinuous([int(x.tokenId) for x in rel.connective]): # continuous connective
                            if utils.contains_sublist(tokens, [x.token for x in rel.connective]):
                                match_positions = utils.get_match_positions(tokens, [x.token for x in rel.connective])
                                if len(rel.connective) == 1:
                                    for position in match_positions:
                                        feat = utils.getFeaturesFromTreeCont(ptree, position, rel.connective[0].token)
                                elif len(rel.connective) > 1:
                                    for startposition in match_positions:
                                        positions = list(range(startposition, startposition+len(rel.connective)))
                                        feat = utils.getFeaturesFromTreeCont(ptree, list(positions), tuple([x.token for x in rel.connective]))
                        else: # discontinuous connective
                            if utils.contains_discont_sublist(tokens, [x.token for x in rel.connective]):
                                match_positions = utils.get_discont_match_positions(tokens, [x.token for x in rel.connective])
                                feat = utils.getFeaturesFromTreeDiscont(ptree, match_positions, tuple([x.token for x in rel.connective]))
                        synfeats = [self.encode(v) for v in feat]

                        conn = ' '.join([x.token for x in rel.connective])
                        intarg = ' '.join([x.token for x in rel.arg2])
                        extarg = ' '.join([x.token for x in rel.arg1])
                        leftarg = extarg # unmarked order
                        rightarg = intarg
                        if rel.arg2[-1].tokenId < rel.arg1[0].tokenId: # marked order
                            leftarg = intarg
                            rightarg = extarg
                        enc = None
                        if tuple(tuple([leftarg, rightarg, conn])) in self.bertmap:
                            enc = self.bertmap[tuple(tuple([leftarg, rightarg, conn]))]
                        else:
                            enc = self.bertclient.encode([utils.bertclient_safe(leftarg.split()), utils.bertclient_safe(rightarg.split()), utils.bertclient_safe(conn.split())], is_tokenized=True)
                        bertfeats = numpy.concatenate(enc)

                        X_test_syn.append(synfeats)
                        X_test_bert.append(bertfeats)
                        y_test.append(rel.sense)
                        candidates.append(tuple([x.token for x in rel.connective]))

        if candidates:
            X_test = [X_test_bert, X_test_syn]
            pred1 = numpy.asarray([clf.predict_proba(X) for clf, X in zip(self.clfs, X_test)])
            pred2 = numpy.average(pred1, axis=0)
            pred = numpy.argmax(pred2, axis=1)

            assert len(pred) == len(candidates)

            pred = self.le.inverse_transform(pred)

            # checking predicted sense with dimlex and overriding if not matching:
            # one way to speed up this code is to take unambiguous sense conns from dimlex right away, without the prediction part
            for i, t in enumerate(zip(pred, candidates)):
                p, s = t
                if s in self.conn2senses:
                    if not p in self.conn2senses[s]:
                        if len(self.conn2senses[s]) == 1:
                            pred[i] = list(self.conn2senses[s])[0]
                        else:
                            if s in self.conn2mostfrequent:
                                top = sorted(self.conn2mostfrequent[s].items(), key = lambda x: x[1], reverse=True)[0][0]
                                pred[i] = top
            
            detailed_f1 = f1_score(pred, y_test, average='weighted')
            second_level_f1 = f1_score(['.'.join(x.split('.')[:2]) for x in pred], ['.'.join(x.split('.')[:2]) for x in y_test], average='weighted')
            first_level_f1 = f1_score([x.split('.')[0] for x in pred], [x.split('.')[0] for x in y_test], average='weighted')
            
        return detailed_f1, second_level_f1, first_level_f1


    
    def evaluate_pred(self, pred_relations, gold_relations):

        tot = 0
        dcor = 0
        scor = 0
        fcor = 0
        for grel in gold_relations:
            tot += 1
            grel_conn = sorted([int(x.tokenId) for x in grel.connectiveTokens])
            found = False
            for prel in pred_relations:
                prel_conn = sorted([x.tokenId for x in prel.connective])
                if prel_conn == grel_conn:
                    found = True
                    if grel.sense == prel.sense:
                        dcor += 1
                    if grel.sense.split('.')[:2] == prel.sense.split('.')[:2]:
                        scor += 1
                    if grel.sense.split('.')[0] == prel.sense.split('.')[0]:
                        fcor += 1
            
        return tot, dcor, scor, fcor

    
    def predict(self, relations):

        X_test_syn = []
        X_test_bert = []
        candidates = []
        for rel in relations:
            sentence = rel.connective[0].fullSentence
            tokens = sentence.split()
            ptree = None
            tree = self.lexparser.parse(re.sub('\)', ']', re.sub('\(', '[', sentence)).split())
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                self.parsermap[sentence] = ptree
                break # always taking the first, assuming that this is the best scoring tree.
            feat = ['_']*12
            match_positions = None
            if utils.iscontinuous([int(x.tokenId) for x in rel.connective]): # continuous connective
                if utils.contains_sublist(tokens, [x.token for x in rel.connective]):
                    match_positions = utils.get_match_positions(tokens, [x.token for x in rel.connective])
                    if len(rel.connective) == 1:
                        for position in match_positions:
                            feat = utils.getFeaturesFromTreeCont(ptree, position, rel.connective[0].token)
                    elif len(rel.connective) > 1:
                        for startposition in match_positions:
                            positions = list(range(startposition, startposition+len(rel.connective)))
                            feat = utils.getFeaturesFromTreeCont(ptree, list(positions), tuple([x.token for x in rel.connective]))
            else: # discontinuous connective
                if utils.contains_discont_sublist(tokens, [x.token for x in rel.connective]):
                    match_positions = utils.get_discont_match_positions(tokens, [x.token for x in rel.connective])
                    feat = utils.getFeaturesFromTreeDiscont(ptree, match_positions, tuple([x.token for x in rel.connective]))
            synfeats = [self.encode(v) for v in feat]
            
            conn = ' '.join([x.token for x in rel.connective])
            intarg = ' '.join([x.token for x in rel.arg2])
            extarg = ' '.join([x.token for x in rel.arg1])
            leftarg = extarg # unmarked order
            rightarg = intarg
            try:
                if rel.arg2[-1].tokenId < rel.arg1[0].tokenId: # marked order
                    leftarg = intarg
                    rightarg = extarg
            except IndexError:
                pass # one of the two (or both) not found/empty
            enc = self.bertclient.encode([utils.bertclient_safe(leftarg.split()), utils.bertclient_safe(rightarg.split()), utils.bertclient_safe(conn.split())], is_tokenized=True)
            bertfeats = numpy.concatenate(enc)

            X_test_syn.append(synfeats)
            X_test_bert.append(bertfeats)
            candidates.append(tuple([x.token for x in rel.connective]))

        if candidates:
            X_test = [X_test_bert, X_test_syn]
            pred1 = numpy.asarray([clf.predict_proba(X) for clf, X in zip(self.clfs, X_test)])
            pred2 = numpy.average(pred1, axis=0)
            pred = numpy.argmax(pred2, axis=1)

            assert len(pred) == len(candidates)

            pred = self.le.inverse_transform(pred)

            # checking predicted sense with dimlex and overriding if not matching:
            # one way to speed up this code is to take unambiguous sense conns from dimlex right away, without the prediction part
            for i, t in enumerate(zip(pred, candidates)):
                p, s = t
                if s in self.conn2senses:
                    if not p in self.conn2senses[s]:
                        if len(self.conn2senses[s]) == 1:
                            pred[i] = list(self.conn2senses[s])[0]
                        else:
                            if s in self.conn2mostfrequent:
                                top = sorted(self.conn2mostfrequent[s].items(), key = lambda x: x[1], reverse=True)[0][0]
                                pred[i] = top

            for pair in zip(relations, pred):
                rel, prediction = pair
                rel.addSense(prediction)
            
