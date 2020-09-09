import os
import re
import sys
import time
import numpy
import codecs
import dill as pickle
import configparser
from collections import defaultdict
from nltk.tree import ParentedTree
from bert_serving.client import BertClient
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm

# custom modules
import PCCParser
import DimLexParser
import utils


class ImplicitSenseClassifier():

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['parserdir']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['parserdir']
        os.environ['CLASSPATH'] = '%s/stanford-parser.jar' % self.config['lexparser']['parserdir']
        self.bertclient = None
        self.labelencodict = {}
        self.labeldecodict = {}
        self.maxencid = 42
        if os.path.exists(os.path.join(os.getcwd(), 'bert_client_encodings.pickle')):
            self.bertmap = pickle.load(codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'rb'))
        else:
            self.bertmap = {}


    def explicitRelationExists(self, relations, i, j):

        for rel in relations:
            intargsids = set([x.sentenceId for x in rel.arg2])
            extargsids = set([x.sentenceId for x in rel.arg1])
            if i in intargsids and j in extargsids:
                return True
            elif j in intargsids and i in extargsids:
                return True
        return False

    def getGoldImplicits(self, testfiles):

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
                if rel.relationType == 'implicit':
                    goldsenses.append(rel)
        return goldsenses


    def evaluate_gold(self, testfiles, f2gold):

        X_test = []
        y_test = []
        candidates = []

        for f in f2gold:
            if f in testfiles:
                relations, sents, tokens = f2gold[f]
                for rel in relations:
                    if rel.relationType == 'implicit':
                        intarg = ' '.join([x.token for x in rel.arg2])
                        extarg = ' '.join([x.token for x in rel.arg1])
                        leftarg = extarg # unmarked order
                        rightarg = intarg
                        if rel.arg2[-1].tokenId < rel.arg1[0].tokenId: # marked order
                            leftarg = intarg
                            rightarg = extarg
                        if tuple(tuple([leftarg, rightarg])) in self.bertmap:
                            enc = self.bertmap[tuple(tuple([leftarg, rightarg]))]
                        else:
                            enc = self.bertclient.encode([leftarg.split(), rightarg.split()], is_tokenized=True)
                            self.bertmap[tuple(tuple([leftarg, rightarg]))] = enc
                        bertfeats = numpy.concatenate(enc)
                        X_test.append(bertfeats)
                        y_test.append(rel.sense)

        pred = self.mlp.predict(X_test)
            
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
            grel_arg1 = sorted([int(x.tokenId) for x in grel.extArgTokens])
            grel_arg2 = sorted([int(x.tokenId) for x in grel.intArgTokens])
            found = False
            for prel in pred_relations:
                prel_arg1 = sorted([x.tokenId for x in prel.arg1])
                prel_arg2 = sorted([x.tokenId for x in prel.arg2])
                if prel_arg1 == grel_arg1 and prel_arg2 == grel_arg2:
                    found = True
                    if grel.sense == prel.sense:
                        dcor += 1
                    if grel.sense.split('.')[:2] == prel.sense.split('.')[:2]:
                        scor += 1
                    if grel.sense.split('.')[0] == prel.sense.split('.')[0]:
                        fcor += 1
            
        return tot, dcor, scor, fcor
        


    def train(self, trainfiles=[]): # second arg is to use only train filtes in cross-evaluation setup (empty by default)):

        start = time.time()
        sys.stderr.write('INFO: Starting training of implicit sense classifier...\n')
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

        X_train = []
        y_train = []

        # filtering out test files if a list of train fileids is specified
        if trainfiles:
            fd = {f:fd[f] for f in fd if f in trainfiles}

        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'implicit':
                    intarg = ' '.join([x.token for x in rel.intArgTokens])
                    extarg = ' '.join([x.token for x in rel.extArgTokens])
                    leftarg = extarg # unmarked order
                    rightarg = intarg
                    if rel.intArgTokens[-1].tokenId < rel.extArgTokens[0].tokenId: # marked order
                        leftarg = intarg
                        rightarg = extarg
                    if tuple(tuple([leftarg, rightarg])) in self.bertmap:
                        enc = self.bertmap[tuple(tuple([leftarg, rightarg]))]
                    else:
                        enc = self.bertclient.encode([leftarg.split(), rightarg.split()], is_tokenized=True)
                        self.bertmap[tuple(tuple([leftarg, rightarg]))] = enc
                    bertfeats = numpy.concatenate(enc)
                    X_train.append(bertfeats)
                    y_train.append(rel.sense)

        # overwriting memory maps (commented out because the ones uploaded to github contain all training input)
        #pickle.dump(self.bertmap, codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'wb'))
        
        self.mlp = MLPClassifier()
        self.le = LabelEncoder()
        self.le.fit(y_train)
        self.mlp.fit(X_train, y_train)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write('INFO: Done training implicit sense classifier...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes), int(seconds)))

        #pickle.dump(self.mlp, codecs.open('implicit_sense_classifier.pickle', 'wb'))
        #sys.stderr.write('INFO: Saved classifier to implicit_sense_classifier.pickle.\n')

    def load(self):

        if not os.path.exists(os.path.join(os.getcwd(), 'implicit_sense_classifier.pickle')):
            return 'ERROR: implicit_sense_classifier.pickle not found.\n'
        """
        try:
            self.bertclient = BertClient(timeout=10000) # milliseconds...
            self.bertclient.encode(["I'm gone, and I best believe I'm leaving.", "Pack up my belongings then it's off into the evening.", "Now I haven't exactly been embraced by the populace.", "Set sail upon the seven deadly seas of the anonymous."])
        except TimeoutError:
            sys.stderr.write('ERROR: Time-out! Please verify that bert-serving server is running (see docs).\n') # example call: bert-serving-start -model_dir /share/bert-base-german-cased_tf_version/ -num_worker=4 -max_seq_len=52
            return
        """
        self.mlp = pickle.load(codecs.open('implicit_sense_classifier.pickle', 'rb'))
    


    def predict(self, relations, sents):

        if not self.bertclient:
            try:
                self.bertclient = BertClient(timeout=10000) # milliseconds...
                self.bertclient.encode(["I'm gone, and I best believe I'm leaving.", "Pack up my belongings then it's off into the evening.", "Now I haven't exactly been embraced by the populace.", "Set sail upon the seven deadly seas of the anonymous."])
            except TimeoutError:
                sys.stderr.write('ERROR: Time-out! Please verify that bert-serving server is running (see docs).\n') # example call: bert-serving-start -model_dir /share/bert-base-german-cased_tf_version/ -num_worker=4 -max_seq_len=52
                return
        
        newrels = []
        for i in range(len(sents)-1):
            if self.explicitRelationExists(relations, i, i+1):
                pass
            
            else:
                i_tokens = [x.token for x in sents[i]]
                j_tokens = [x.token for x in sents[i+1]]
                i_tokens = utils.bertclient_safe(i_tokens)
                j_tokens = utils.bertclient_safe(j_tokens)
                if re.search('\w+', ' '.join(i_tokens)) and re.search('\w+', ' '.join(j_tokens)): # excluding cases where one of the two is only newlines
                    enc = self.bertclient.encode([i_tokens, j_tokens], is_tokenized=True)
                    bertfeats = numpy.concatenate(enc)
                    pred = self.mlp.predict(bertfeats.reshape(1, -1))
                    newrels.append([[t for t in sents[i]], [t for t in sents[i+1]], pred[0]])
                
        return newrels


                
