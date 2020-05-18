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
from tqdm import tqdm

# custom modules
import PCCParser
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

        
    def train(self):

        start = time.time()
        sys.stderr.write('INFO: Starting training of connective classifier...\n')
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

        X_train_bert = []
        X_train_syn = []
        y_train = []

        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    bertfeats, synfeats = self.getFeatures(rel)
                    X_train_bert.append(bertfeats)
                    X_train_syn.append(synfeats)
                    y_train.append(rel.sense)

        # overwriting memory maps
        pickle.dump(self.bertmap, codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'wb'))
        pickle.dump(self.parsermap, codecs.open(os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'wb'))
        
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)
        mlp = MLPClassifier()

        clfs = [rf, mlp]
        X_train = [X_train_syn, X_train_bert]
        self.clfs = [clf.fit(X, y_train) for clf, X in zip(clfs, X_train)]

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write('INFO: Done training explicit sense classifier...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes), int(seconds)))

