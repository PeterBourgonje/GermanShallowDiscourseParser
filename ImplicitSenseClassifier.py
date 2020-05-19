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

    def train(self):

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



    def predict(self, relations, sents):

        newrels = []
        for i in range(len(sents)-1):
            if self.explicitRelationExists(relations, i, i+1):
                pass
            else:
                i_tokens = [x.token for x in sents[i]]
                j_tokens = [x.token for x in sents[i+1]]
                enc = self.bertclient.encode([i_tokens, j_tokens], is_tokenized=True)
                bertfeats = numpy.concatenate(enc)
                pred = self.mlp.predict(bertfeats.reshape(1, -1))
                newrels.append([[t for t in sents[i]], [t for t in sents[i+1]], pred[0]])
                
        return newrels


                
