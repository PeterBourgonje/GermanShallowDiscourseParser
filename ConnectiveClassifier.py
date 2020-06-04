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
#from bert_serving.server import BertServer
#from bert_serving.server.helper import get_run_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# custom modules
import PCCParser
import DimLexParser
import utils


class ConnectiveClassifier():

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
        self.syndict = {
            "konnadv": "adv",
            "padv": "adv",
            "subj": "csu",
            "einzel": "other",
            "postp": "cco",
            "v2emb": "cco",
            "praep": "prep",
            "apci": "prep",
            "konj": "cco",
            "appo": "prep",
            "coordconj": "cco"}
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
        
    def getDimlexCandidates(self):

        dimlex = DimLexParser.parseXML(os.path.join(self.config['DiMLex']['dimlexdir'], 'DimLex.xml'))
        self.dimlextuples = {}
        for entry in dimlex:
            altdict = entry.alternativeSpellings
            syncat = self.syndict[entry.syncats[0].strip()] # taking coarse type of connective-lex.info here
            for item in altdict: # canonical form is always in list of alt spellings
                if altdict[item]['phrasal'] == 'cont':
                    tupl = tuple(word_tokenize(item))
                    self.dimlextuples[tupl] = {'type':'cont','surefire':entry.surefire, 'syncat':syncat}
                elif altdict[item]['single'] == 'cont':
                    tupl = tuple(word_tokenize(item))
                    self.dimlextuples[tupl] = {'type':'cont','surefire':entry.surefire, 'syncat':syncat}
                elif altdict[item]['single'] == 'discont':
                    tupl = tuple(word_tokenize(item))
                    self.dimlextuples[tupl] = {'type':'discont','surefire':entry.surefire, 'syncat':syncat}


    def getFeatures(self, sd, sid, dc):

        tokens = [x.token for x in sd[sid]]
        sentence = ' '.join(tokens)

        ### syntactic features ###
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
        if self.dimlextuples[dc]['type'] == 'cont':
            if utils.contains_sublist(tokens, list(dc)):
                match_positions = utils.get_match_positions(tokens, list(dc))
                if len(dc) == 1:
                    for position in match_positions:
                        feat = utils.getFeaturesFromTreeCont(ptree, position, dc[0])
                elif len(dc) > 1:
                    for startposition in match_positions:
                        positions = list(range(startposition, startposition+len(dc)))
                        feat = utils.getFeaturesFromTreeCont(ptree, list(positions), dc)
        elif self.dimlextuples[dc]['type'] == 'discont':
            if utils.contains_discont_sublist(tokens, list(dc)):
                match_positions = utils.get_discont_match_positions(tokens, list(dc))
                feat = utils.getFeaturesFromTreeDiscont(ptree, match_positions, dc)

        synfeats = [self.encode(v) for v in feat]
        
        ### bert representations ###
        prevsent = ['_']
        if sid > 0 and match_positions[0] > 0:
            prevsent = [t.token for t in sd[sid-1]]
        bertrep = None
        if tuple(tuple([' '.join(prevsent), sentence, dc])) in self.bertmap:
            bertrep = self.bertmap[tuple(tuple([' '.join(prevsent), sentence, dc]))]
        else:
            bertrep = self.bertclient.encode([utils.bertclient_safe(prevsent), utils.bertclient_safe(tokens), utils.bertclient_safe(list(dc))], is_tokenized=True)
            self.bertmap[tuple(tuple([' '.join(prevsent), sentence, dc]))] = bertrep
        bertrep = numpy.concatenate(bertrep)

        return bertrep, synfeats


    def evaluate(self, testfiles):

        connectivefiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'connectives')) if re.search('/maz-\d+.xml', x)] # filtering out temp/hidden files that may be there
        syntaxfiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'syntax')) if re.search('/maz-\d+.xml', x)]

        fd = defaultdict(lambda : defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

        self.getDimlexCandidates()

        X_test_bert = []
        X_test_syn = []
        y_test = []
        candidates = []
        predictionTokenIds = []

        # taking test files only
        fd = {f:fd[f] for f in fd if f in testfiles}

        f2tokens = {}
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            sents = PCCParser.wrapTokensInSentences(pccTokens)
            f2tokens[f] = pccTokens
            for sid in sents:
                sentlist = [t.token for t in sents[sid]]
                sentence = ' '.join(sentlist)
                for dc in sorted(self.dimlextuples):
                    isConnective = False
                    if self.dimlextuples[dc]['type'] == 'cont': # continuous connectives
                        if utils.contains_sublist(sentlist, list(dc)):
                            match_positions = utils.get_match_positions(sentlist, list(dc))
                            if len(dc) > 1: # establishing link between phrasal connectives (continuous)
                                for mp in match_positions:
                                    for k in range(mp+1, mp+len(dc)):
                                        sents[sid][mp].setMultiToken(sents[sid][k].tokenId)
                                        sents[sid][k].setMultiToken(sents[sid][mp].tokenId)
                            if not utils.iscontinuous(match_positions):
                                for submatch in match_positions:
                                    if sents[sid][submatch].isConnective:
                                        isConnective = True
                                    bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                                    if len(synfeats) and len(bertfeats):
                                        candidates.append(tuple([sents[sid][submatch]]))
                                        X_test_syn.append(synfeats)
                                        X_test_bert.append(bertfeats)
                                        y_test.append(isConnective)
                                        predictionTokenIds.append([(f, [sents[sid][submatch].tokenId][0])])

                            else:
                                if all([sents[sid][x].isConnective for x in match_positions]):
                                    isConnective = True
                                bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                                lc = []
                                # seems like match_positions returns single position for phrasal continuous matches, so:
                                if len(dc) > 1:
                                    for r in range(1, len(dc)):
                                        match_positions.append(match_positions[0]+r)
                                for mp in match_positions:
                                    lc.append(sents[sid][mp])
                                if len(synfeats) and len(bertfeats):
                                    candidates.append(tuple(lc))
                                    X_test_syn.append(synfeats)
                                    X_test_bert.append(bertfeats)
                                    y_test.append(isConnective)
                                    predictionTokenIds.append([(f, sents[sid][mp].tokenId) for mp in match_positions])

                    elif self.dimlextuples[dc]['type'] == 'discont': # discontinuous connectives
                        if utils.contains_discont_sublist(sentlist, list(dc)):
                            match_positions = utils.get_discont_match_positions(sentlist, list(dc))
                            for k in range(len(match_positions)-1): # establishing link between phrasal connectives (discontinuous)
                                sents[sid][match_positions[k]].setMultiToken(sents[sid][match_positions[k+1]].tokenId)
                                sents[sid][match_positions[k+1]].setMultiToken(sents[sid][match_positions[k]].tokenId)
                                # due to known bug in utils.get_discont_match_positions, only one discont conn per sent is detected. Should this behaviour in get_discont_match_positions change, this needs changing too.
                            if all([sents[sid][x].isConnective for x in match_positions]):
                                isConnective = True
                            bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                            lc = []
                            for mp in match_positions:
                                lc.append(sents[sid][mp])
                            if len(synfeats) and len(bertfeats):
                                candidates.append(tuple(lc))
                                X_test_syn.append(synfeats)
                                X_test_bert.append(bertfeats)
                                y_test.append(isConnective)
                                predictionTokenIds.append([(f, sents[sid][mp].tokenId) for mp in match_positions])

        X_test = [X_test_syn, X_test_bert]
        pred1 = numpy.asarray([clf.predict_proba(X) for clf, X in zip(self.clfs, X_test)])
        pred2 = numpy.average(pred1, axis=0)
        pred = numpy.argmax(pred2, axis=1)
        
        assert len(pred) == len(candidates)
        
        # overriding predictions with dimlex surefires:
        # one way to speed this up would be to do surefire connectives first, so that they don't even have to be considered during prediction
        for index, item in enumerate(zip(pred, candidates)):
            if self.dimlextuples[tuple(x.token for x in item[1])]['surefire']:
                pred[index] = 1

        for index, p in enumerate(pred):
            if p:
                for tupl in predictionTokenIds[index]:
                    f2tokens[tupl[0]][int(tupl[1])].setPredictedConnective(index)

        return pred, y_test, f2tokens

        
       
    def train(self, trainfiles=[]): # second arg is to use only train filtes in cross-evaluation setup (empty by default)

        start = time.time()
        sys.stderr.write('INFO: Starting training of connective classifier...\n')

        # check if there is a BertServer instance running:
        try:
            self.bertclient = BertClient(timeout=10000) # milliseconds...
            self.bertclient.encode(["I'm gone, and I best believe I'm leaving.", "Pack up my belongings then it's off into the evening.", "Now I haven't exactly been embraced by the populace.", "Set sail upon the seven deadly seas of the anonymous."])
        except TimeoutError:
            sys.stderr.write('ERROR: Time-out! Please verify that bert-serving server is running (see docs).\n') # example call: bert-serving-start -model_dir /share/bert-base-german-cased_tf_version/ -num_worker=4 -max_seq_len=52
            return
            # code to start bert-serving server inside python. Rather hacky though, and could not find how to stop server when done, so decided to make it a requirement to manually start this before running the parser (or starting it first in Dockerfile)
            """
            sys.argv.extend(['-model_dir', self.config['bert-serving']['modeldir'], '-num_worker', self.config['bert-serving']['numworkers'], '-max_seq_len', '52'])
            args = get_run_args()
            bs = BertServer(args)
            bs.start()
            """
        

        connectivefiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'connectives')) if re.search('/maz-\d+.xml', x)] # filtering out temp/hidden files that may be there
        syntaxfiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'syntax')) if re.search('/maz-\d+.xml', x)]

        fd = defaultdict(lambda : defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

        self.getDimlexCandidates()

        X_train_bert = []
        X_train_syn = []
        y_train = []

        # filtering out test files if a list of train fileids is specified
        if trainfiles:
            fd = {f:fd[f] for f in fd if f in trainfiles}
        
        #for f in tqdm(fd):
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            sents = PCCParser.wrapTokensInSentences(pccTokens)
            for sid in sents:
                sentlist = [t.token for t in sents[sid]]
                sentence = ' '.join(sentlist)
                for dc in sorted(self.dimlextuples):
                    isConnective = False
                    if self.dimlextuples[dc]['type'] == 'cont': # continuous connectives
                        if utils.contains_sublist(sentlist, list(dc)):
                            match_positions = utils.get_match_positions(sentlist, list(dc))
                            if not utils.iscontinuous(match_positions):
                                for submatch in match_positions:
                                    if sents[sid][submatch].isConnective:
                                        isConnective = True
                                    bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                                    X_train_bert.append(bertfeats)
                                    X_train_syn.append(synfeats)
                                    y_train.append(isConnective)
                                    
                            else:
                                if all([sents[sid][x].isConnective for x in match_positions]):
                                    isConnective = True
                                bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                                X_train_bert.append(bertfeats)
                                X_train_syn.append(synfeats)
                                y_train.append(isConnective)
                    elif self.dimlextuples[dc]['type'] == 'discont': # discontinuous connectives
                        if utils.contains_discont_sublist(sentlist, list(dc)):
                            match_positions = utils.get_discont_match_positions(sentlist, list(dc))
                            if all([sents[sid][x].isConnective for x in match_positions]):
                                isConnective = True
                            bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                            X_train_bert.append(bertfeats)
                            X_train_syn.append(synfeats)
                            y_train.append(isConnective)
                

        # overwriting memory maps (commented out because the ones uploaded to github contain all training input)
        #pickle.dump(self.bertmap, codecs.open(os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'wb'))
        #pickle.dump(self.parsermap, codecs.open(os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'wb'))
        
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)
        mlp = MLPClassifier()

        clfs = [rf, mlp]
        X_train = [X_train_syn, X_train_bert]
        self.clfs = [clf.fit(X, y_train) for clf, X in zip(clfs, X_train)]

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write('INFO: Done training connective classifier...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes), int(seconds)))

        

    def predict(self, sents):

        # check if training has happened already
        if not hasattr(self, 'clfs') or not hasattr(self, 'dimlextuples') or not hasattr(self, 'bertclient'):
            sys.stderr.write('ERROR: Required attributes not set. Please verify the connective classifier was successfully trained.\n')
            return
        
        candidates = []
        X_test_syn = []
        X_test_bert = []
        for sid in sents:
            sentlist = [t.token for t in sents[sid]]
            sentence = ' '.join(sentlist)
            for dc in sorted(self.dimlextuples):
                if self.dimlextuples[dc]['type'] == 'cont': # continuous connectives
                    if utils.contains_sublist(sentlist, list(dc)):
                        match_positions = utils.get_match_positions(sentlist, list(dc))
                        if len(dc) > 1: # establishing link between phrasal connectives (continuous)
                            for mp in match_positions:
                                for k in range(mp+1, mp+len(dc)):
                                    sents[sid][mp].setMultiToken(sents[sid][k].tokenId)
                                    sents[sid][k].setMultiToken(sents[sid][mp].tokenId)
                        if not utils.iscontinuous(match_positions):
                            for submatch in match_positions:
                                bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                                if len(synfeats) and len(bertfeats):
                                    candidates.append(tuple([sents[sid][submatch]]))
                                    X_test_syn.append(synfeats)
                                    X_test_bert.append(bertfeats)

                        else:
                            bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                            lc = []
                            # seems like match_positions returns single position for phrasal continuous matches, so:
                            if len(dc) > 1:
                                for r in range(1, len(dc)):
                                    match_positions.append(match_positions[0]+r)
                            for mp in match_positions:
                                lc.append(sents[sid][mp])
                            if len(synfeats) and len(bertfeats):
                                candidates.append(tuple(lc))
                                X_test_syn.append(synfeats)
                                X_test_bert.append(bertfeats)

                elif self.dimlextuples[dc]['type'] == 'discont': # discontinuous connectives
                    if utils.contains_discont_sublist(sentlist, list(dc)):
                        match_positions = utils.get_discont_match_positions(sentlist, list(dc))
                        for k in range(len(match_positions)-1): # establishing link between phrasal connectives (discontinuous)
                            sents[sid][match_positions[k]].setMultiToken(sents[sid][match_positions[k+1]].tokenId)
                            sents[sid][match_positions[k+1]].setMultiToken(sents[sid][match_positions[k]].tokenId)
                            # due to known bug in utils.get_discont_match_positions, only one discont conn per sent is detected. Should this behaviour in get_discont_match_positions change, this needs changing too.
                        bertfeats, synfeats = self.getFeatures(sents, sid, dc)
                        lc = []
                        for mp in match_positions:
                            lc.append(sents[sid][mp])
                        if len(synfeats) and len(bertfeats):
                            candidates.append(tuple(lc))
                            X_test_syn.append(synfeats)
                            X_test_bert.append(bertfeats)
                        
        
        if candidates: # do not predict if input didn't contain a candidate
            X_test = [X_test_syn, X_test_bert]
            pred1 = numpy.asarray([clf.predict_proba(X) for clf, X in zip(self.clfs, X_test)])
            pred2 = numpy.average(pred1, axis=0)
            pred = numpy.argmax(pred2, axis=1)

            assert len(pred) == len(candidates)

            # overriding predictions with dimlex surefires:
            # one way to speed this up would be to do surefire connectives first, so that they don't even have to be considered during prediction
            for index, item in enumerate(zip(pred, candidates)):
                if tuple(x.token for x in item[1]) in self.dimlextuples:
                    if self.dimlextuples[tuple(x.token for x in item[1])]['surefire']:
                        pred[index] = 1

            # filtering out submatches (for multiword connectives that also work stand-alone (anstatt dass/anstatt), we get multiple predictions; taking the longest version)
            delpositions = []
            for k in range(len(pred)):
                for l in range(k+1, len(pred)):
                    k_indices = [x.tokenId for x in candidates[k]]
                    l_indices = [x.tokenId for x in candidates[l]]
                    if set(k_indices).intersection(set(l_indices)):
                        if len(k_indices) > len(l_indices):
                            delpositions.append(l)
                        else:
                            delpositions.append(k)
            candidates = [x for i, x in enumerate(candidates) if not i in delpositions]
            pred = [x for i, x in enumerate(pred) if not i in delpositions]

            for index, p in enumerate(pred):
                if p == 1:
                    for x in candidates[index]:
                        x.setConnective()
        
