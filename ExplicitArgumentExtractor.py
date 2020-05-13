import os
import re
import sys
import time
import numpy
import codecs
import configparser
import dill as pickle
from nltk.tree import ParentedTree
from nltk.parse import stanford
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

# custom modules
import PCCParser
import utils


class ExplicitArgumentExtractor:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['parserdir']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['parserdir']
        os.environ['CLASSPATH'] = '%s/stanford-parser.jar' % self.config['lexparser']['parserdir']
        self.lexparser = stanford.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz')
        if os.path.exists(os.path.join(os.getcwd(), 'pcc_memorymap.pickle')):
            self.parsermap = pickle.load(codecs.open(os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'rb'))
        else:
            self.parsermap = {}

        self.labelencodict = {}
        self.labeldecodict = {}
        self.maxencid = 42

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

        

    def train(self):

        start = time.time()
        sys.stderr.write('INFO: Starting training of explicit argument extractor...\n') # actually, the only thing being trained is/are the position classifiers, since the rest is rule-based

        connectivefiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'connectives')) if re.search('/maz-\d+.xml', x)] # filtering out temp/hidden files that may be there
        syntaxfiles = [x for x  in utils.listfolder(os.path.join(self.config['PCC']['pccdir'], 'syntax')) if re.search('/maz-\d+.xml', x)]

        fd = defaultdict(lambda : defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

        X_train_pos = []
        y_train_pos = []
        X_train_samesent = []
        y_train_samesent = []

        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                connective = ' '.join([x.token for x in rel.connectiveTokens])
                refcon = rel.connectiveTokens[0]
                sentence = refcon.fullSentence
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
                ptree = ParentedTree.convert(ptree)
                postag = ptree.pos()[refcon.sentencePosition][1]
                ln = 'SOS' if refcon.sentencePosition == 0 else ptree.pos()[refcon.sentencePosition-1][0]
                rn = 'EOS' if refcon.sentencePosition == len(ptree.pos()) else ptree.pos()[refcon.sentencePosition+1][0]
                lnpos = 'SOS' if refcon.sentencePosition == 0 else ptree.pos()[refcon.sentencePosition-1][1]
                rnpos = 'EOS' if refcon.sentencePosition == len(ptree.pos()) else ptree.pos()[refcon.sentencePosition+1][1]
                nodePosition = ptree.leaf_treeposition(refcon.sentencePosition)
                parent = ptree[nodePosition[:-1]].parent()
                rootroute = utils.getPathToRoot(parent, [])
                feat = [connective, postag, ln, rn, lnpos, rnpos, '-'.join(rootroute), refcon.sentencePosition]
                enc_feat = [self.encode(v) for v in feat]
                extargsent = list(set(t.sentenceId for t in rel.extArgTokens))[0] # taking first sent only in case ext arg is spread over multiple sentences
                sentposlabel = extargsent - refcon.sentenceId
                X_train_pos.append(enc_feat)
                y_train_pos.append(sentposlabel)
                if sentposlabel == 0:
                    X_train_samesent.append(enc_feat)
                    if rel.extArgTokens[-1].tokenId < refcon.tokenId:
                        y_train_samesent.append(-1)
                    else:
                        y_train_samesent.append(1)

                        
                
        self.sentposclf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)
        self.samesentclf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)

        self.sentposclf.fit(X_train_pos, y_train_pos)
        self.samesentclf.fit(X_train_samesent, y_train_samesent)
        """
        import numpy
        from sklearn.model_selection import cross_val_score
        print('sentpos avg acc:', numpy.mean(cross_val_score(self.sentposclf, X_train_pos, y_train_pos, cv=10)))
        print('samesent avg acc:', numpy.mean(cross_val_score(self.samesentclf, X_train_samesent, y_train_samesent, cv=10)))
        """
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write('INFO: Done training explicit argument extractor...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes), int(seconds)))


        
    def predict(self, relations, sents, tokens):

        for rel in relations:
            connective = ' '.join([x.token for x in rel.connective])
            refcon = rel.connective[0]
            sentence = refcon.fullSentence
            tree = self.lexparser.parse(re.sub('\)', ']', re.sub('\(', '[', sentence)).split())
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break # always taking the first, assuming that this is the best scoring tree.
            ptree = ParentedTree.convert(ptree)
            postag = ptree.pos()[refcon.sentenceTokenId][1]
            ln = 'SOS' if refcon.sentenceTokenId == 0 else ptree.pos()[refcon.sentenceTokenId-1][0]
            rn = 'EOS' if refcon.sentenceTokenId == len(ptree.pos()) else ptree.pos()[refcon.sentenceTokenId+1][0]
            lnpos = 'SOS' if refcon.sentenceTokenId == 0 else ptree.pos()[refcon.sentenceTokenId-1][1]
            rnpos = 'EOS' if refcon.sentenceTokenId == len(ptree.pos()) else ptree.pos()[refcon.sentenceTokenId+1][1]
            nodePosition = ptree.leaf_treeposition(refcon.sentenceTokenId)
            parent = ptree[nodePosition[:-1]].parent()
            rootroute = utils.getPathToRoot(parent, [])
            feat = [connective, postag, ln, rn, lnpos, rnpos, '-'.join(rootroute), refcon.sentenceTokenId]
            enc_feat = [self.encode(v) for v in feat]

            relative_position = self.sentposclf.predict(numpy.array(enc_feat).reshape(1, -1))
            print('connective:', connective)
            print('relpos:', relative_position)
            # GOT SENT POS; IF 0, PREDICT WITH SAMESENT.
            # then get ext arg
            # then get int arg
            # then done

            
