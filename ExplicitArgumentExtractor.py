import os
import re
import sys
import time
import numpy
import string
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

        
    def getIntArg(self, tree, rel, tokens):

        refcon = rel.connective[0]
        leavenr = utils.getLeaveNr(refcon, tree)
        
        refconpos = tree.leaf_treeposition(leavenr)
        connnode = tree[refconpos[:-1]]
        refcontype = connnode.label()

        plain_tokens = []
        if refcontype == 'KON':
            plain_tokens = utils.get_right_sibling(tree, leavenr, refcon)
        elif refcontype == 'PROAV' or refcontype.startswith('A') or refcontype.startswith('K'):
            plain_tokens = utils.get_parent_phrase_plus_phrase_after_comma(tree, leavenr, ['S', 'CS', 'VP'], refcon)
        else:
            plain_tokens = utils.get_parent_phrase(tree, leavenr, ['S', 'CS', 'VP'], refcon)
        
        # because arg finding uses NLTK tree and positions may be off due to brackets etc., getting back on track with actual Tokens here:
        id2token = {}
        for i3, j3 in enumerate(refcon.fullSentence.split()):
            diff = refcon.sentenceTokenId - i3
            tokenId = refcon.tokenId - diff
            id2token[tokenId] = j3
        intargtokens = utils.matchPlainTokensWithIds(plain_tokens, id2token)
        
        # excluding the connective token(s) from intarg:
        intargtokens = [x for x in intargtokens if not x in [y.tokenId for y in rel.connective]]
        
        # in PCC, intarg is what comes after second item for discont conns. Guess this is a relatively arbitrary decision.
        if not utils.iscontinuous([x.tokenId for x in rel.connective]):
            intargtokens = [x for x in intargtokens if x > rel.connective[-1].tokenId]

        return intargtokens


    def getExtArg(self, tree, rel, tokens, sents):

        refcon = rel.connective[0]
        leavenr = utils.getLeaveNr(refcon, tree)
        
        postag = tree.pos()[refcon.sentenceTokenId][1]
        ln = 'SOS' if refcon.sentenceTokenId == 0 else tree.pos()[refcon.sentenceTokenId-1][0]
        rn = 'EOS' if refcon.sentenceTokenId == len(tree.pos()) else tree.pos()[refcon.sentenceTokenId+1][0]
        lnpos = 'SOS' if refcon.sentenceTokenId == 0 else tree.pos()[refcon.sentenceTokenId-1][1]
        rnpos = 'EOS' if refcon.sentenceTokenId == len(tree.pos()) else tree.pos()[refcon.sentenceTokenId+1][1]
        nodePosition = tree.leaf_treeposition(refcon.sentenceTokenId)
        parent = tree[nodePosition[:-1]].parent()
        rootroute = utils.getPathToRoot(parent, [])
        feat = [' '.join([x.token for x in rel.connective]), postag, ln, rn, lnpos, rnpos, '-'.join(rootroute), refcon.sentenceTokenId]
        enc_feat = [self.encode(v) for v in feat]
        
        relative_position = self.sentposclf.predict(numpy.array(enc_feat).reshape(1, -1))

        # the way this works is that first all tokens from the predicted sentenceId are taken. Then, if extarg is predicted to be in the same sent, more restrictive filtering is applied below. After intarg has been extracted, both are filtered against each other again.
        temparg = []
        targetsid = refcon.sentenceId + relative_position[0]
        if targetsid in sents:
            for token in sents[targetsid]:
                temparg.append(token.tokenId)

        if relative_position == 0:
            samesentpos = self.samesentclf.predict(numpy.array(enc_feat).reshape(1, -1))
            if samesentpos[0] == 1: # extarg is predicted after (ref)conn
                # above, already the entire sent is taken by default, and filtered here, and against intarg later. This sct is a bit more restrictive and only takes the bit to the left/right (depending on samesentclf prediction) in the same S, CS or VP. May want to take out VP later if recall is too low. Or take right sibling (of the parent node) as well.
                labels = ['S', 'CS', 'VP']
                for i, node in enumerate(tree.pos()):
                    if i == leavenr:
                        nodePosition = tree.leaf_treeposition(i)
                        pn = utils.climb_tree(tree, nodePosition, labels)
                        # taking a guess here, which is that inside the parent, the connective appears only once (taking its first index below)... this bit is not guaranteed to be correct
                        parent_tokens = [x[0] for x in pn.pos()]
                        ind = parent_tokens.index(refcon.token)
                        temparg = []
                        for i6, node in enumerate(pn.pos()):
                            if i6 > ind:
                                temparg.append(refcon.tokenId + (i6 - ind))
                                    
            elif samesentpos[0] == -1: # extarg is predicted before (ref)conn
                labels = ['S', 'CS', 'VP']
                for i, node in enumerate(tree.pos()):
                    if i == leavenr:
                        nodePosition = tree.leaf_treeposition(i)
                        pn = utils.climb_tree(tree, nodePosition, labels)
                        parent_tokens = [x[0] for x in pn.pos()]
                        ind = parent_tokens.index(refcon.token)
                        temparg = []
                        for i5, node in enumerate(pn.pos()):
                            if i5 < ind:
                                temparg.append(refcon.tokenId - (ind-i5))

            # adding final punctuation here if it's not part of arg (but the only trailing char in the sentence) (Mind indentation level here: only do this for same sentence cases)
            if sents[refcon.sentenceId][-1].token in string.punctuation:
                temparg.append(sents[refcon.sentenceId][-1].tokenId)


        # in PCC, extarg is what comes before second item for discont conns. Guess this is a relatively arbitrary decision. Hardcoding a fix here:
        if not utils.iscontinuous([x.tokenId for x in rel.connective]):
            temparg = [x for x in range(rel.connective[0].tokenId, rel.connective[-1].tokenId)]

        # filtering out connective tokens (may particularly be needed for phrasal connectives) (and also for discont ones, in which as per line above, connective is included
        temparg = [x for x in temparg if not x in [y.tokenId for y in rel.connective]]


        return temparg
            

    
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


            
            tempextarg = self.getExtArg(ptree, rel, tokens, sents)
            tempintarg = self.getIntArg(ptree, rel, tokens)

            # since generally, intarg tokens are easier to predict than extarg tokens, filtering out any intarg tokens that may be in the extarg list
            intarg = tempintarg
            extarg = [x for x in tempextarg if not x in tempintarg]

            for iat in intarg:
                rel.addIntArgToken(tokens[iat])
            for eat in extarg:
                rel.addExtArgToken(tokens[eat])
