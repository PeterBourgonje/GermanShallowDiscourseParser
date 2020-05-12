#!/usr/bin/python3

import os
import sys
import json
import codecs
import configparser
from nltk.parse import stanford
from spacy.lang.de import German

# custom modules
import utils
import ConnectiveClassifier

nlp = German()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

class Parser:

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['parserdir']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['parserdir']
        os.environ['CLASSPATH'] = '%s/stanford-parser.jar' % self.config['lexparser']['parserdir']
        self.lexparser = stanford.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz')

class Token:

    def __init__(self, token, sentenceId, sentenceTokenId):
        self.token = token.text
        self.tokenId = token.i
        self.sentenceId = sentenceId
        self.sentenceTokenId = sentenceTokenId
        self.span = tuple((token.idx, token.idx+len(token.text)))

    def setConnective(self):
        self.isConnective = True

def custom_tokenize(inp):

    # using spacy sentencizer/tokenizer since most(all?) nltk ones replace double quotes (and some other chars: https://www.nltk.org/_modules/nltk/tokenize/treebank.html)
    doc = nlp(inp)
    sents = {}
    for si, sent in enumerate(doc.sents):
        tokens = [Token(token, si, ti) for ti, token in enumerate(sent)]
        sents[si] = tokens
        
    return sents
    
    

if __name__ == '__main__':


    cc = ConnectiveClassifier.ConnectiveClassifier()
    cc.train() # TODO: currently training on PCC only, allow setting to combine PCC+WN, or train on one of the two only (PCC and WN files should have the same format, so should be no problem)

    inp = 'Wie schwierig es ist, in dieser Region einen Ausbildungsplatz zu finden, haben wir an dieser und anderer Stelle oft und ausführlich bewertet. Trotzdem bemühen sich Unternehmen sowie die Industrie- und Handelskammer Potsdam den Schulabgängern Wege in die Ausbildung aufzuzeigen. Und ein Beispiel mit entweder dies oder das und anstatt dass.'

    sents = custom_tokenize(inp)
    cc.predict(sents)

    for sid in sents:
        print('sid:', sid)
        print('sent:', ' '.join([x.token for x in sents[sid]]))
        print('connective tokenIds:', [(x.tokenId, x.token) for x in sents[sid] if hasattr(x, 'isConnective')])
    # OK, NOTE TO SELF: connective prediction done. Note that from now on, consecutive isConnective tokens have to be treated as one connective (make sure this is the case, as they are not marked as belonging together, other than by the fact that they are consecutive (filtering on cc.predict makes sure this is the case, think this is a reasonable assumption, it does mean however that compounded connectives are always treated as one connective and cannot have different args))

    # move on with args, eval function can come last, to make sure all work with the same train/test file split...
    
