import sys
import re
import os
import lxml.etree
import codecs
from collections import defaultdict

xmlp = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments=True)

class DimLex:

    def __init__(self, entryId, word):
        self.entryId = entryId
        self.word = word
        self.alternativeSpellings = defaultdict(lambda : defaultdict(str))
        self.syncats = []
        self.sense2Probs = defaultdict(float)
        self.surefire = False

    def addAlternativeSpelling(self, alt, singleOrPhrasal, contOrDiscont):
        self.alternativeSpellings[alt][singleOrPhrasal] = contOrDiscont
    def addFunctionalAmbiguityInfo(self, v):
        self.functionalAmbiguity = v
    def setConnectiveReadingProbability(self, p):
        self.connectiveReadingProbability = p
    def addSynCat(self, syncat):
        self.syncats.append(syncat)
    def addSense(self, sense, prob):
        self.sense2Probs[sense] = prob
    def setSurefire(self):
        self.surefire = True


def parseXML(dimlexml):

    tree = lxml.etree.parse(dimlexml, parser=xmlp)
    l = []

    for entry in tree.getroot():
        dl = DimLex(entry.get('id'), entry.get('word'))
        for orth in entry.find('orths').findall('orth'):
            text = ' '.join([x.text for x in orth.findall('part')])
            t1 = 'single' if set([x.get('type') for x in orth.findall('part')]) else 'phrasal'
            t2 = orth.get('type') # cont or discont
            dl.addAlternativeSpelling(text, t1, t2)
            
        
        ambiguity = entry.find('ambiguity')
        non_connNode = None
        if ambiguity is not None:
            non_connNode = ambiguity.find('non_conn')
        if non_connNode is not None:
            dl.addFunctionalAmbiguityInfo(non_connNode.text)
            if non_connNode.text == '1':
                if 'freq' in non_connNode.attrib and 'anno_N' in non_connNode.attrib:
                    p = 1 - (float(non_connNode.get('freq')) / float(non_connNode.get('anno_N')))
                    dl.setConnectiveReadingProbability(p)
            else:
                dl.setSurefire()
                    
        syns = entry.findall('syn')
        for syn in syns:
            dl.addSynCat(syn.find('cat').text)
            for sem in syn.findall('sem'):
                for sense in sem:
                    freq = sense.get('freq')
                    anno = sense.get('anno_N')
                    if not freq == '0' and not freq == '' and not anno == '0' and not anno == '' and not anno == None:
                        ##pdtb3sense = sense.get('pdtb3_relation sense') # bug in lxml due to whitespace in attrib name, or by design?
                        pdtb3sense = sense.get('sense')
                        prob = 1 - (float(freq) / float(anno))
                        dl.addSense(pdtb3sense, prob)
                    elif freq == None and anno == None:
                        dl.addSense(sense.get('sense'), 0)

        l.append(dl)

    return l

