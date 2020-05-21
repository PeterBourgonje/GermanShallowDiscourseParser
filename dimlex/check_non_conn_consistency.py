#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import lxml.etree


xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments=False)
#tree = lxml.etree.parse('DimLex.xml', parser = xmlParser)
tree = lxml.etree.parse('dimlex.xml', parser = xmlParser)
l = []
root = lxml.etree.Element('dimlex')
for entry in tree.getroot():
    ambiguity = entry.find('ambiguity')
    ncnval = None
    ncn = None
    if ambiguity is not None:
        non_conn_node = ambiguity.find('non_conn')
        ncnval = non_conn_node.text
        ncn = non_conn_node
    
    non_conn_reading = entry.find('non_conn_reading')
    nonconnreadingtext = None
    if non_conn_reading.text == '0':
        new_non_conn_reading = lxml.etree.Element('non_conn_reading')
        entry.replace(non_conn_reading, new_non_conn_reading)
    elif list(non_conn_reading):
        if ncnval == '0':
            new_ncn = lxml.etree.Element('non_conn')
            new_ncn.text = '1'
            ambiguity.replace(ncn, new_ncn)
        
        

    root.append(entry)
doc = lxml.etree.ElementTree(root)
doc.write('temp.xml', xml_declaration=True, encoding='utf-8', pretty_print=True) 

#print('word:', entry.get('word'))
#print('deb:', non_conn_reading.text)
"""
if non_conn_reading is not None:
nonconnreadingFound = True
for ex in non_conn_reading:
nonconnreadingtext = ex.text

if ncnval == '0' and nonconnreadingFound:
print('Entry with id "%s" ("%s") has 0 for non_conn_node in ambiguity, yet non_conn_reading example (%s)' % (entry.get('id'), entry.get('word'), nonconnreadingtext))


elif ncnval == '1' and not nonconnreadingFound:
print('aap') # this does not happen...
"""
