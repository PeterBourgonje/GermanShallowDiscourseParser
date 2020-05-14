import os
import re
import sys
from nltk.tree import ParentedTree

def listfolder(folder):
    return [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder)]

def addAnnotationLayerToDict(flist, fdict, annname):
    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict

def contains_sublist(lst, sublst): 
    n = len(sublst)
    return any((sublst == lst[i:i+n]) for i in range(len(lst)-n+1))

def contains_discont_sublist(lst, sublst):
    stripped = [x for x in lst if x in sublst]
    if compressRoute(stripped) == sublst:
        return True

def get_match_positions(lst, sublst):
    return [i for i, j in enumerate(lst) if lst[i:i+len(sublst)] == sublst]

def get_discont_match_positions(lst, sublst):
    # known bug: this function only returns the first match. Chances of multiple discontinuous connectives in one sentence are low though.
    positions = []
    for item in sublst:
        positions.append(lst.index(item))
    return positions

def iscontinuous(l):
    for i in range(len(l)-1):
        if l[i+1] - l[i] > 1:
            return False
    return True

def getPathToRoot(ptree, route):
    if ptree.parent() == None:
        route.append(ptree.label())
        return route
    else:
        route.append(ptree.label())
        getPathToRoot(ptree.parent(), route)
    return route

def compressRoute(r): # filtering out adjacent identical tags
    delVal = "__DELETE__"
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            r[i+1] = delVal
    return [x for x in r if x != delVal]

def get_parent(pt, i):
    nodePosition = pt.leaf_treeposition(i)
    parent = pt[nodePosition[:-1]].parent()
    return parent

def find_lowest_embracing_node(node, reftoken):
    if not contains_sublist(node.leaves(), list(reftoken)):
        while node.parent(): # recurse till rootnode
            return find_lowest_embracing_node(node.parent(), reftoken)
    return node

def find_lowest_embracing_node_discont(node, reftoken):
    if not contains_discont_sublist(node.leaves(), list(reftoken)):
        while node.parent(): # recurse till rootnode
            return find_lowest_embracing_node(node.parent(), reftoken)
    return node

def get_parent_phrase_plus_phrase_after_comma(tree, pos, labels, ct):
    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodePosition = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            children = pt[nodePosition[:1]]
            labelnode = climb_tree(tree, nodePosition, labels)
            predictedIntArgTokens = labelnode.leaves()
            rs = labelnode.right_sibling()
            if rs:
                if rs.label() == '$,':
                    predictedIntArgTokens += rs.right_sibling().leaves()
            return predictedIntArgTokens

def climb_tree(tree, nodePosition, labels):
    pTree = ParentedTree.convert(tree)
    parent = pTree[nodePosition[:-1]].parent()
    if parent.label() in labels or parent.label() == 'ROOT':
        return parent
        
    else:
        return climb_tree(tree, nodePosition[:-1], labels)
    
def get_right_sibling(tree, pos, ct):
    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodepos = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            rs = pt[nodepos[:-1]].right_sibling()
            if rs:
                if rs.label() == 'S':
                    return rs.leaves()
                else:
                    parent = pt[nodepos[:-1]].parent()
                    # assuming that there are no duplicates of the connective anymore at this level of detail:
                    leaves = parent.leaves()
                    connindex = leaves.index(ct.token)
                    remainder = [xj for xi, xj in enumerate(leaves) if xi >= connindex]
                    return remainder
            else:
                parent = pt[nodepos[:-1]].parent()
                right_sibling = parent.right_sibling()
                leaves = parent.leaves()
                leaves = leaves + right_sibling.leaves()
                connindex = leaves.index(ct.token)
                remainder = [xj for xi, xj in enumerate(leaves) if xi >= connindex]
                return remainder

def get_parent_phrase(tree, pos, labels, ct):
    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodePosition = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            children = pt[nodePosition[:1]]
            labelnode = climb_tree(tree, nodePosition, labels)
            predictedIntArgTokens = labelnode.leaves()
            return predictedIntArgTokens
            

def matchPlainTokensWithIds(plain_tokens, id2Token):

    _sorted = sorted(id2Token.items(), key = lambda x: x[0])
    for i, pair in enumerate(_sorted):
        if plain_tokens[0] == pair[1]:
            if len(plain_tokens) > 1:
                if plain_tokens[1] == _sorted[i+1][1] or plain_tokens[1] == _sorted[i+2][1] and _sorted[i+1][1] in '()': # second condition to accomodate stupid bracket deletion requirement
                    anchor = _sorted[0][0]
                    if len(plain_tokens) == 2:
                        anchor = _sorted[i][0]
                    elif plain_tokens[2] == _sorted[i+2][1] or plain_tokens[2] == _sorted[i+3][1] and plain_tokens[1] in '()':
                        anchor = _sorted[i][0]
                    else:
                        pass
                    ret = []
                    for i in range(anchor, anchor + len(plain_tokens)):
                        ret.append(i)
                    return ret
            else: # in case of VP args it can sometimes be only 1 or 2 tokens I guess
                anchor = _sorted[0][0]
                ret = []
                for i in range(anchor, anchor + len(plain_tokens)):
                    ret.append(i)
                return ret      

def left_siblings(st, l):
    if st.left_sibling():
        l.append(st.left_sibling())
        l = left_siblings(st.left_sibling(), l)
    return l

def right_siblings(st, l):
    if st.right_sibling():
        l.append(st.right_sibling())
        l = right_siblings(st.right_sibling(), l)
    return l

def getLeaveNr(refcon, tree):
    leavenr = 0
    if tree.leaves()[refcon.sentenceTokenId] == refcon.token: # this is not necessarily the case, since brackets are deleted for parsing
        leavenr = refcon.sentenceTokenId
    else:
        bracketnr = 0
        restored = False
        if re.search('[\(\)]', refcon.fullSentence):
            for char in ' '.join(refcon.fullSentence.split()[:refcon.sentenceTokenId]):
                if char == '(' or char == ')':
                    bracketnr += 1
        if bracketnr:
            if tree.leaves()[refcon.sentenceTokenId-bracketnr] == refcon.token:
                restored = True
                leavnr = refcon.sentenceTokenId-bracketnr
        assert restored # if this is not the case, something else went wrong locating the token in the tree (any other special characters NLTK treats differently during parsing?)
    return leavenr
    
