import os
import re
import sys

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




        
    
