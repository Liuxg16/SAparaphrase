import numpy as np
import sys
import os
from collections import deque


def list_rules(attn_ops, attn_mems, the):
    """
    Given attentions over operators and memories, 
    enumerate all rules and compute the weights for each.
    
    Args:
        attn_ops: a list of num_step vectors, 
                  each vector of length num_operator.
        attn_mems: a list of num_step vectors,
                   with length from 1 to num_step.
        the: early prune by keeping rules with weights > the
    
    Returns:
        a list of (rules, weight) tuples.
        rules is a list of operator ids. 
    
    """
    
    num_step = len(attn_ops)
    paths = {t+1: [] for t in xrange(num_step)}
    paths[0] = [([], 1.)]
    for t in xrange(num_step):
        for m, attn_mem in enumerate(attn_mems[t]):
            for p, w in paths[m]:
                paths[t+1].append((p, w * attn_mem))
        if t < num_step - 1:
            new_paths = []           
            for o, attn_op in enumerate(attn_ops[t]):
                for p, w in paths[t+1]:
                    if w * attn_op > the:
                        new_paths.append((p + [o], w * attn_op))
            paths[t+1] = new_paths
    this_the = min([the], max([w for (_, w) in paths[num_step]]))
    final_paths = filter(lambda x: x[1] >= this_the, paths[num_step])
    final_paths.sort(key=lambda x: x[1], reverse=True)
    
    return final_paths


def print_rules(q_id, rules, parser, query_is_language):
    """
    Print rules by replacing operator ids with operator names
    and formatting as logic rules.
    
    Args:
        q_id: the query id (the head)
        rules: a list of ([operator ids], weight) (the body)
        parser: a dictionary that convert q_id and operator_id to 
                corresponding names
    
    Returns:
        a list of strings, each string is a printed rule
    """
    
    if len(rules) == 0:
        return []
    
    if not query_is_language: 
        query = parser["query"][q_id]
    else:
        query = parser["query"](q_id)
        
    # assume rules are sorted from high to lows
    max_w = rules[0][1]
    # compute normalized weights also    
    rules = [[rule[0], rule[1], rule[1]/max_w] for rule in rules]

    printed_rules = [] 
    for rule, w, w_normalized in rules:
        if len(rule) == 0:
            printed_rules.append(
                "%0.3f (%0.3f)\t%s(B, A) <-- equal(B, A)" 
                % (w, w_normalized, query))
        else:
            lvars = [chr(i + 65) for i in xrange(1 + len(rule))]
            printed_rule = "%0.3f (%0.3f)\t%s(%c, %c) <-- " \
                            % (w, w_normalized, query, lvars[-1], lvars[0]) 
            for i, literal in enumerate(rule):
                if not query_is_language:
                    literal_name = parser["operator"][q_id][literal]
                else:
                    literal_name = parser["operator"][literal]
                printed_rule += "%s(%c, %c), " \
                                % (literal_name, lvars[i+1], lvars[i])
            printed_rules.append(printed_rule[0: -2])
    
    return printed_rules

def findCircleNum(M):
    """
    :type M: List[List[int]]
    :rtype: int
    """
    if not M or not M[0]:
        return 0
    
    n = len(M)

    uf = UnionFind(n)
    for i in xrange(n):
        for j in xrange(n):
            if M[i][j] == 1:
                uf.union(i, j)

    res = set()
    for i in xrange(n):
        for j in xrange(n):
            if M[i][j] == 1:
                uf.compressed_find(j)
                res.add(uf.father[j])
                
    return list(res)
    

def findCircle(M, id):
    """
    :type M: List[List[int]]
	: id: node
    :rtype: int
    """
    if not M or not M[0]:
        return 0
    
    n = len(M)

    uf = UnionFind(n)
    for i in xrange(n):
        for j in xrange(n):
            if M[i][j] == 1:
                uf.union(i, j)
	res = []
	fid = uf.father[id]		
    for i in xrange(n):
		if uf.father[i]==fid:
			res.append(i)
                
    return res
 
def sparseFindCircle(M,n, id):
    """
    :type M: List[(nodei, nodej)]
	:N: number of nodes
	: id: node
    :rtype: list
    """
    if not M or not M[0]:
        return None
    
    uf = UnionFind(n)
    for tup in M:
        uf.union(tup[0], tup[1])
	res = []
	fid = uf.father[id]		
    for i in xrange(n):
		if uf.father[i]==fid:
			res.append(i)
                
    return res
   


class UnionFind:
    def __init__(self, n, M):
	"""
	:type M: List[(nodei, nodej)]
	:N: number of nodes
	"""
        self.father = {}
        self.N = n

        for i in xrange(n):
            self.father[i] = i

        for tup in M:
			self.union(tup[0], tup[1])


    def compressed_find(self, x):
        ancestor = self.father[x]

        while ancestor != self.father[ancestor]:
            ancestor = self.father[ancestor]

        while x != self.father[x]:
            next = self.father[x]
            self.father[x] = ancestor
            x = next

        return ancestor

    def union(self, x, y):
        fa_x = self.compressed_find(x)
        fa_y = self.compressed_find(y)

        if fa_x != fa_y:
            self.father[fa_y] = fa_x

    def sparseFindCircle(self, id):
		"""
		:type M: List[(nodei, nodej)]
		:N: number of nodes
		: id: node
		:rtype: list
		"""
		
		res = []
		fid = self.father[id]		
		for i in xrange(self.N):
			if self.father[i]==fid:
				res.append(i)
					
		return res

    def sparseFindCircles(self, ids):
		"""
		: id: list of nodes
		:rtype: list
		"""
		
		nodes = list(set([self.father[id] for id in ids]))

		res = []
		for i in xrange(self.N):
			if self.father[i] in nodes:
				res.append(i)
					
		return res




def test():
    M = [[0,1],[1,0],[6,5],[5,6],[8,2],[5,7],[2,8],[7,5]]
    uf = UnionFind(9,M)
    print uf.sparseFindCircle(0)
    print uf.sparseFindCircle( 1)
    print uf.sparseFindCircle( 2)
    print uf.sparseFindCircle( 3)
    print uf.sparseFindCircle( 7)

    print uf.sparseFindCircle([3])




if __name__ == "__main__":

	test()
