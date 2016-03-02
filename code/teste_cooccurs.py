# -*- coding: utf-8 -*-
'''
Created on Mar 2, 2016

@author: hugohrosa
'''
import cPickle
import numpy as np
import itertools

def pmi(a, b, co_occurs):

    # print "PMI(%s, %s)" % (a,b)
    #This is overcounting the occurrences because if A co-occurs with B, then B also co-occurs with A 
    # set_trace()
    total_occurs = sum([x['total_occurences'] for x in co_occurs.values() if x is not None])
    
    try:
        #P(a)
        p_a = co_occurs[a]["total_occurences"]*1.0 / total_occurs
        #P(b)
        p_b = co_occurs[b]["total_occurences"]*1.0 / total_occurs
        #Note: the co-occurrence data is indexed by the token found on text
        #whereas the co-occurence data in verbetes is indexed by official_name    
        b_official_name = co_occurs[b]["official_name"]
        #P(a,b)
        p_a_b = co_occurs[a]["verbetes"][b_official_name]*1.0 / total_occurs
    except KeyError:
        return -2
    #PMI    
    pmi = np.log(p_a_b/(p_a*p_b))
    #Normalized PMI
    npmi = pmi/-np.log(p_a_b)
    
    return npmi

#open co-occurrences dictionary
with open("DATA/coocurs_all_sapo.pkl","rb") as fid:
    co_occurs = cPickle.load(fid)

entities = ['cristiano ronaldo', 'vítor gaspar', 'futebol', 'antónio costa','finanças']

pairwise_pmis = []
if len(entities)>=2:
    # print "\nPOI: ", poi
    entity_pairs = itertools.combinations(entities, 2)               
    for e_a, e_b in entity_pairs:
        print e_a + ',' + e_b + ' ' + str(pmi(e_a, e_b, co_occurs))
