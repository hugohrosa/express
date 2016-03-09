#-*- coding: utf-8 -*-
import cPickle
import json
import pprint
#from ipdb import set_trace
import itertools
import nltk
import numpy as np
import sys
import unicodecsv
import urllib
import os
import string

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
    except:
        print 'EXCEPT ' +a + ' ' + b +  ' no cooccurences '
        return -1
    #PMI
    if p_a_b ==0:
        #print 'ZERO OCCURENCES '+ a + ' ' + b +  ' no cooccurences '
        return -1
    
    if p_a == 0:
        #print 'PROBABILITY ZERO '+ a
        return -1
    if p_b == 0:
        #print 'PROBABILITY ZERO '+ b
        return -1
    
    pmi = np.log(p_a_b/(p_a*p_b))
    #Normalized PMI
    npmi = pmi/-np.log(p_a_b)
    
    #print a + ',' + b + '\t' + str(npmi)
    return npmi

#open co-occurrences dictionary
with open("DATA/_dispair_cooccurs_sapo.pkl","rb") as fid:
    co_occurs = cPickle.load(fid)

#open POS tagger 
with open("Models/tagger.pkl") as fid:
    tagger_fast = cPickle.load(fid)
    
with open("DATA/_dispair_sentences_sapo.pkl","rb") as fid:
    sent_entities = cPickle.load(fid)
pmi_threshold_list = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
#pmi_threshold_list = [-0.1]

print 'Treshold\tIMP\tNoIMP\tTP\tFP\tTN\tFN\tPrecision\tRecall\tF-Measure'
try:
    for pmi_threshold in pmi_threshold_list:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        n_imp = 0
        n_non_imp = 0
        for key,value in sent_entities.iteritems():
            imp = False
            entities = value['entities']                                
        #entities = ['vítor gaspar', 'cristiano ronaldo', 'governo', 'portugal']
            pairwise_pmis = []
            if len(entities)>=2:
                entity_pairs = itertools.combinations(entities, 2)               
                for e_a, e_b in entity_pairs:
                    pairwise_pmis.append(pmi(e_a.encode('utf-8'), e_b.encode('utf-8'), co_occurs))
                    #pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
            
            #if there is at least one PMI lower than the treshold classify as imparity
            if len(entities) >= 2 and np.min(pairwise_pmis) < pmi_threshold:
                imp=True
                n_imp += 1
            else:
                imp=False
                n_non_imp+=1
             
            if imp and int(value['imparity'])==1:
                tp+=1
            elif imp and int(value['imparity'])==0:
                fp+=1
            elif not imp and int(value['imparity'])==1:
                fn+=1
            elif not imp and int(value['imparity'])==0:
                tn+=1
         
                 
        #print "IMP: %d | NO IMP: %d" % (n_imp,n_non_imp)
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        fmeasure = 2 * ((precision * recall) / float(precision + recall))
#         print 'TP ' + str(tp)
#         print 'FP ' + str(fp)
#         print 'TN ' + str(tn)
#         print 'FN ' + str(fn) 
#         print precision
#         print recall
#         print fmeasure
        print str(pmi_threshold)+'\t'+str(n_imp)+'\t'+str(n_non_imp)+'\t'+ str(tp)+'\t'+str(fp)+'\t'+str(tn)+'\t'+str(fn)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(fmeasure)
    
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print 'ERROR ' + str(exc_type) + ',' + str(exc_obj) + ',' +fname+',' + str(exc_tb.tb_lineno)+'\n'
        