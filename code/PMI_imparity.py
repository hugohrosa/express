#-*- coding: utf-8 -*-
import cPickle
import json
import pprint
from ipdb import set_trace
import itertools
import nltk
import numpy as np
import sys
import unicodecsv
import urllib
import os

def pmi(a, b, co_occurs):

    # print "PMI(%s, %s)" % (a,b)
    #This is overcounting the occurrences because if A co-occurs with B, then B also co-occurs with A 
    # set_trace()
    total_occurs = sum([x['total_occurences'] for x in co_occurs.values() if x is not None])
    
    #P(a)
    p_a = co_occurs[a]["total_occurences"]*1.0 / total_occurs
    #P(b)
    p_b = co_occurs[b]["total_occurences"]*1.0 / total_occurs
    #Note: the co-occurrence data is indexed by the token found on text
    #whereas the co-occurence data in verbetes is indexed by official_name    
    b_official_name = co_occurs[b]["official_name"]
    try:
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
with open("DATA/coocurs.pkl","rb") as fid:
    co_occurs = cPickle.load(fid)

#open POS tagger 
with open("Models/tagger.pkl") as fid:
    tagger_fast = cPickle.load(fid)

# try:
input = open('DATA/data_all.csv', 'rU')
reader = unicodecsv.DictReader(input, encoding = 'utf-8', delimiter = '\t')#, fieldnames = fieldnames ) 
tp = 0
fp = 0
tn = 0
fn = 0
total = 0 
err=0
pmi_threshold = - 0.2
n_imp = 0
n_non_imp = 0
for row in reader:
    if int(row['num_de_anotadores_total'])==1:
    #if True:            
        # print row['texto'] + ' ' + row['Imparidade']
        total+=1            
        sent_tagged = tagger_fast.tag( nltk.word_tokenize(row['texto']))
        sent_tagged = [w for w in sent_tagged if w[0] not in unicode(nltk.corpus.stopwords.words('portuguese'))]
        # print "SENTENCE:",  sent_tagged            
        poi = []
        for w in sent_tagged:
            token = w[0]
            pos = w[1]
            #ignore the last token!? Why?
            if sent_tagged.index(w)+1 < len(sent_tagged):
                #if the next token is also a NOUN combine the tokens
                n_pos=sent_tagged[sent_tagged.index(w)+1][1]
                if pos in [u'NOUN',u'N'] and n_pos in [u'NOUN',u'N']:
                    token = token + ' ' + sent_tagged[sent_tagged.index(w)+1][0]
                    if token in co_occurs and token not in poi:
                        poi.append(token)
                    
                                    
        pairwise_pmis = []
        if len(poi)>2:
            # print "\nPOI: ", poi
            entity_pairs = itertools.combinations(poi, 2)               
            for e_a, e_b in entity_pairs:
                pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
            # print pairwise_pmis
            # set_trace()
        #if there is at least one PMI lower than the treshold classify as imparity
        if len(poi) > 2 and np.min(pairwise_pmis) < pmi_threshold:
            imp=True
            # print "IMPARIDADE"
        else:
            imp=False
        
        # print imp, ":", int(row['Imparidade'])==1
        if imp and int(row['Imparidade'])==1:
            tp+=1
        elif imp and int(row['Imparidade'])==0:
            fp+=1
        elif not imp and int(row['Imparidade'])==1:
            fn+=1
        elif not imp and int(row['Imparidade'])==0:
            tp+=1

        
print "IMP: %d | NO IMP: %d" % (n_imp,n_non_imp)
precision = tp / float(tp + fp)
recall = tp / float(tp + fn)
fmeasure = 2 * ((precision * recall) / float(precision + recall))
print 'TP ' + str(tp)
print 'FP ' + str(fp)
print 'TN ' + str(tn)
print 'FN ' + str(fn) 
print precision
print recall
print fmeasure
# print err       

# except Exception as e:
#     exc_type, exc_obj, exc_tb = sys.exc_info()
#     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#     print 'ERROR ' + str(exc_type) + ',' + str(exc_obj) + ',' +fname+',' + str(exc_tb.tb_lineno)+'\n'
        