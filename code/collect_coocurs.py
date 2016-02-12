# -*- coding: utf-8 -*-
'''
Created on Jan 18, 2016

@author: hugohrosa
'''

import urllib,os, sys
import nltk
import json
import unicodecsv
import cPickle
from ipdb import set_trace
import time

def PoI(entity, poi, err):
    try:       
        sys.stdout.write("\rlookup: %s               " % entity)
        sys.stdout.flush()
        url = 'http://services.sapo.pt/InformationRetrieval/Verbetes/WhoIs?name='+entity.encode('utf-8')
        response = urllib.urlopen(url)
        content = response.read()
        data = json.loads(content)
        if len(data.values()[0])==0:
            return ''
        else:
            for x in data['verbetes']:
                if x['officialName'] in poi:
                    return ''
            return data['verbetes'][0]['officialName']
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print 'Decoding JSON has failed'
        err+=1
        pass


data = open('DATA/data_all.csv', 'rU')
reader = unicodecsv.DictReader(data, encoding = 'utf-8', delimiter = '\t')#, 
with open("Models/tagger.pkl") as fid:
    tagger_fast = cPickle.load(fid)

url_verbetes = "http://services.sapo.pt/InformationRetrieval/Verbetes/GetCoOccurrences?name={0}&begin_date=2015-01-01&end_date=2015-12-31"

err = 0
co_occurs = {}
json_errors = []
sent_id = 0
prev_requests = {}
t0 = time.time()
skip_next=False
for row in reader:
    if int(row['num_de_anotadores_total'])==1:            
        if sent_id > 100:
            print "bailed!"
            break
        sent_id+=1
        
        #POS tagging
        sent_tagged = tagger_fast.tag( nltk.word_tokenize(row['texto']))
        sent_tagged = [w for w in sent_tagged if w[0] not in unicode(nltk.corpus.stopwords.words('portuguese'))]
        # print "SENTENCE:",  sent_tagged            
        poi = []
        for w in sent_tagged:
            token = w[0]
            pos = w[1]
            #if the last entity was a bi-gram skip this word
            if skip_next:
                print "\nskipped: %s" % token
                skip_next=False
                continue
            #ignore the last token!? Why?
            if sent_tagged.index(w)+1 < len(sent_tagged):
                #if the next token is also a NOUN combine the tokens
                n_pos = sent_tagged[sent_tagged.index(w)+1][1]
                if pos in [u'NOUN',u'N'] and n_pos in [u'NOUN',u'N']:
                    token = token + ' ' + sent_tagged[sent_tagged.index(w)+1][0]
                    
                #avoid sending the same requests    
                if token in prev_requests or token in co_occurs: 
                    print "\nignored: %s" % token
                    continue
                
                official_name = PoI(token, poi, err)                                            
                if official_name != '' and token not in co_occurs:
                    sys.stdout.write("\rco-occurs of: %s              " % official_name)
                    sys.stdout.flush()
                    #if the entity is a multiword skip next word
                    if pos in [u'NOUN',u'N'] and n_pos in [u'NOUN',u'N']:
                        skip_next=True
                    print ""
                    try:
                        response = urllib.urlopen(url_verbetes.format(official_name.encode("utf-8")))    
                    except UnicodeDecodeError:
                        print "Unicode Error %s" % official_name
                        continue
                    try:
                        verbetes = json.loads(response.read())
                        if len(verbetes)>0:
                            total_occurences = sum([x.values()[0] for x in verbetes])
                            co_occurs_with = {x.keys()[0]:x.values()[0] for x in verbetes}
                            co_occurs[token] = {"official_name": official_name ,                "verbetes"     : co_occurs_with,
                                                "total_occurences": total_occurences}       
                    except ValueError:
                        print "JSON error: %s" % official_name
                        json_errors.append(official_name)
                else:
                    prev_requests[token] = None



with open("DATA/coocurs.pkl","wb") as fid:
    cPickle.dump(co_occurs, fid)

print "DONE! :) "
print "Took %d minutes" % ((time.time()-t0 )/60)

sys.exit()




