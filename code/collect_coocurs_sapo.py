# -*- coding: utf-8 -*-
'''
Created on Jan 18, 2016

@author: hugohrosa
'''

import urllib,os, sys, re
import nltk
import json
import unicodecsv
import cPickle
#from ipdb import set_trace
import time
import itertools
from bs4 import BeautifulSoup
import requests
from selenium import webdriver 
import re
import codecs

def getSearchNumHits(url):
    driver = webdriver.PhantomJS()
    try:
        driver.get(url)
        page_html = driver.execute_script('return document.documentElement.innerHTML;')
    except Exception:
        print 'Webdriver Error ' + url
        driver.close()
    soup = BeautifulSoup(page_html, 'html.parser')
    search_results = soup.find('div', {'id': 'resInfo-0'})
    if search_results is not None:
        hits = numbersOnly(search_results.text.partition('(')[0])
        good = codecs.open('html_good.html', 'w', 'utf-8')
        good.write(page_html)
        good.close()
    else:
        print '\t' + url
        bad = codecs.open('html_bad.html', 'w', 'utf-8')
        bad.write(page_html)
        bad.close()
        hits = 0
    driver.quit()
    return hits

def numbersOnly(text):
    return re.sub('[^0-9]','',text)

def PoI(entity, poi, err):
    try:       
        sys.stdout.write("\tlookup: %s\n" % entity)
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

noun_list = [u'NOUN',u'N']
err = 0
co_occurs = {}
json_errors = []
sent_id = 0
prev_requests = {}
t0 = time.time()
skip_next=False
for row in reader:
    entities = []
    if int(row['num_de_anotadores_total'])==1:            
        if sent_id > 0:
            print "\tbailed!"
            break
        sent_id+=1
        
        #remove punctuation and numbers from text string
        text = re.sub(ur"\p{P}+", '', row['texto'])
        print text
        
        #POS tagging
        sent_tagged = tagger_fast.tag(nltk.word_tokenize(text))
        sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
        # print "SENTENCE:",  sent_tagged            
        poi = []
        skip_next=False
        for w in sent_tagged:
            token = w[0]
            pos = w[1]
            #if the token only has one character, it doesn't interest us
            if len(token)<2:
                continue
            #if the last entity was a bi-gram skip this word
            if skip_next:
                print "\tskipped: %s" % token
                skip_next=False
                continue
            #ignore the last token!? Why?
            
            if pos in noun_list:
                if sent_tagged.index(w)+1 < len(sent_tagged):
                #if the next token is also a NOUN combine the tokens
                    n_pos = sent_tagged[sent_tagged.index(w)+1][1]
                    if n_pos in noun_list:
                        token_cmp = token + ' ' + sent_tagged[sent_tagged.index(w)+1][0]
                        entity=None
                        while entity is None:
                            entity = PoI(token, poi, err)
                        entity = PoI(token_cmp, poi, err)
                        if entity != '' and token not in co_occurs:
                            #print entity
                            skip_next=True
                            token = entity
                print '\tadded: ' + token
                entities.append(token)
                    
            #avoid sending the same requests    
            if token in prev_requests or token in co_occurs or pos not in [u'NOUN',u'N']: 
                print "\tignored: %s" % token
                continue
            
        print '\t' + str(entities)
        
        #make combinations of entities for Sapo search engine
        entity_pairs = itertools.combinations(entities, 2)               
        for e_1, e_2 in entity_pairs:
            e_a = e_1.encode('utf-8')
            e_b = e_2.encode('utf-8')
            url_sapo_co_ab = 'http://www.sapo.pt/pesquisa?q="'+e_a+'" + "'+e_b+'"'
            url_sapo_co_ba = 'http://www.sapo.pt/pesquisa?q="'+e_b+'" + "'+e_a+'"'
            url_sapo_a = 'http://www.sapo.pt/pesquisa?q="'+e_a+'"'
            url_sapo_b = 'http://www.sapo.pt/pesquisa?q="'+e_b+'"'
            co_hits_ab = getSearchNumHits(url_sapo_co_ab)
            co_hits_ba = getSearchNumHits(url_sapo_co_ba)
            total_occurences_a = getSearchNumHits(url_sapo_a)
            total_occurences_b = getSearchNumHits(url_sapo_b)
            
            if co_occurs.has_key(e_a):
                co_item = co_occurs[e_a]['verbetes']
                if not co_item.has_key(e_b):
                    co_item[e_b]=co_hits_ab
                co_occurs[e_a]['verbetes']=co_item      
            elif co_occurs.has_key(e_b):
                co_item = co_occurs[e_b]['verbetes']
                if not co_item.has_key(e_a):
                    co_item[e_a]=co_hits_ba
                co_occurs[e_b]['verbetes']=co_item
            else:
                co_item_ab = dict()
                co_item_ba = dict()
                co_item_ab[e_b]=co_hits_ab
                co_item_ba[e_a]=co_hits_ba
                co_occurs[e_a] = {"official_name": e_b, "verbetes": co_item_ab, "total_occurences": total_occurences_a}
                co_occurs[e_b] = {"official_name": e_a, "verbetes": co_item_ba, "total_occurences": total_occurences_b}

print 'sai'
print co_occurs
            
       

# 
# 
# with open("DATA/coocurs_all_sapo.pkl","wb") as fid:
#     cPickle.dump(co_occurs, fid)
# 
# print "DONE! :) "
# print "Took %d minutes" % ((time.time()-t0 )/60)
# 
# sys.exit()




